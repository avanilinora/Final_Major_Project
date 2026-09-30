"""
utils/dataset_rr.py
====================
Adds R-R interval features to the existing beat pipeline. New file — nothing in
dataset_stgnn.py, model_stgnn.py or train_stgnn.py is modified; ECGBeatDataset
is subclassed, not edited.

Why
---
Four of the five classes are defined by rate or rhythm regularity, not by
waveform shape:

    SB    < 60 bpm          } identical P-QRS-T morphology;
    SR    60-100 bpm        } the ONLY difference is R-R interval
    ST    > 100 bpm         }
    AFIB  irregular R-R     (irregularity, not rate)
    SVT   > 100 bpm, regular, narrow

The model currently sees R-peak-centred 400-sample windows and nothing else.
Every window is aligned on its R-peak, so the time BETWEEN beats — the exact
quantity the class definitions turn on — is thrown away. The BiGRU can only
infer rate indirectly from beat ordering.

After calibration the dominant remaining error is SB<->SR (99 of 299), which is
purely a 60 bpm threshold decision. Rate-boundary confusions account for ~129
of the 299.

preprocessing_physionet_stgnn.py already saves `r_peaks` in every .npz; the
current dataset just never loads it. This file uses it.

Features, per beat (6)
----------------------
    0  prev_rr    interval to the previous R-peak, normalised
    1  next_rr    interval to the next R-peak, normalised
    2  log_ratio  log(prev_rr / next_rr) -- 0 means regular, large means the
                  beat sits in an irregular sequence. This is the AFIB cue.
    3  median_rr  record-level median interval, normalised -- the rate cue that
                  separates SB / SR / ST. Median not mean, because a couple of
                  missed R-peaks would otherwise drag the rate estimate.
    4  cv_rr      std/mean of the record's intervals -- record-level
                  irregularity. Regular rhythms sit near 0, AFIB well above.
    5  hr         heart rate in bpm from median_rr, normalised. Redundant with
                  feature 3 by construction, but gives the network the clinical
                  quantity directly instead of asking it to learn a reciprocal.

Normalisation uses FIXED constants, not statistics computed over the split, so
there is no train/test leakage through the feature scaling and inference on a
single new recording is well defined.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config_physionet import PhysioNetConfig as PNConfig
from dataset_stgnn import ECGBeatDataset

N_RR_FEATURES = 6

# Fixed normalisation constants — see module docstring.
RR_CENTER = 0.8      # seconds, ~75 bpm
RR_SCALE  = 0.4
HR_CENTER = 75.0     # bpm
HR_SCALE  = 40.0
RR_MIN, RR_MAX = 0.20, 3.00   # plausible interval range (200 bpm .. 20 bpm)


def compute_rr_features(r_peaks: np.ndarray, fs: int = PNConfig.SAMPLING_FREQ,
                        n_beats: Optional[int] = None) -> np.ndarray:
    """
    r_peaks : (n_beats,) sample indices of the retained R-peaks
    returns : (n_beats, 6) float32
    """
    r_peaks = np.asarray(r_peaks, dtype=np.float64).ravel()
    n = len(r_peaks) if n_beats is None else n_beats

    if len(r_peaks) < 2:
        # Can't form an interval. Fall back to the population centre rather
        # than zeros, so the feature reads "unremarkable" instead of "0 s".
        out = np.zeros((n, N_RR_FEATURES), dtype=np.float32)
        out[:, 5] = 0.0
        return out

    rr = np.diff(r_peaks) / float(fs)                 # (n_beats-1,) seconds
    rr = np.clip(rr, RR_MIN, RR_MAX)

    prev_rr = np.concatenate([rr[:1], rr])            # (n_beats,)
    next_rr = np.concatenate([rr, rr[-1:]])
    prev_rr, next_rr = prev_rr[:n], next_rr[:n]

    median_rr = float(np.median(rr))
    cv_rr = float(np.std(rr) / max(np.mean(rr), 1e-6))
    hr = 60.0 / max(median_rr, 1e-6)

    feats = np.empty((n, N_RR_FEATURES), dtype=np.float32)
    feats[:, 0] = (prev_rr - RR_CENTER) / RR_SCALE
    feats[:, 1] = (next_rr - RR_CENTER) / RR_SCALE
    feats[:, 2] = np.log(prev_rr / np.maximum(next_rr, 1e-6))
    feats[:, 3] = (median_rr - RR_CENTER) / RR_SCALE
    feats[:, 4] = cv_rr
    feats[:, 5] = (hr - HR_CENTER) / HR_SCALE
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


class ECGBeatRRDataset(ECGBeatDataset):
    """ECGBeatDataset plus an `rr` tensor of shape (n_beats, 6)."""

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        npz = np.load(rec["npz_path"])
        beats = torch.from_numpy(npz["beats"]).float()          # (n_beats, 400, 12)
        rr = compute_rr_features(npz["r_peaks"], n_beats=beats.shape[0])
        return {
            "beats": beats,
            "rr": torch.from_numpy(rr).float(),                 # (n_beats, 6)
            "label": torch.from_numpy(npz["label"]).float(),
            "lead_valid": torch.from_numpy(npz["lead_valid"]).bool(),
            "record_id": rec["record_id"],
        }


def identity_collate_rr(batch: List[Dict]) -> Dict:
    assert len(batch) == 1, "identity_collate_rr only supports batch_size=1"
    return batch[0]


def build_rr_loader(split: str, shuffle: bool = False, limit: Optional[int] = None,
                    num_workers: int = 4, seed: int = PNConfig.RANDOM_SEED) -> DataLoader:
    index_path = os.path.join(PNConfig.PROCESSED_PATH, f"index_{split}.json")
    dataset = ECGBeatRRDataset(index_path)
    if limit is not None and limit < len(dataset):
        rng = np.random.RandomState(seed)
        dataset = Subset(dataset, sorted(rng.choice(len(dataset), limit, replace=False).tolist()))
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=identity_collate_rr,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


# ===========================================================================
# Diagnostic: is the hypothesis actually true?
# ===========================================================================

if __name__ == "__main__":
    import collections

    PNConfig.load_label_vocab()
    class_names = PNConfig.CLASS_NAMES
    N_SAMPLE = 3000

    print("=" * 78)
    print("  R-R feature diagnostic — can rate alone separate SB / SR / ST?")
    print("=" * 78)

    with open(os.path.join(PNConfig.PROCESSED_PATH, "index_train.json")) as fh:
        records = json.load(fh)
    rng = np.random.RandomState(PNConfig.RANDOM_SEED)
    sample = [records[i] for i in rng.choice(len(records),
                                             min(N_SAMPLE, len(records)), replace=False)]

    by_class = collections.defaultdict(list)
    for rec in sample:
        lab = np.asarray(rec["label"])
        if lab.sum() != 1:
            continue
        npz = np.load(rec["npz_path"])
        f = compute_rr_features(npz["r_peaks"], n_beats=int(rec["n_beats"]))
        hr = f[0, 5] * HR_SCALE + HR_CENTER
        by_class[class_names[int(lab.argmax())]].append((hr, float(f[0, 4])))

    print(f"\n  Heart rate by class (n={sum(len(v) for v in by_class.values()):,} records)")
    print(f"  {'class':<6}{'n':>7}{'HR p25':>9}{'HR med':>9}{'HR p75':>9}"
          f"{'CV med':>9}{'CV p90':>9}")
    print(f"  {'-'*58}")
    for name in class_names:
        vals = by_class.get(name, [])
        if not vals:
            continue
        hrs = np.array([v[0] for v in vals])
        cvs = np.array([v[1] for v in vals])
        print(f"  {name:<6}{len(vals):>7d}{np.percentile(hrs,25):>9.1f}"
              f"{np.median(hrs):>9.1f}{np.percentile(hrs,75):>9.1f}"
              f"{np.median(cvs):>9.3f}{np.percentile(cvs,90):>9.3f}")

    # Textbook rule, no learning at all: HR<60 -> SB, 60-100 -> SR, >100 -> ST.
    # Whatever this scores is roughly the free accuracy the network is
    # currently NOT getting, because it cannot see rate.
    correct = total = 0
    per_class = collections.Counter()
    per_class_n = collections.Counter()
    for name in ("SB", "SR", "ST"):
        for hr, _ in by_class.get(name, []):
            pred = "SB" if hr < 60 else ("SR" if hr <= 100 else "ST")
            correct += pred == name
            per_class[name] += pred == name
            per_class_n[name] += 1
            total += 1

    print(f"\n  Textbook rate rule (HR<60=SB, 60-100=SR, >100=ST), no model:")
    for name in ("SB", "SR", "ST"):
        if per_class_n[name]:
            print(f"    {name:<4} {per_class[name]:>5d}/{per_class_n[name]:<5d}"
                  f"  = {per_class[name]/per_class_n[name]:.4f}")
    if total:
        print(f"    overall on SB/SR/ST: {correct}/{total} = {correct/total:.4f}")

    print(f"\n  AFIB irregularity check (CV of R-R):")
    for name in class_names:
        vals = by_class.get(name, [])
        if vals:
            cvs = np.array([v[1] for v in vals])
            print(f"    {name:<6} median CV {np.median(cvs):.3f}")

    loader = build_rr_loader("train", limit=4, num_workers=0)
    b = next(iter(loader))
    print(f"\n  Loader check — {b['record_id']}")
    print(f"    beats {tuple(b['beats'].shape)}   rr {tuple(b['rr'].shape)}")
    print(f"    rr[0] = {np.round(b['rr'][0].numpy(), 3)}")
    print("\nOK")
