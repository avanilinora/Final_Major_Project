"""
utils/dataset_stgnn.py
=======================
PyTorch Dataset + fixed adjacency matrix for the STGNN pipeline. Reads the
index_{split}.json + beats/*.npz files written by
preprocessing_physionet_stgnn.py.

Batching: batch_size=1, one recording (one variable-length beat sequence)
per batch. Avoids pad/mask correctness risk while beat counts are still
small (~3-40/recording). Revisit (bucketed batching or pad+mask) only if
training throughput becomes the actual bottleneck.
"""

import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config_physionet import PhysioNetConfig as PNConfig


# ===========================================================================
# Fixed anatomical adjacency (12x12)
# ===========================================================================

LEAD_ORDER = PNConfig.LEAD_NAMES  # ["I","II","III","aVR","aVL","aVF","V1",...,"V6"]
LEAD_IDX = {name: i for i, name in enumerate(LEAD_ORDER)}


def build_anatomical_adjacency(symmetric: bool = True) -> torch.Tensor:
    """
    Fixed 12x12 adjacency encoding standard 12-lead electrical/anatomical
    relationships. Weighted, not binary — closer/more directly-related
    leads get higher weight. This is the anatomical-prior half of the
    hybrid graph; GAT's learned edge refinement adjusts this per beat.

    Edge groups
    -----------
    - Einthoven triangle (I, II, III): mutually related by Einthoven's law
      (I + III = II) -> strong edges among the three.
    - Goldberger augmented leads (aVR, aVL, aVF): each is a linear
      combination of two limb leads -> edge to each contributing lead.
        aVR ~ -(I + II)/2   -> edges to I, II
        aVL ~ (I - III)/2   -> edges to I, III
        aVF ~ (II + III)/2  -> edges to II, III
    - Precordial chain (V1..V6): adjacent leads are physically adjacent
      electrode positions on the chest -> chain edges V1-V2-V3-V4-V5-V6,
      plus weaker next-nearest-neighbour edges.
    - Precordial-to-limb: V1/V2 are anatomically closer to the right
      side/RV (weak edge to aVR); V5/V6 closer to the lateral/left side
      (weak edge to aVL, I).
    """
    n = len(LEAD_ORDER)
    A = np.zeros((n, n), dtype=np.float32)

    def link(a, b, w):
        A[LEAD_IDX[a], LEAD_IDX[b]] = w
        if symmetric:
            A[LEAD_IDX[b], LEAD_IDX[a]] = w

    # Einthoven triangle
    link("I", "II", 1.0)
    link("II", "III", 1.0)
    link("I", "III", 1.0)

    # Goldberger augmented leads -> their source limb leads
    link("aVR", "I", 0.7)
    link("aVR", "II", 0.7)
    link("aVL", "I", 0.7)
    link("aVL", "III", 0.7)
    link("aVF", "II", 0.7)
    link("aVF", "III", 0.7)

    # Precordial chain
    precordial = ["V1", "V2", "V3", "V4", "V5", "V6"]
    for a, b in zip(precordial, precordial[1:]):
        link(a, b, 1.0)
    for a, b in zip(precordial, precordial[2:]):  # next-nearest-neighbour
        link(a, b, 0.4)

    # Precordial-to-limb laterality links
    link("V1", "aVR", 0.3)
    link("V2", "aVR", 0.3)
    link("V5", "aVL", 0.3)
    link("V6", "aVL", 0.3)
    link("V5", "I", 0.3)
    link("V6", "I", 0.3)

    np.fill_diagonal(A, 1.0)  # self-loops — GAT wants these for stability

    return torch.from_numpy(A)


# ===========================================================================
# Dataset
# ===========================================================================

class ECGBeatDataset(Dataset):
    """
    One item = one recording's full beat sequence.

    Returns dict with:
      beats      : FloatTensor (n_beats, window_len, num_leads)
      label      : FloatTensor (num_classes,)   multi-hot
      lead_valid : BoolTensor  (num_leads,)
      record_id  : str

    Note the beats axis order: samples BEFORE leads, which is how
    extract_beats() slices them and therefore how they sit in the .npz.
    model_stgnn.STGNN takes this order and transposes internally.
    """

    def __init__(self, index_path: str):
        with open(index_path) as fh:
            self.records: List[Dict] = json.load(fh)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        npz = np.load(rec["npz_path"])
        beats = torch.from_numpy(npz["beats"]).float()           # (n_beats, window_len, 12)
        label = torch.from_numpy(npz["label"]).float()           # (5,)
        lead_valid = torch.from_numpy(npz["lead_valid"]).bool()  # (12,)
        return {
            "beats": beats,
            "label": label,
            "lead_valid": lead_valid,
            "record_id": rec["record_id"],
        }


def identity_collate(batch: List[Dict]) -> Dict:
    """
    batch_size=1 collate: unwrap the single-element list instead of letting
    default_collate stack across a batch dim (which would add a redundant
    leading dim of 1). Keeps tensor shapes exactly (n_beats, window_len, 12)
    as the model expects, with nothing to squeeze downstream.
    """
    assert len(batch) == 1, "identity_collate only supports batch_size=1"
    return batch[0]


def make_dataloader(split: str, processed_path: str = PNConfig.PROCESSED_PATH,
                     shuffle: bool = None, num_workers: int = 0) -> DataLoader:
    """split : "train" | "val" | "test" """
    index_path = os.path.join(processed_path, f"index_{split}.json")
    dataset = ECGBeatDataset(index_path)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=identity_collate,
    )


if __name__ == "__main__":
    adj = build_anatomical_adjacency()
    print("Adjacency matrix:\n", adj)

    loader = make_dataloader("train")
    batch = next(iter(loader))
    print("beats:", batch["beats"].shape)
    print("label:", batch["label"], batch["label"].shape)
    print("lead_valid:", batch["lead_valid"])
    print("record_id:", batch["record_id"])