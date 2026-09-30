"""
utils/calibrate_stgnn_rr.py
============================
Post-hoc calibration for the R-R model. Same method as calibrate_stgnn.py —
per-class vector scaling fitted on validation, applied to test — but pointed at
STGNN_RR and build_rr_loader.

All the fitting and scoring logic is imported from calibrate_stgnn, not copied:
only the model class and the loader differ, so only those are re-stated here.

Why this is needed at all: the R-R model reaches macro AUROC 0.9931 on test but
only 0.8989 argmax accuracy, with tuned thresholds spanning 0.066 (AFIB) to
0.940 (SR). Near-perfect ranking, badly mismatched scales — exactly the profile
vector scaling repaired on the baseline (+2.34 points from a lower AUROC).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from calibrate_stgnn import (
    fit_temperature,
    fit_vector_scaling,
    print_confusion,
    score,
)
from config_physionet import PhysioNetConfig as PNConfig
from dataset_rr import build_rr_loader
from model_stgnn import count_parameters
from model_stgnn_rr import STGNN_RR

CKPT_DIR = os.path.join(os.path.dirname(PNConfig.CHECKPOINT_PATH), "checkpoints_stgnn_rr")


@torch.no_grad()
def collect_logits_rr(model, loader, device):
    """Raw logits — the only thing that differs from the non-RR version is the
    extra `rr` argument to forward()."""
    model.eval()
    logits, labels = [], []
    for batch in tqdm(loader, desc="  forward", ncols=88, leave=False):
        z = model(batch["beats"].to(device, non_blocking=True),
                  batch["rr"].to(device, non_blocking=True),
                  batch["lead_valid"].to(device, non_blocking=True))
        logits.append(z.float().cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.vstack(logits), np.vstack(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(CKPT_DIR, "stgnn_rr_best.pt"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    PNConfig.load_label_vocab()
    device = PNConfig.DEVICE

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"No checkpoint at {args.ckpt}")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    class_names = ck.get("class_names", PNConfig.CLASS_NAMES)
    tc = ck.get("train_config", {})

    model = STGNN_RR(
        num_classes=len(class_names),
        node_dim=tc.get("node_dim", 128),
        gat_heads=tc.get("gat_heads", 4),
        gru_hidden=tc.get("gru_hidden", 128),
        dropout=tc.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ck["model_state"])

    print("=" * 84)
    print("  STGNN + R-R — post-hoc calibration (per-class vector scaling)")
    print("=" * 84)
    print(f"  checkpoint : {args.ckpt}  (epoch {ck.get('epoch', '?')})")
    print(f"  device     : {device}   params: {count_parameters(model):,}")

    print("\n  collecting validation logits (fitting set)...")
    z_val, y_val = collect_logits_rr(model, build_rr_loader("val", False, None, args.workers), device)
    print("  collecting test logits (held out)...")
    z_test, y_test = collect_logits_rr(model, build_rr_loader("test", False, None, args.workers), device)
    print(f"  val {z_val.shape[0]:,} | test {z_test.shape[0]:,}")
    print("=" * 84)

    base = score(z_test, y_test, class_names)
    T = fit_temperature(z_val, y_val, device)
    temp = score(z_test / T, y_test, class_names)
    a, b = fit_vector_scaling(z_val, y_val, device)
    vec = score(z_test * a + b, y_test, class_names)

    print(f"\n  Fitted on validation, applied to test")
    print(f"    temperature T = {T:.4f}")
    print(f"    {'class':<6}{'scale a':>10}{'bias b':>10}")
    for i, n in enumerate(class_names):
        print(f"    {n:<6}{a[i]:>10.4f}{b[i]:>10.4f}")

    print(f"\n  TEST RESULTS (n={base['n']:,} single-label records)")
    print(f"  {'':<22}{'argmax acc':>12}{'errors':>9}{'NLL':>9}{'Brier':>9}{'ECE':>9}")
    print(f"  {'-'*70}")
    for tag, m in [("baseline", base), ("temperature (control)", temp),
                   ("vector scaling", vec)]:
        print(f"  {tag:<22}{m['argmax_accuracy']:>12.4f}{m['n_errors']:>9d}"
              f"{m['nll']:>9.4f}{m['brier']:>9.4f}{m['ece']:>9.4f}")

    print(f"\n  Per-class argmax recall")
    print(f"  {'class':<6}{'baseline':>11}{'vector':>11}{'change':>11}")
    print(f"  {'-'*39}")
    for n in class_names:
        r0, r1 = base["per_class_recall"][n], vec["per_class_recall"][n]
        print(f"  {n:<6}{r0:>11.4f}{r1:>11.4f}{r1 - r0:>+11.4f}")

    print_confusion(base["confusion"], class_names, "BEFORE (rows = true, cols = predicted)")
    print_confusion(vec["confusion"], class_names, "AFTER vector scaling")

    delta = vec["argmax_accuracy"] - base["argmax_accuracy"]
    print(f"\n  Accuracy {base['argmax_accuracy']:.4f} -> {vec['argmax_accuracy']:.4f} "
          f"({delta:+.4f}, {base['n_errors'] - vec['n_errors']:+d} errors)")
    print(f"  Reference: plain STGNN was 0.9148 raw / 0.9382 calibrated.")
    need = int(np.ceil(0.95 * base["n"]))
    print(f"  For 95%: need <= {base['n'] - need} errors; currently {vec['n_errors']}.")

    if args.save:
        out = os.path.join(CKPT_DIR, f"calibration_rr_epoch{ck.get('epoch', 0)}.json")
        with open(out, "w") as fh:
            json.dump({"checkpoint": args.ckpt, "epoch": ck.get("epoch"),
                       "class_names": class_names, "fitted_on": "val",
                       "temperature": T, "vector_scale_a": a.tolist(),
                       "vector_bias_b": b.tolist(), "test_baseline": base,
                       "test_temperature": temp, "test_vector_scaling": vec}, fh, indent=2)
        print(f"\n  saved : {out}")

    print("\nOK")


if __name__ == "__main__":
    main()
