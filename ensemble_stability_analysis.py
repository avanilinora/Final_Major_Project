"""
ensemble_stability_analysis.py
================================
Runs the ensemble weight-optimization step N times with different random
seeds (default N=4, matching your guide's "3-4 runs" instruction), and
reports mean ± std for accuracy and macro-F1 — the same reporting
convention used in your reference papers (e.g. "0.9980 ± 0.0014").

*** Scope, stated honestly ***
This captures variance from the differential-evolution weight optimizer
only. It does NOT capture variance from retraining the underlying models
themselves (different weight initialization / data shuffling), which
would be the more complete version of what your guide is asking for.
That would require each teammate to retrain their model 3-4 times — not
realistic on a tight deadline. This script is the honest, achievable
subset: it tells you whether your CHOICE OF WEIGHTS is stable, even if
it can't tell you whether the underlying models themselves are stable.
State this scope explicitly in your report — don't let the ± numbers
imply more than they actually cover.

Usage
-----
    # Run stability analysis on one model-set configuration
    python ensemble_stability_analysis.py \
        --val_probs val_probs_cnn_bilstm.npy,val_probs_inception.npy,val_probs_stgnn.npy,val_probs_xgboost.npy \
        --val_labels y_val.npy \
        --test_probs test_probs_cnn_bilstm.npy,test_probs_inception.npy,test_probs_stgnn.npy,test_probs_xgboost.npy \
        --test_labels y_test.npy \
        --model_names cnn_bilstm,inception,stgnn,xgboost \
        --n_runs 4

    # Then run it again with xgboost removed, and compare the two printed
    # summaries directly — that comparison IS your "select 3 or 4 models"
    # decision, backed by variance, not a single lucky/unlucky run.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def weighted_ensemble_probs(probs_list, weights):
    weights = weights / weights.sum()
    out = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        out += w * p
    return out


def optimize_weights(val_probs_list, y_val, num_classes, seed):
    def neg_macro_f1(weights):
        ensembled = weighted_ensemble_probs(val_probs_list, np.abs(weights))
        y_pred = ensembled.argmax(axis=1)
        return -f1_score(y_val, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0)

    bounds = [(0.0, 1.0)] * len(val_probs_list)
    result = differential_evolution(neg_macro_f1, bounds, seed=seed, maxiter=100, popsize=20, tol=1e-6)
    weights = np.abs(result.x)
    return weights / weights.sum()


def run_stability_analysis(val_probs_list, y_val, test_probs_list, y_test, model_names, n_runs, seeds=None):
    num_classes = val_probs_list[0].shape[1]
    seeds = seeds or list(range(n_runs))

    per_run_acc, per_run_f1, per_run_weights = [], [], []

    for run_idx, seed in enumerate(seeds, 1):
        weights = optimize_weights(val_probs_list, y_val, num_classes, seed)
        test_ensembled = weighted_ensemble_probs(test_probs_list, weights)
        y_pred = test_ensembled.argmax(axis=1)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0)

        per_run_acc.append(acc)
        per_run_f1.append(f1)
        per_run_weights.append(weights)

        logger.info("Run %d/%d (seed=%d): acc=%.4f  macro-F1=%.4f  weights=%s",
                    run_idx, len(seeds), seed, acc, f1, dict(zip(model_names, weights.round(4))))

    per_run_weights = np.array(per_run_weights)  # (n_runs, n_models)

    summary = {
        "model_names": model_names,
        "n_runs": len(seeds),
        "seeds": seeds,
        "accuracy_mean": float(np.mean(per_run_acc)),
        "accuracy_std": float(np.std(per_run_acc)),
        "macro_f1_mean": float(np.mean(per_run_f1)),
        "macro_f1_std": float(np.std(per_run_f1)),
        "per_model_weight_mean": {name: float(w) for name, w in zip(model_names, per_run_weights.mean(axis=0))},
        "per_model_weight_std": {name: float(w) for name, w in zip(model_names, per_run_weights.std(axis=0))},
        "per_run_accuracy": per_run_acc,
        "per_run_macro_f1": per_run_f1,
    }
    return summary


def print_summary(summary: dict, label: str) -> None:
    print("\n" + "=" * 70)
    print(f"STABILITY SUMMARY — {label}  ({summary['n_runs']} runs)")
    print("=" * 70)
    print(f"Accuracy : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"Macro-F1 : {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    print("-" * 70)
    print("Per-model weight stability (mean ± std across runs):")
    for name in summary["model_names"]:
        mean_w = summary["per_model_weight_mean"][name]
        std_w = summary["per_model_weight_std"][name]
        flag = "  <- consistently near-zero, low real contribution" if mean_w < 0.05 and std_w < 0.05 else ""
        print(f"  {name:<14}: {mean_w:.4f} ± {std_w:.4f}{flag}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble weight-optimization stability analysis (mean ± std over N runs).")
    parser.add_argument("--val_probs", type=str, required=True)
    parser.add_argument("--val_labels", type=str, required=True)
    parser.add_argument("--test_probs", type=str, required=True)
    parser.add_argument("--test_labels", type=str, required=True)
    parser.add_argument("--model_names", type=str, required=True)
    parser.add_argument("--n_runs", type=int, default=4, help="Matches your guide's '3-4 runs' instruction.")
    parser.add_argument("--label", type=str, default="ensemble")
    parser.add_argument("--output_json", type=str, default="ensemble_stability_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    model_names = args.model_names.split(",")
    val_probs_list = [np.load(p.strip()) for p in args.val_probs.split(",")]
    test_probs_list = [np.load(p.strip()) for p in args.test_probs.split(",")]
    y_val = np.load(args.val_labels)
    y_test = np.load(args.test_labels)

    logger.info("Running %d ensemble-optimization runs for: %s", args.n_runs, model_names)
    summary = run_stability_analysis(val_probs_list, y_val, test_probs_list, y_test, model_names, args.n_runs)
    print_summary(summary, args.label)

    with open(args.output_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Full results saved -> %s", args.output_json)


if __name__ == "__main__":
    main()