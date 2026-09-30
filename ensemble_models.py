"""
ensemble_models.py
===================
Decision-level ensembling of N heterogeneous models (your CNN-BiLSTM +
teammates' Inception-CNN, STGNN, XGBoost), via weighted softmax
probability averaging with weights optimized on the validation set.

    p_ens = w1*p1 + w2*p2 + ... + wN*pN     (weights sum to 1)
    y_hat = argmax(p_ens)

Weights are found via differential evolution (scipy), maximizing
validation macro-F1 — the same approach named in your reference paper's
ensemble section, generalised from 3 models to N.

*** REQUIREMENT — check this BEFORE running anything ***
Every model must output an (N_samples, N_classes) probability array over
the EXACT SAME class order (e.g. [SB, AFIB, SR, ST, SVT]). If a
teammate's model was trained on a different label vocabulary, this
breaks silently (columns won't mean the same thing) — confirm this
first, not after.

Usage
-----
Each model needs a `.predict_proba(X_val) -> (N, C)` -style function, OR
just pre-compute each model's val/test probabilities once and save them
as .npy files (recommended — decouples this script from needing every
teammate's framework/dependencies installed at once):

    np.save("val_probs_cnn_bilstm.npy", probs)   # from your PyTorch model
    np.save("val_probs_inception.npy", probs)    # from teammate 1
    np.save("val_probs_stgnn.npy", probs)        # from teammate 2
    np.save("val_probs_xgboost.npy", probs)      # from teammate 3 (predict_proba)

Then:
    python ensemble_models.py \
        --val_probs val_probs_cnn_bilstm.npy,val_probs_inception.npy,val_probs_stgnn.npy,val_probs_xgboost.npy \
        --val_labels y_val.npy \
        --test_probs test_probs_cnn_bilstm.npy,test_probs_inception.npy,test_probs_stgnn.npy,test_probs_xgboost.npy \
        --test_labels y_test.npy \
        --model_names cnn_bilstm,inception,stgnn,xgboost
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def load_probs(paths_csv: str) -> list:
    arrays = [np.load(p.strip()) for p in paths_csv.split(",")]
    shapes = [a.shape for a in arrays]
    if len(set(s[1] for s in shapes)) != 1:
        raise ValueError(
            f"Model outputs have different numbers of classes: {shapes}. "
            f"Every model MUST be trained on the same class vocabulary, in "
            f"the same order, before ensembling. Fix this before continuing."
        )
    if len(set(s[0] for s in shapes)) != 1:
        raise ValueError(
            f"Model outputs have different numbers of samples: {shapes}. "
            f"Every model must have been evaluated on the SAME ordered set "
            f"of examples."
        )
    return arrays


def weighted_ensemble_probs(probs_list: list, weights: np.ndarray) -> np.ndarray:
    weights = weights / weights.sum()  # normalise to sum to 1
    ensembled = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        ensembled += w * p
    return ensembled


def metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
    }


def optimize_weights(val_probs_list: list, y_val: np.ndarray, num_classes: int) -> np.ndarray:
    n_models = len(val_probs_list)

    def neg_macro_f1(weights):
        ensembled = weighted_ensemble_probs(val_probs_list, np.abs(weights))
        y_pred = ensembled.argmax(axis=1)
        return -f1_score(y_val, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0)

    bounds = [(0.0, 1.0)] * n_models
    result = differential_evolution(
        neg_macro_f1, bounds, seed=42, maxiter=100, popsize=20, tol=1e-6,
    )
    weights = np.abs(result.x)
    weights = weights / weights.sum()
    logger.info("Optimized weights (validation macro-F1 = %.4f): %s", -result.fun, weights)
    return weights


def parse_args():
    parser = argparse.ArgumentParser(description="Weighted-probability ensemble of N heterogeneous models.")
    parser.add_argument("--val_probs", type=str, required=True, help="Comma-separated .npy paths, val set.")
    parser.add_argument("--val_labels", type=str, required=True)
    parser.add_argument("--test_probs", type=str, required=True, help="Comma-separated .npy paths, test set.")
    parser.add_argument("--test_labels", type=str, required=True)
    parser.add_argument("--model_names", type=str, required=True, help="Comma-separated, same order as --*_probs.")
    parser.add_argument("--output_json", type=str, default="ensemble_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    model_names = args.model_names.split(",")

    val_probs_list = load_probs(args.val_probs)
    test_probs_list = load_probs(args.test_probs)
    y_val = np.load(args.val_labels)
    y_test = np.load(args.test_labels)
    num_classes = val_probs_list[0].shape[1]

    if len(val_probs_list) != len(model_names) or len(test_probs_list) != len(model_names):
        raise ValueError("Number of probability files must match number of model names.")

    logger.info("=" * 60)
    logger.info("Individual model performance (test set)")
    logger.info("=" * 60)
    individual_results = {}
    for name, probs in zip(model_names, test_probs_list):
        y_pred = probs.argmax(axis=1)
        m = metrics(y_test, y_pred, num_classes)
        individual_results[name] = m
        logger.info("  %-14s  acc=%.4f  macro-F1=%.4f", name, m["accuracy"], m["f1"])

    logger.info("=" * 60)
    logger.info("Optimizing ensemble weights on VALIDATION set...")
    weights = optimize_weights(val_probs_list, y_val, num_classes)
    for name, w in zip(model_names, weights):
        logger.info("  %-14s  weight=%.4f", name, w)

    logger.info("=" * 60)
    logger.info("Ensemble performance (test set, using validation-optimized weights)")
    ensembled_test_probs = weighted_ensemble_probs(test_probs_list, weights)
    y_pred_ensemble = ensembled_test_probs.argmax(axis=1)
    ensemble_metrics = metrics(y_test, y_pred_ensemble, num_classes)
    logger.info("  %-14s  acc=%.4f  macro-F1=%.4f", "ENSEMBLE", ensemble_metrics["accuracy"], ensemble_metrics["f1"])

    best_individual = max(individual_results.values(), key=lambda m: m["f1"])
    improvement = ensemble_metrics["f1"] - best_individual["f1"]
    logger.info("=" * 60)
    logger.info("Ensemble vs. best individual model: %+.4f macro-F1", improvement)
    if improvement <= 0:
        logger.warning(
            "Ensemble did NOT beat the best individual model. This can genuinely "
            "happen (e.g. one model dominates and others add noise) — report it "
            "honestly rather than only showing configurations where it helps."
        )

    results = {
        "individual_models": individual_results,
        "ensemble_weights": {name: float(w) for name, w in zip(model_names, weights)},
        "ensemble_metrics": ensemble_metrics,
        "improvement_over_best_individual_macro_f1": float(improvement),
    }
    with open(args.output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved -> %s", args.output_json)


if __name__ == "__main__":
    main()