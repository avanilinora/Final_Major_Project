"""
ensemble_combiner.py  --  the FINAL ensemble (stack+bias), saved to a file so every
later step (XAI, deletion check, demos) uses exactly the model you report.

Step 1 (once):  python ensemble_combiner.py
    - fits the stack+bias combiner on VALIDATION probs only (same recipe as
      ensemble_improve.py, which chose it by cross-validation on val)
    - saves it to ensemble_combiner.json
    - prints the test score, which must match ensemble_improve_results.json (0.9625)
    - saves test_probs_ensemble.npy (used to verify the XAI wrapper later)

Later, anywhere:
    from ensemble_combiner import StackedCombiner
    comb = StackedCombiner.load("ensemble_combiner.json")
    p_ens = comb.predict_proba([p_cnn_bilstm, p_inception, p_stgnn])   # each (N,5) -> (N,5)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from ensemble_improve import EPS, fit_bias, macro_f1, ece, softmax


class StackedCombiner:
    """Logistic regression on the members' log-probabilities, plus one offset per class."""

    def __init__(self, model_names, class_names, coef, intercept, bias):
        self.model_names = list(model_names)
        self.class_names = list(class_names)
        self.coef = np.asarray(coef, float)            # (C, n_models*C)
        self.intercept = np.asarray(intercept, float)  # (C,)
        self.bias = np.asarray(bias, float)            # (C,)

    # ---- use
    def predict_proba(self, probs_list):
        feats = np.hstack([np.log(np.clip(np.asarray(p, float), EPS, 1)) for p in probs_list])
        p_stack = softmax(feats @ self.coef.T + self.intercept)
        return softmax(np.log(np.clip(p_stack, EPS, 1)) + self.bias)

    def model_shares(self):
        """How much the stacker leans on each model (share of total |coefficient|).
        Shown as 'w=' in the model-agreement figure; sums to 1."""
        C = len(self.class_names)
        mass = np.array([np.abs(self.coef[:, i * C:(i + 1) * C]).sum() for i in range(len(self.model_names))])
        return dict(zip(self.model_names, (mass / mass.sum()).tolist()))

    # ---- save / load
    def save(self, path):
        json.dump({"method": "stack+bias", "fitted_on": "validation set only",
                   "model_names": self.model_names, "class_names": self.class_names,
                   "coef": self.coef.tolist(), "intercept": self.intercept.tolist(),
                   "bias": self.bias.tolist(), "model_shares": self.model_shares()},
                  open(path, "w"), indent=2)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["model_names"], d["class_names"], d["coef"], d["intercept"], d["bias"])

    # ---- fit (same recipe as ensemble_improve.py's stack+bias)
    @classmethod
    def fit(cls, val_probs, y_val, model_names, class_names):
        C = len(class_names)
        feats = np.hstack([np.log(np.clip(p, EPS, 1)) for p in val_probs])
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(feats, y_val)
        comb = cls(model_names, class_names, clf.coef_, clf.intercept_, np.zeros(C))
        # sanity: our hand-written formula must equal sklearn's own output
        assert np.allclose(softmax(feats @ comb.coef.T + comb.intercept), clf.predict_proba(feats), atol=1e-6)
        comb.bias = fit_bias(clf.predict_proba(feats), y_val, C)
        return comb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_names", default="cnn_bilstm,inception,stgnn")
    ap.add_argument("--class_names", default="SB,AFIB,SR,ST,SVT")
    ap.add_argument("--out", default="ensemble_combiner.json")
    a = ap.parse_args()
    names, cn = a.model_names.split(","), a.class_names.split(",")

    V = [np.load(f"val_probs_{n}.npy") for n in names]
    T = [np.load(f"test_probs_{n}.npy") for n in names]
    yv, yt = np.load("y_val.npy").astype(int), np.load("y_test.npy").astype(int)

    comb = StackedCombiner.fit(V, yv, names, cn)
    comb.save(a.out)

    pt = comb.predict_proba(T)
    np.save("test_probs_ensemble.npy", pt)
    f, pc = macro_f1(yt, pt.argmax(1), len(cn))
    print(f"saved -> {a.out}  and  test_probs_ensemble.npy")
    print(f"test macro-F1 = {f:.4f}   acc = {(pt.argmax(1) == yt).mean():.4f}   ECE = {ece(pt, yt):.4f}")
    print("per-class F1:", {c: round(float(v), 4) for c, v in zip(cn, pc)})
    print("model shares (how much the stacker leans on each model):",
          {k: round(v, 3) for k, v in comb.model_shares().items()})
    print("\nThe test macro-F1 above should be 0.9625 (same as ensemble_improve_results.json).")


if __name__ == "__main__":
    main()