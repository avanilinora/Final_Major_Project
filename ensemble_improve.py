"""
ensemble_improve.py  --  try better ways to combine the 3 models, WITHOUT peeking at test.

How it stays honest
-------------------
Every method is scored by 5-fold cross-validation INSIDE the validation set
(fit on 4/5 of val, score on the other 1/5). The method with the best val score
is chosen BEFORE test is looked at. Only then is it refit on all of val and
evaluated on test once, with the same significance tests as ensemble_eval.py.
The test score of every other method is printed too, for transparency 
Methods tried
-------------
  equal          plain average of the three models' probabilities (your current baseline)
  temp           first "temperature-scale" each model on val (fixes over-confidence), then average
  temp_geo       temperature-scale, then average log-probabilities instead of probabilities
  stack          a small logistic-regression that learns how to combine the 15 probabilities
  *+bias         any of the above plus one learned offset per class (helps rare classes like SVT)
  temp_gate      (only with --stgnn_status) like temp, but STGNN is ignored on rows where its own
                 preprocessing reported a problem (status != ok)

Usage (from the project root)
-----------------------------
  python ensemble_improve.py
  python ensemble_improve.py --stgnn_status teammates/stgnn/status_val.csv,teammates/stgnn/status_test.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

EPS = 1e-7


# ---------------------------------------------------------------- metrics
def macro_f1(y, p, C):
    cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(f1.mean()), f1


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def boot_ci(y, pa, pb, C, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    fa, fb = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        fa.append(macro_f1(y[i], pa[i], C)[0])
        fb.append(macro_f1(y[i], pb[i], C)[0])
    fa, fb = np.array(fa), np.array(fb)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return q(fa), q(fb), q(fa - fb)


# ---------------------------------------------------------------- building blocks
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(p, y):
    """One number T per model. T > 1 softens over-confident probabilities."""
    logp = np.log(np.clip(p, EPS, 1))
    nll = lambda T: -np.mean(np.log(np.clip(softmax(logp / T)[np.arange(len(y)), y], EPS, 1)))
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def apply_temperature(p, T):
    return softmax(np.log(np.clip(p, EPS, 1)) / T)


def fit_bias(p, y, C, grid=np.arange(-2.0, 2.01, 0.1), passes=2):
    """One additive offset per class on log-probabilities, chosen to maximise macro-F1."""
    logp = np.log(np.clip(p, EPS, 1))
    b = np.zeros(C)
    for _ in range(passes):
        for c in range(C):
            best_v, best_f = b[c], -1
            for v in grid:
                b[c] = v
                f = macro_f1(y, (logp + b).argmax(1), C)[0]
                if f > best_f + 1e-9:
                    best_v, best_f = v, f
            b[c] = best_v
    return b


def apply_bias(p, b):
    return softmax(np.log(np.clip(p, EPS, 1)) + b)


# ---------------------------------------------------------------- methods
# Each method: fit(Ps, y, S) -> predict(Ps, S) -> probs.   Ps = list of (N,C) arrays,
# S = per-row "STGNN ok?" boolean array or None.

def m_equal(Ps, y, S, C):
    return lambda Q, S2: sum(Q) / len(Q)


def m_temp(Ps, y, S, C):
    Ts = [fit_temperature(p, y) for p in Ps]
    return lambda Q, S2: sum(apply_temperature(q, t) for q, t in zip(Q, Ts)) / len(Q)


def m_temp_geo(Ps, y, S, C):
    Ts = [fit_temperature(p, y) for p in Ps]
    return lambda Q, S2: softmax(sum(np.log(np.clip(apply_temperature(q, t), EPS, 1))
                                     for q, t in zip(Q, Ts)) / len(Q))


def m_stack(Ps, y, S, C):
    feats = lambda Q: np.hstack([np.log(np.clip(q, EPS, 1)) for q in Q])
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(feats(Ps), y)
    return lambda Q, S2: clf.predict_proba(feats(Q))


def m_temp_gate(Ps, y, S, C, stgnn_idx=2):
    Ts = [fit_temperature(p, y) for p in Ps]

    def pred(Q, S2):
        cal = [apply_temperature(q, t) for q, t in zip(Q, Ts)]
        w = np.ones((len(Q[0]), len(Q)))
        w[~S2, stgnn_idx] = 0.0                     # ignore STGNN where its preprocessing struggled
        w = w / w.sum(1, keepdims=True)
        return sum(w[:, [i]] * c for i, c in enumerate(cal))
    return pred


def with_bias(base):
    def fit(Ps, y, S, C):
        f = base(Ps, y, S, C)
        b = fit_bias(f(Ps, S), y, C)
        return lambda Q, S2: apply_bias(f(Q, S2), b)
    return fit


# ---------------------------------------------------------------- main
def cv_score(method, V, yv, Sv, C, k=5, seed=0):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(np.zeros(len(yv)), yv):
        pred = method([v[tr] for v in V], yv[tr], None if Sv is None else Sv[tr], C)
        p = pred([v[te] for v in V], None if Sv is None else Sv[te])
        scores.append(macro_f1(yv[te], p.argmax(1), C)[0])
    return float(np.mean(scores)), float(np.std(scores))


def load_status(path):
    import csv
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return np.array([r["status"] == "ok" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_names", default="cnn_bilstm,inception,stgnn")
    ap.add_argument("--class_names", default="SB,AFIB,SR,ST,SVT")
    ap.add_argument("--val_labels", default="y_val.npy")
    ap.add_argument("--test_labels", default="y_test.npy")
    ap.add_argument("--stgnn_status", default="",
                    help="status_val.csv,status_test.csv from the STGNN handoff (optional)")
    ap.add_argument("--output_json", default="ensemble_improve_results.json")
    a = ap.parse_args()

    names, cn = a.model_names.split(","), a.class_names.split(",")
    V = [np.load(f"val_probs_{n}.npy") for n in names]
    T = [np.load(f"test_probs_{n}.npy") for n in names]
    yv, yt = np.load(a.val_labels).astype(int), np.load(a.test_labels).astype(int)
    C = len(cn)

    Sv = St = None
    methods = {"equal": m_equal, "temp": m_temp, "temp_geo": m_temp_geo, "stack": m_stack}
    if a.stgnn_status:
        pv, pt = a.stgnn_status.split(",")
        Sv, St = load_status(pv), load_status(pt)
        assert len(Sv) == len(yv) and len(St) == len(yt), "status CSV row count does not match labels"
        print(f"STGNN status ok: val {Sv.mean():.1%}, test {St.mean():.1%}")
        methods["temp_gate"] = lambda Ps, y, S, C: m_temp_gate(Ps, y, S, C, names.index("stgnn"))
    for k in list(methods):
        methods[k + "+bias"] = with_bias(methods[k])

    # ---- 1) choose on validation only
    print("\n== STEP 1: 5-fold cross-validation inside VAL (test not touched) ==")
    cv = {}
    for k, m in methods.items():
        cv[k] = cv_score(m, V, yv, Sv, C)
        print(f"  {k:<16} val-CV macro-F1 = {cv[k][0]:.4f} ± {cv[k][1]:.4f}")
    chosen = max(cv, key=lambda k: cv[k][0])
    print(f"\n  -> chosen on validation: {chosen}")

    # ---- 2) refit every method on all of val, evaluate on test
    print("\n== STEP 2: refit on all of VAL, evaluate on TEST ==")
    best_single = max(range(len(names)), key=lambda i: macro_f1(yt, T[i].argmax(1), C)[0])
    pb = T[best_single].argmax(1)
    f_single = macro_f1(yt, pb, C)[0]
    print(f"  best single model: {names[best_single]}  test macro-F1 = {f_single:.4f}")

    out = {"chosen_on_val": chosen, "best_single": names[best_single],
           "best_single_test_f1": f_single, "methods": {}}
    test_probs = {}
    for k, m in methods.items():
        pred = m(V, yv, Sv, C)
        pt_ = pred(T, St)
        test_probs[k] = pt_
        f, pc = macro_f1(yt, pt_.argmax(1), C)
        acc = float((pt_.argmax(1) == yt).mean())
        out["methods"][k] = {"val_cv_f1": cv[k][0], "val_cv_std": cv[k][1], "test_f1": f,
                             "test_acc": acc, "test_ece": ece(pt_, yt),
                             "per_class_f1": dict(zip(cn, map(float, pc)))}
        tag = "  <== chosen" if k == chosen else ""
        print(f"  {k:<16} test F1={f:.4f} acc={acc:.4f} ECE={ece(pt_, yt):.4f} "
              f"SVT-F1={pc[cn.index('SVT')] if 'SVT' in cn else pc[-1]:.3f}{tag}")

    # ---- 3) significance for the chosen method only
    pe = test_probs[chosen].argmax(1)
    ci_e, ci_b, ci_d = boot_ci(yt, pe, pb, C)
    ce, cb = pe == yt, pb == yt
    b, c = int((ce & ~cb).sum()), int((~ce & cb).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    out["significance_chosen_vs_best_single"] = {
        "ens_f1_ci95": ci_e, "best_f1_ci95": ci_b, "diff_ci95": ci_d,
        "mcnemar_ens_only_correct": b, "mcnemar_single_only_correct": c, "mcnemar_p": float(p)}
    print(f"\n== STEP 3: chosen method '{chosen}' vs {names[best_single]} ==")
    print(f"  F1 difference 95% CI: [{ci_d[0]:+.4f}, {ci_d[1]:+.4f}]")
    print(f"  McNemar: ensemble-only-correct={b}, single-only-correct={c}, p={p:.4f} "
          f"-> {'significant' if p < 0.05 else 'NOT significant'} at 0.05")

    json.dump(out, open(a.output_json, "w"), indent=2)
    print(f"\nsaved -> {a.output_json}")
    print("Report the CHOSEN row. The other test rows are for transparency only.")


if __name__ == "__main__":
    main()
