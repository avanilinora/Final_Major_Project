"""
ensemble_eval.py  --  3-model ensemble (XGBoost dropped): honest evaluation in one run.

Produces:
  * individual test metrics + per-class F1 (incl. SVT) + ECE
  * DE-optimised weights (fit on VAL only)  vs  equal-weight average
  * every subset of >=2 models (drop-one ablation); gain is measured against
    each row's OWN best member
  * bootstrap 95% CI on test macro-F1 (ensemble, best single, paired difference)
  * exact McNemar test: ensemble vs the best single model
  * sanity check that every model's saved probs really line up with y_test

Usage:
  python ensemble_eval.py \
    --val_probs val_probs_cnn_bilstm.npy,val_probs_inception.npy,val_probs_stgnn.npy \
    --test_probs test_probs_cnn_bilstm.npy,test_probs_inception.npy,test_probs_stgnn.npy \
    --val_labels y_val.npy --test_labels y_test.npy \
    --model_names cnn_bilstm,inception,stgnn \
    --class_names SB,AFIB,SR,ST,SVT
"""
from __future__ import annotations
import argparse, itertools, json
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import binomtest


def macro_f1(y, p, C):
    cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(f1.mean()), f1


def ens(probs, w):
    w = np.asarray(w, float); w = w / w.sum()
    return sum(wi * p for wi, p in zip(w, probs))


def fit_weights(val_probs, y_val, C, seed=42):
    f = lambda w: -macro_f1(y_val, ens(val_probs, np.abs(w) + 1e-12).argmax(1), C)[0]
    r = differential_evolution(f, [(0, 1)] * len(val_probs), seed=seed, maxiter=100, popsize=20, tol=1e-6)
    w = np.abs(r.x) + 1e-12
    return w / w.sum()


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def boot_ci(y, pa, pb, C, n=2000, seed=0):
    rng = np.random.default_rng(seed); N = len(y)
    fa, fb = [], []
    for _ in range(n):
        i = rng.integers(0, N, N)
        fa.append(macro_f1(y[i], pa[i], C)[0]); fb.append(macro_f1(y[i], pb[i], C)[0])
    fa, fb = np.array(fa), np.array(fb)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return q(fa), q(fb), q(fa - fb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_probs", required=True); ap.add_argument("--test_probs", required=True)
    ap.add_argument("--val_labels", required=True); ap.add_argument("--test_labels", required=True)
    ap.add_argument("--model_names", required=True); ap.add_argument("--class_names", default="")
    ap.add_argument("--output_json", default="ensemble_eval_results.json")
    a = ap.parse_args()

    names = a.model_names.split(",")
    V = [np.load(p) for p in a.val_probs.split(",")]
    T = [np.load(p) for p in a.test_probs.split(",")]
    yv, yt = np.load(a.val_labels).astype(int), np.load(a.test_labels).astype(int)
    C = V[0].shape[1]
    cn = a.class_names.split(",") if a.class_names else [str(i) for i in range(C)]
    assert all(t.shape == (len(yt), C) for t in T) and all(v.shape == (len(yv), C) for v in V), "shape mismatch"

    out = {"individual": {}, "subsets": {}}
    print("\n== SANITY: each model on YOUR y_test.npy (must equal what its owner reports) ==")
    for n, t, v in zip(names, T, V):
        f, pc = macro_f1(yt, t.argmax(1), C)
        acc = float((t.argmax(1) == yt).mean())
        out["individual"][n] = {"acc": acc, "macro_f1": f, "per_class_f1": dict(zip(cn, map(float, pc))),
                                "ece": ece(t, yt), "val_acc": float((v.argmax(1) == yv).mean())}
        print(f"  {n:<12} acc={acc:.4f} F1={f:.4f} ECE={out['individual'][n]['ece']:.4f}  SVT-F1={pc[-1]:.3f}")
        if acc < 0.5:
            print(f"  !! {n}: accuracy this low means its rows are NOT in y_test order")

    def run(idx, label):
        vs, ts = [V[i] for i in idx], [T[i] for i in idx]
        w = fit_weights(vs, yv, C)
        pe = ens(ts, w).argmax(1)
        pe_eq = ens(ts, np.ones(len(idx))).argmax(1)
        f, pc = macro_f1(yt, pe, C); f_eq, _ = macro_f1(yt, pe_eq, C)
        best_i = max(idx, key=lambda i: out["individual"][names[i]]["macro_f1"])
        best_f = out["individual"][names[best_i]]["macro_f1"]
        r = {"members": [names[i] for i in idx], "weights": dict(zip([names[i] for i in idx], map(float, w))),
             "opt_f1": f, "equal_f1": f_eq, "best_member": names[best_i], "best_member_f1": best_f,
             "gain_opt_vs_own_best": f - best_f, "gain_equal_vs_own_best": f_eq - best_f}
        out["subsets"][label] = r
        print(f"  {label:<28} opt={f:.4f} equal={f_eq:.4f} | own best={names[best_i]} {best_f:.4f} "
              f"| gain(opt)={f-best_f:+.4f} gain(eq)={f_eq-best_f:+.4f} | w={ {k: round(v,3) for k,v in r['weights'].items()} }")
        return w, pe

    print("\n== ENSEMBLE ABLATION (weights fit on VAL only; DE seed 42) ==")
    w_full, pe_full = run(list(range(len(names))), "FULL (" + "+".join(names) + ")")
    for k in range(len(names) - 1, 1, -1):
        for idx in itertools.combinations(range(len(names)), k):
            run(list(idx), "+".join(names[i] for i in idx))

    best = max(names, key=lambda n: out["individual"][n]["macro_f1"]); bi = names.index(best)
    pb = T[bi].argmax(1)
    ci_e, ci_b, ci_d = boot_ci(yt, pe_full, pb, C)
    ce, cb = pe_full == yt, pb == yt
    b, c = int((ce & ~cb).sum()), int((~ce & cb).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    out["significance"] = {"best_single": best, "ens_f1_ci95": ci_e, "best_f1_ci95": ci_b,
                           "diff_ci95": ci_d, "mcnemar_b_ens_only_correct": b, "mcnemar_c_single_only_correct": c,
                           "mcnemar_p": float(p)}
    out["ensemble_ece"] = ece(ens(T, w_full), yt)
    out["ensemble_per_class_f1"] = dict(zip(cn, map(float, macro_f1(yt, pe_full, C)[1])))
    print(f"\n== SIGNIFICANCE: ensemble vs {best} ==")
    print(f"  ens F1 95% CI {np.round(ci_e,4)} | {best} {np.round(ci_b,4)} | diff {np.round(ci_d,4)}")
    print(f"  McNemar exact: ens-only-correct={b}, single-only-correct={c}, p={p:.4f}"
          f"  -> {'significant' if p < 0.05 else 'NOT significant'} at 0.05")
    print(f"  Ensemble ECE={out['ensemble_ece']:.4f}  per-class F1={out['ensemble_per_class_f1']}")
    print("\nNOTE: DE-seed variation only measures weight-search stability, not model stability.")
    json.dump(out, open(a.output_json, "w"), indent=2); print("saved ->", a.output_json)


if __name__ == "__main__":
    main()