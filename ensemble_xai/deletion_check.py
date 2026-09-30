"""
ensemble_xai/deletion_check.py  --  quantitative test of the ensemble occlusion maps.

For each recording: rank every (lead, 0.5 s window) cell by ensemble saliency, then zero cells
and track the ENSEMBLE probability of its original predicted class.
  MoRF   = most-relevant first        (should fall fastest if the map is faithful)
  random = random cells, mean of R    (null baseline)
  LeRF   = least-relevant first       (should barely fall)
Faithful map:  drop(MoRF) > drop(random) > drop(LeRF).   AOPC = mean drop over the fractions.

Usage:
  python -m ensemble_xai.deletion_check --n_per_class 8
      (default: tests the FINAL stack+bias ensemble saved in ensemble_combiner.json)
  python -m ensemble_xai.deletion_check --combiner none --weights 0.45,0.39,0.16
      (old weighted-average ensemble, for comparison only)
"""
from __future__ import annotations
import argparse, json
import numpy as np
from ensemble_xai.occlusion import compute_occlusion_explanation

FRACS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


def _ens(adapters, w, x, combiner=None):
    probs = [a.predict_proba(x) for a in adapters]
    if combiner is not None:
        return combiner.predict_proba(probs)
    w = np.asarray(w, float) / np.sum(w)
    return sum(wi * p for wi, p in zip(w, probs))


def _zero_cells(x, cells, ws):
    y = x.copy()
    for l, k in cells:
        y[k * ws:(k + 1) * ws, l] = 0.0
    return y


def deletion_curves(x, adapters, weights, class_names, ws=250, R=5, rng=None, combiner=None):
    rng = rng or np.random.default_rng(0)
    res = compute_occlusion_explanation(x, adapters, np.asarray(weights, float), class_names,
                                        window_size=ws, stride=ws, combiner=combiner)
    L, T = res.ensemble_lead_time_map.shape
    nw = int(np.ceil(T / ws)); c = res.predicted_class
    score = np.zeros((L, nw))
    for k in range(nw):
        score[:, k] = res.ensemble_lead_time_map[:, k * ws:(k + 1) * ws].mean(1) * res.ensemble_per_lead_importance
    flat = [(l, k) for l in range(L) for k in range(nw)]
    order = np.argsort(-score.ravel(), kind="stable")           # most relevant first
    p0 = res.confidence

    def curve(sel_fn):
        out = []
        for f in FRACS:
            n = int(round(f * len(flat)))
            out.append(float(_ens(adapters, weights, _zero_cells(x, sel_fn(n), ws)[None], combiner)[0, c]))
        return out

    morf = curve(lambda n: [flat[i] for i in order[:n]])
    lerf = curve(lambda n: [flat[i] for i in order[::-1][:n]])
    rnd = np.mean([curve(lambda n: [flat[i] for i in rng.permutation(len(flat))[:n]]) for _ in range(R)], axis=0)
    return p0, morf, rnd.tolist(), lerf


def run(X, y, adapters, weights, class_names, n_per_class=8, seed=0, combiner=None):
    rng = np.random.default_rng(seed); M, Rn, Lr, P0 = [], [], [], []
    for cls in range(len(class_names)):
        idx = np.where(y == cls)[0]
        for i in rng.choice(idx, size=min(n_per_class, len(idx)), replace=False):
            p0, m, r, l = deletion_curves(X[i], adapters, weights, class_names, rng=rng, combiner=combiner)
            P0.append(p0); M.append(m); Rn.append(r); Lr.append(l)
    M, Rn, Lr = map(np.array, (M, Rn, Lr))
    aopc = lambda A: float(np.mean(A[:, [0]] - A[:, 1:]))
    frac_ok = float(np.mean((M[:, 1:] <= Rn[:, 1:]).mean(1) >= 0.8))
    out = {"n_recordings": len(M), "fractions": FRACS,
           "mean_prob_MoRF": M.mean(0).tolist(), "mean_prob_random": Rn.mean(0).tolist(), "mean_prob_LeRF": Lr.mean(0).tolist(),
           "AOPC_MoRF": aopc(M), "AOPC_random": aopc(Rn), "AOPC_LeRF": aopc(Lr),
           "share_recordings_MoRF_below_random": frac_ok}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from config_physionet import PhysioNetConfig as PNConfig
    from ensemble_xai.adapters import build_adapters, verify_adapter
    ap = argparse.ArgumentParser()
    ap.add_argument("--combiner", default="ensemble_combiner.json",
                    help="saved stack+bias ensemble; 'none' to use --weights instead")
    ap.add_argument("--weights", default="1,1,1", help="only used with --combiner none; same order as build_adapters()")
    ap.add_argument("--n_per_class", type=int, default=8)
    ap.add_argument("--test_probs", default="test_probs_cnn_bilstm.npy,test_probs_inception.npy,test_probs_stgnn.npy")
    a = ap.parse_args()
    PNConfig.load_label_vocab()
    import os
    X = np.load(os.path.join(PNConfig.PROCESSED_PATH, "X_test.npy")); y = np.load(os.path.join(PNConfig.PROCESSED_PATH, "y_test.npy"))
    ads = build_adapters()
    for ad, p in zip(ads, a.test_probs.split(",")):
        verify_adapter(ad, X, np.load(p))
    w = np.array([float(v) for v in a.weights.split(",")])
    comb = None
    if a.combiner.lower() != "none":
        from ensemble_combiner import StackedCombiner
        comb = StackedCombiner.load(a.combiner)
        assert comb.model_names == [ad.name for ad in ads], "combiner model order != adapter order"
        got = _ens(ads, w, np.asarray(X[:64]), comb)
        diff = np.abs(got - np.load("test_probs_ensemble.npy")[:64]).max()
        print(f"[verify] ensemble     max|diff|={diff:.5f}")
        if diff > 1e-3:
            raise RuntimeError("stacked ensemble does not reproduce test_probs_ensemble.npy")
    out = run(X, y, ads, w, PNConfig.CLASS_NAMES, a.n_per_class, combiner=comb)
    out["ensemble"] = "stack+bias (ensemble_combiner.json)" if comb else f"weighted average {w.tolist()}"
    json.dump(out, open("ensemble_deletion_results.json", "w"), indent=2)
    fr = np.array(FRACS) * 100
    plt.figure(figsize=(6, 4))
    for k, lab in [("mean_prob_MoRF", "Most-relevant first"), ("mean_prob_random", "Random"), ("mean_prob_LeRF", "Least-relevant first")]:
        plt.plot(fr, out[k], marker="o", label=lab)
    plt.xlabel("% of (lead, window) cells zeroed"); plt.ylabel("Ensemble P(predicted class)")
    plt.title("Deletion test of ensemble occlusion maps"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("ensemble_deletion_curve.png", dpi=200); print("saved ensemble_deletion_curve.png")