"""
generate_ensemble_xai.py  --  the ensemble XAI figures, for the recordings you'll put in the report.

Picks, for each class, test recordings the FINAL (stack+bias) ensemble gets RIGHT, and runs the
occlusion pipeline on them. Optionally also picks one recording where the three models disagree,
which is the most interesting case for the model-agreement figure.

Everything is explained through ensemble_combiner.json, so the figures describe the same model
whose 96.25% you report.

Usage (from the project root):
    python generate_ensemble_xai.py                     # 1 correct recording per class
    python generate_ensemble_xai.py --per_class 2       # 2 each
    python generate_ensemble_xai.py --no_disagreement   # skip the disagreement case

Output:
    outputs/ensemble_xai/<CLASS>_<row>/...           figures per recording
    outputs/ensemble_xai/summary.json                all the numbers, ready for the report
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")

from config_physionet import PhysioNetConfig as PNConfig
from ensemble_combiner import StackedCombiner
from ensemble_xai.adapters import build_adapters, verify_adapter
from ensemble_xai.pipeline import generate_ensemble_assets_for_recording


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class", type=int, default=1)
    ap.add_argument("--combiner", default="ensemble_combiner.json")
    ap.add_argument("--out_dir", default=os.path.join("outputs", "ensemble_xai"))
    ap.add_argument("--no_disagreement", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_verify", action="store_true",
                    help="skip the adapter check (only if you just ran it and nothing changed)")
    a = ap.parse_args()

    PNConfig.load_label_vocab()
    class_names, lead_names = list(PNConfig.CLASS_NAMES), list(PNConfig.LEAD_NAMES)

    X = np.load(os.path.join(PNConfig.PROCESSED_PATH, "X_test.npy"), mmap_mode="r")
    y = np.load(os.path.join(PNConfig.PROCESSED_PATH, "y_test.npy")).astype(int)

    comb = StackedCombiner.load(a.combiner)
    adapters = build_adapters()
    assert comb.model_names == [ad.name for ad in adapters], "combiner order != adapter order"

    if not a.skip_verify:
        for ad in adapters:
            verify_adapter(ad, X, np.load(f"test_probs_{ad.name}.npy"))

    # Saved per-model probs -> the ensemble's own prediction on every test row.
    member_probs = [np.load(f"test_probs_{n}.npy") for n in comb.model_names]
    ens_probs = comb.predict_proba(member_probs)
    ens_pred = ens_probs.argmax(1)
    member_pred = np.stack([p.argmax(1) for p in member_probs], 1)      # (N, 3)
    n_agree = (member_pred == ens_pred[:, None]).sum(1)

    rng = np.random.default_rng(a.seed)
    chosen = []          # (row, class_index, why)

    for c, name in enumerate(class_names):
        ok = np.where((y == c) & (ens_pred == c) & (n_agree == len(adapters)))[0]
        if len(ok) == 0:                                   # fall back: correct, even if models differ
            ok = np.where((y == c) & (ens_pred == c))[0]
        if len(ok) == 0:
            print(f"  !! no correctly-classified {name} recording found -- skipping")
            continue
        conf = ens_probs[ok, c]
        pick = ok[np.argsort(-conf)][:max(10, a.per_class)]  # confident, but not always the same one
        for r in rng.choice(pick, size=min(a.per_class, len(pick)), replace=False):
            chosen.append((int(r), c, "correct"))

    if not a.no_disagreement:
        dis = np.where((n_agree < len(adapters)) & (ens_pred == y))[0]
        if len(dis):
            r = int(dis[np.argsort(-ens_probs[dis, y[dis]])[0]])
            chosen.append((r, int(y[r]), "disagreement"))

    os.makedirs(a.out_dir, exist_ok=True)
    print(f"\ngenerating figures for {len(chosen)} recordings "
          f"(a few minutes each -- 240 occlusion passes per model)\n")

    summaries = {}
    for i, (row, c, why) in enumerate(chosen, 1):
        label = class_names[c]
        tag = f"{label}_{row}" + ("_disagreement" if why == "disagreement" else "")
        save_dir = os.path.join(a.out_dir, tag)
        t0 = time.time()
        s = generate_ensemble_assets_for_recording(
            np.asarray(X[row]), adapters, np.ones(len(adapters)), label,
            save_dir, class_names, lead_names, combiner=comb,
        )
        s["test_row"] = row
        s["true_class"] = label
        s["selected_because"] = why
        s["correct"] = (s["predicted_class"] == label)
        summaries[tag] = s
        top = sorted(s["ensemble_lead_importance"].items(), key=lambda t: -t[1])[:3]
        print(f"  [{i}/{len(chosen)}] {tag:<24} pred={s['predicted_class']} "
              f"conf={s['confidence']:.3f}  models agreeing={s['n_models_agreeing']}/{s['n_models_total']}  "
              f"top leads={[f'{k} {v:.2f}' for k, v in top]}  ({time.time()-t0:.0f}s)")

    out_json = os.path.join(a.out_dir, "summary.json")
    json.dump({"combiner": a.combiner, "model_shares": comb.model_shares(),
               "recordings": summaries}, open(out_json, "w"), indent=2)
    print(f"\nsaved -> {out_json}")
    print(f"figures are in {a.out_dir}\\<CLASS>_<row>\\")


if __name__ == "__main__":
    main()