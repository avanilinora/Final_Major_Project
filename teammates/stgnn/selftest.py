"""
selftest.py — confirms this wrapper reproduces the submitted probabilities.

Run from this folder, with X_test.npy and probs_test.npy alongside it (or pass
their paths):

    python selftest.py
    python selftest.py --x path/to/X_test.npy --probs path/to/probs_test.npy
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

from predict_wrapper import predict_proba  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--x", default="X_test.npy")
ap.add_argument("--probs", default="probs_test.npy")
ap.add_argument("--n", type=int, default=64)
args = ap.parse_args()

X = np.asarray(np.load(args.x, mmap_mode="r")[: args.n])
saved = np.load(args.probs)[: args.n]

got = predict_proba(X)
print("shape:", got.shape, " rows sum to 1:", np.allclose(got.sum(1), 1, atol=1e-4))
print("max abs diff vs saved probs:", np.abs(got - saved).max())
print("argmax agreement:", (got.argmax(1) == saved.argmax(1)).mean())

# batch-size robustness: 1..64 must agree with the batched call
for n in (1, 7, args.n):
    sub = predict_proba(X[:n])
    print(f"batch {n:>2}: max diff vs batched {np.abs(sub - got[:n]).max():.3e}")
