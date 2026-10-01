"""
ensemble_xai/adapters.py
========================
Every ensemble member is wrapped so it exposes:
    .name
    .predict_proba(x: np.ndarray (N, 5000, 12)) -> np.ndarray (N, 5)   # rows sum to 1
Class order for ALL members: [SB, AFIB, SR, ST, SVT].

Folder layout this file expects (project root = Final_Major_Project):

    teammates/
      inception/
        predict_wrapper.py                       (Inception teammate's wrapper)
        train_inception_5class_recovered.py      (defines InceptionTime )
        best_inception_5class_PROTECTED.pth      (checkpoint )
      stgnn/
        utils/predict_wrapper.py, model_stgnn_rr.py, 
        outputs/checkpoints_stgnn_rr/stgnn_rr_best.pt
        outputs/checkpoints_stgnn_rr/calibration_rr_epoch7.json

Why the special loader: both teammates ship a file called predict_wrapper.py, and the
STGNN folder ships its own config_physionet.py (different bandpass cut-off, extra
fields). A plain `import` would mix them up with each other and with cnn-bilstm
config_physionet.py. _load_isolated() imports each teammate's code with only their
own folder visible, then puts cnn-bilstm  modules back.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
INCEPTION_DIR = os.path.join(ROOT, "teammates", "inception")
STGNN_DIR = os.path.join(ROOT, "teammates", "stgnn", "utils")
STGNN_CKPT_DIR = os.path.join(ROOT, "teammates", "stgnn", "outputs", "checkpoints_stgnn_rr")


# ---------------------------------------------------------------------------
# Adapter classes
# ---------------------------------------------------------------------------

class CallableAdapter:
    """Wrap any function fn((N,T,L) ndarray) -> (N,C) probabilities."""
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def predict_proba(self, x):
        return np.asarray(self.fn(x), dtype=np.float64)


class TorchAdapter:
    """PyTorch model. layout='NTL' if the model takes (N,T,L), 'NLT' if (N,L,T)."""
    def __init__(self, name, model, device, layout="NTL", apply_softmax=True):
        import torch
        self.torch, self.name, self.device = torch, name, device
        self.model = model.to(device).eval()
        self.layout, self.apply_softmax = layout, apply_softmax

    def predict_proba(self, x):
        torch = self.torch
        with torch.no_grad():
            t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=self.device)
            if self.layout == "NLT":
                t = t.transpose(1, 2)
            out = self.model(t)
            out = out[0] if isinstance(out, tuple) else out
            if self.apply_softmax:
                out = torch.softmax(out, dim=1)
            return out.cpu().numpy().astype(np.float64)


def verify_adapter(adapter, X_test, saved_test_probs, n=64, atol=1e-3):
    """MUST pass before any XAI run: adapter output on the first n test rows has to
    match the probabilities that were used in ensemble_eval.py."""
    got = adapter.predict_proba(np.asarray(X_test[:n]))
    saved = saved_test_probs[:n]
    diff = np.abs(got - saved).max()
    agree = (got.argmax(1) == saved.argmax(1)).mean()
    rows_ok = np.allclose(got.sum(1), 1.0, atol=1e-4)
    print(f"[verify] {adapter.name:<12} shape={got.shape} rows-sum-to-1={rows_ok} "
          f"max|diff|={diff:.5f} argmax-agreement={agree:.3f}")
    if got.shape != saved.shape or diff > atol or not rows_ok:
        raise RuntimeError(f"{adapter.name}: adapter does not reproduce the saved probs "
                           f"(wrong folder / checkpoint / transpose / class order?)")
    return True


# ---------------------------------------------------------------------------
# Isolated import of a teammate's module
# ---------------------------------------------------------------------------

def _load_isolated(folder, filename, alias):
    """Import folder/filename as module `alias`, letting its own `import x` lines see
    only the .py files in `folder` for any name that clashes with cnn-bilstm."""
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path} -- see the folder layout at the top of adapters.py")

    local_names = {f[:-3] for f in os.listdir(folder) if f.endswith(".py")}
    saved = {n: sys.modules.pop(n) for n in list(sys.modules) if n in local_names}
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location(alias, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    finally:
        if folder in sys.path:
            sys.path.remove(folder)
        for n in local_names:            # drop the teammate's bare-named copies ...
            sys.modules.pop(n, None)
        sys.modules.update(saved)        # ... and put cnn-bilstm back
    return mod


# ---------------------------------------------------------------------------
# Build the three members  (order MUST match the weights vector)
# ---------------------------------------------------------------------------

def build_adapters():
    """Returns [cnn_bilstm, inception, stgnn]."""
    import torch
    from config_physionet import PhysioNetConfig as PNConfig
    from models.cnn_bilstm2 import CNNBiLSTM
    PNConfig.load_label_vocab()

    # 1) your CNN-BiLSTM -- takes (N, 5000, 12) directly
    m = CNNBiLSTM(num_classes=PNConfig.NUM_CLASSES, num_leads=PNConfig.NUM_LEADS)
    ck = torch.load(os.path.join(PNConfig.CHECKPOINT_PATH, "physionet_cnn_bilstm2_best_model.pth"),
                    map_location=PNConfig.DEVICE)
    m.load_state_dict(ck["model_state_dict"])
    mine = TorchAdapter("cnn_bilstm", m, PNConfig.DEVICE, layout="NTL")

    # 2) InceptionTime -- teammate's wrapper already transposes and reorders classes
    inc_mod = _load_isolated(INCEPTION_DIR, "predict_wrapper.py", "inception_predict_wrapper")
    inc = inc_mod.InceptionPredictor(
        checkpoint_path=os.path.join(INCEPTION_DIR, "best_inception_5class_PROTECTED.pth"))
    inception = CallableAdapter("inception", inc.predict_proba)

    # 3) STGNN+RR -- teammate's wrapper runs its own beat-extraction pipeline
    os.environ["TRUSTECG_CKPT_DIR"] = STGNN_CKPT_DIR
    st_mod = _load_isolated(STGNN_DIR, "predict_wrapper.py", "stgnn_predict_wrapper")
    stgnn = CallableAdapter("stgnn", st_mod.predict_proba)
    stgnn.predict_status = st_mod.predict_status     # handy for the XAI write-up

    return [mine, inception, stgnn]


# ---------------------------------------------------------------------------
# Quick check:  python -m ensemble_xai.adapters
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config_physionet import PhysioNetConfig as PNConfig
    PNConfig.load_label_vocab()
    X = np.load(os.path.join(PNConfig.PROCESSED_PATH, "X_test.npy"), mmap_mode="r")
    saved = {"cnn_bilstm": "test_probs_cnn_bilstm.npy",
             "inception": "test_probs_inception.npy",
             "stgnn": "test_probs_stgnn.npy"}
    for ad in build_adapters():
        verify_adapter(ad, X, np.load(os.path.join(ROOT, saved[ad.name])))
    print("\nAll three adapters reproduce the saved probabilities. Safe to run the XAI.")
