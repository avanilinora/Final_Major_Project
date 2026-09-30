"""
cleanup_project.py  --  tidy the project folder WITHOUT losing anything.

Run it with no arguments and it only LOOKS: it prints what it would move, and how much
space that frees. Nothing changes until you add --apply, and even then files are MOVED
into _archive_<date>/ rather than deleted, so you can put anything back by dragging it.

    python cleanup_project.py                 # just show me the plan
    python cleanup_project.py --apply         # do it (moves to _archive_<date>/)
    python cleanup_project.py --apply --caches-too   # also DELETE __pycache__ folders

Three groups:
  KEEP          never touched -- your code, models, data, results
  SAFE          leftovers from tonight that nothing uses -> moved to _archive_<date>/
  YOU DECIDE    listed with sizes, never moved; delete by hand if you want

Anything the script does not recognise falls in YOU DECIDE. It errs towards keeping.
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import os
import shutil

# --- never touched, whatever else the rules say ----------------------------
KEEP_DIRS = {
    ".venv", ".git", "data", "models", "outputs", "utils", "xai", "ensemble_xai",
    "xai_report", "teammates", "__pycache__",          # caches handled separately
}
KEEP_FILES = {
    # labels and the three members' probabilities -- the ensemble is rebuilt from these
    "y_val.npy", "y_test.npy",
    "val_probs_cnn_bilstm.npy", "test_probs_cnn_bilstm.npy",
    "val_probs_inception.npy", "test_probs_inception.npy",
    "val_probs_stgnn.npy", "test_probs_stgnn.npy",
    "test_probs_ensemble.npy",
    # the final ensemble and the numbers quoted in the report
    "ensemble_combiner.json", "ensemble_improve_results.json",
    "ensemble_eval_results.json", "ensemble_deletion_results.json",
    "ensemble_deletion_curve.png", "ablation_results.json", "ensemble_results.json",
    # project files
    "requirements.txt", ".gitignore", "README.md",
}

# --- leftovers from tonight that nothing reads any more ---------------------
SAFE_PATTERNS = [
    ("*_BACKUP",             "backup left by fix_checkpoint.py"),
    ("*_BACKUP[0-9]",        "backup left by fix_checkpoint.py"),
    ("*_UNZIPPED_BACKUP",    "backup left by fix_checkpoint.py"),
    ("*.pth.zip",            "the zipped checkpoint, already unpacked"),
    ("*.pth.rebuilt",        "half-finished rebuild"),
    ("*xgboost*",            "XGBoost was dropped from the ensemble"),
]
# The same backup leftovers, but hunted for INSIDE kept folders such as teammates/.
# Deliberately narrower than SAFE_PATTERNS: only files this session created.
NESTED_SAFE_PATTERNS = ["*_BACKUP", "*_BACKUP[0-9]", "*_UNZIPPED_BACKUP", "*.pth.zip", "*.pth.rebuilt"]
NESTED_SCAN_DIRS = ["teammates"]

SAFE_NAMES = {
    "case1.npy": "scratch test file",
    "bad.npy": "scratch test file",
}

# --- worth a look, never moved automatically -------------------------------
DECIDE_NAMES = {
    "serve.py": "the web API -- keep only if you present the web app",
    "test_page.html": "the web test page -- keep only if you present the web app",
    "test_case.npy": "your test recording; handy for demos",
    "fix_checkpoint.py": "one-off repair tool; harmless to keep",
    "ensemble_eval.py": "superseded by ensemble_improve.py, but it produced numbers you may cite",
}
DECIDE_PATTERNS = [
    ("stability_no_*.json", "from the older stability analysis; check if your report cites it"),
]


def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def classify(name, root="."):
    """-> (group, reason). Groups: keep / safe / decide."""
    path = os.path.join(root, name)
    if name in KEEP_FILES or name in KEEP_DIRS:
        return "keep", ""
    if name in SAFE_NAMES:
        return "safe", SAFE_NAMES[name]
    if name in DECIDE_NAMES:
        return "decide", DECIDE_NAMES[name]
    for pat, why in SAFE_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return "safe", why
    for pat, why in DECIDE_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return "decide", why
    if os.path.isdir(path):
        return "keep", ""                      # unknown folders are left alone
    if name.endswith(".py"):
        return "keep", ""                      # never move source code
    return "decide", "not recognised -- check before removing"


def find_nested_safe():
    """Checkpoint backups sitting inside folders we otherwise never touch."""
    found = []
    for base in NESTED_SCAN_DIRS:
        if not os.path.isdir(base):
            continue
        for cur, dirs, files in os.walk(base):
            for name in list(dirs) + files:
                if any(fnmatch.fnmatch(name, p) for p in NESTED_SAFE_PATTERNS):
                    found.append(os.path.join(cur, name))
                    if name in dirs:
                        dirs.remove(name)          # don't descend into it
    return found


def find_caches(root="."):
    out = []
    for cur, dirs, _ in os.walk(root):
        if any(p in cur.split(os.sep) for p in (".venv", ".git", "_archive")):
            dirs[:] = []
            continue
        for d in list(dirs):
            if d == "__pycache__":
                out.append(os.path.join(cur, d))
                dirs.remove(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move the SAFE items")
    ap.add_argument("--caches-too", action="store_true", help="also delete every __pycache__ folder")
    a = ap.parse_args()

    entries = sorted(os.listdir("."))
    groups = {"keep": [], "safe": [], "decide": []}
    for name in entries:
        if name.startswith("_archive_"):
            continue
        g, why = classify(name)
        groups[g].append((name, why, size_of(name)))

    print(f"\n{'=' * 72}\nPROJECT FOLDER: {os.path.abspath('.')}\n{'=' * 72}")

    print(f"\nKEEP -- not touched ({len(groups['keep'])} items, "
          f"{human(sum(s for _, _, s in groups['keep']))})")
    for n, _, s in groups["keep"]:
        print(f"    {n:<44} {human(s):>10}")

    for p in find_nested_safe():
        groups["safe"].append((p, "leftover from repairing the checkpoint", size_of(p)))
    groups["safe"].sort()

    safe_total = sum(s for _, _, s in groups["safe"])
    print(f"\nSAFE TO ARCHIVE ({len(groups['safe'])} items, {human(safe_total)})")
    if not groups["safe"]:
        print("    nothing -- already clean")
    for n, why, s in groups["safe"]:
        print(f"    {n:<44} {human(s):>10}   {why}")

    print(f"\nYOU DECIDE -- never moved by this script ({len(groups['decide'])} items)")
    for n, why, s in groups["decide"]:
        print(f"    {n:<44} {human(s):>10}   {why}")

    caches = find_caches()
    cache_total = sum(size_of(c) for c in caches)
    print(f"\nPYTHON CACHES: {len(caches)} __pycache__ folders, {human(cache_total)}")
    print("    (safe to delete any time -- Python rebuilds them; use --caches-too)")

    if not a.apply:
        print(f"\n{'-' * 72}")
        print("This was a dry run. Nothing changed.")
        if safe_total:
            print(f"Run  python cleanup_project.py --apply   to move the {len(groups['safe'])} "
                  f"SAFE items into _archive_<date>/ and free {human(safe_total)}.")
        return

    # ---- apply ----
    if groups["safe"]:
        dest = "_archive_" + datetime.date.today().isoformat()
        os.makedirs(dest, exist_ok=True)
        moved = 0
        for n, _, _ in groups["safe"]:
            try:
                target = os.path.join(dest, n)                 # keeps teammates/... structure
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.move(n, target)
                moved += 1
            except Exception as e:
                print(f"  could not move {n}: {e}")
        print(f"\nmoved {moved} items into {dest}\\  ({human(safe_total)})")
        print("Nothing was deleted. Check your project still runs, then delete that folder.")

    if a.caches_too:
        for c in caches:
            shutil.rmtree(c, ignore_errors=True)
        print(f"deleted {len(caches)} __pycache__ folders ({human(cache_total)})")

    print("\nNow check nothing broke:")
    print("    python ensemble_combiner.py")
    print("    python analyze_ecg.py --input test_case.npy")


if __name__ == "__main__":
    main()