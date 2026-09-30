"""
fix_checkpoint.py -- make a usable PyTorch checkpoint file, whatever state it arrived in.

A .pth file is really a zip. If it gets extracted along the way, you end up with a FOLDER
called something.pth containing data.pkl, byteorder, version, data/ ... This script puts
it back together. It handles three situations, in this order:

  1. <target>.pth is already a working file  -> nothing to do
  2. <target>.pth.zip exists                 -> use whatever is inside it
  3. <target>.pth is an unzipped folder      -> zip it back up

In every case it verifies the result really loads before replacing anything, and the old
folder is kept as a backup rather than deleted.

Usage (from the project root):
    python fix_checkpoint.py teammates\\inception\\best_inception_5class_PROTECTED.pth
"""
import os
import shutil
import sys
import tempfile
import zipfile

import torch


def try_load(path):
    """Return True if torch can load this file."""
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def find_archive_root(folder):
    """An extracted checkpoint has data.pkl either directly inside, or one level down.
    Returns (source_folder, name_to_use_inside_the_zip) or (None, None)."""
    if os.path.exists(os.path.join(folder, "data.pkl")):
        return folder, "archive"
    subs = [d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
            and os.path.exists(os.path.join(folder, d, "data.pkl"))]
    if len(subs) == 1:
        return os.path.join(folder, subs[0]), subs[0]
    return None, None


def rebuild_from_folder(src, top, out_path):
    """Zip an extracted checkpoint back into a single file."""
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(src):
            for f in sorted(files):
                full = os.path.join(root, f)
                name = top + "/" + os.path.relpath(full, src).replace(os.sep, "/")
                # torch's internal files carry pre-1980 dates, which zipfile refuses
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                with open(full, "rb") as fh:
                    z.writestr(info, fh.read(), compress_type=zipfile.ZIP_STORED)


def backup(path):
    """Move something aside instead of deleting it. Returns the new path."""
    dest = path + "_BACKUP"
    n = 2
    while os.path.exists(dest):
        dest = f"{path}_BACKUP{n}"
        n += 1
    shutil.move(path, dest)
    return dest


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = os.path.normpath(sys.argv[1])
    zip_path = target + ".zip"

    # --- case 1: already fine -------------------------------------------------
    if os.path.isfile(target) and try_load(target):
        print(f"{target} is already a working checkpoint. Nothing to do.")
        return

    tmp_out = target + ".rebuilt"
    if os.path.exists(tmp_out):
        os.remove(tmp_out)

    # --- case 2: a .zip sits next to it ---------------------------------------
    if os.path.isfile(zip_path):
        print(f"found {os.path.basename(zip_path)} -- looking inside it")
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tmp)

            # (a) the zip contains the real .pth file
            found = [os.path.join(r, f) for r, _, fs in os.walk(tmp)
                     for f in fs if f.lower().endswith((".pth", ".pt"))]
            for f in found:
                if try_load(f):
                    shutil.copy2(f, tmp_out)
                    print(f"  the zip contained a working checkpoint ({os.path.basename(f)})")
                    break
            else:
                # (b) the zip contains the extracted folder structure
                roots = [tmp] + [os.path.join(tmp, d) for d in os.listdir(tmp)
                                 if os.path.isdir(os.path.join(tmp, d))]
                for r in roots:
                    src, top = find_archive_root(r)
                    if src:
                        rebuild_from_folder(src, top, tmp_out)
                        print("  the zip contained an extracted checkpoint -- rebuilt it")
                        break
                else:
                    print("  nothing usable inside the zip; trying the folder instead")

    # --- case 3: the .pth itself is an unzipped folder ------------------------
    if not os.path.exists(tmp_out) and os.path.isdir(target):
        src, top = find_archive_root(target)
        if not src:
            sys.exit(f"could not find data.pkl inside {target}. Contents: {os.listdir(target)}")
        rebuild_from_folder(src, top, tmp_out)
        print(f"rebuilt {os.path.basename(target)} from the unzipped folder")

    if not os.path.exists(tmp_out):
        sys.exit(f"nothing to work with: {target} is not a folder and no usable {os.path.basename(zip_path)} was found.")

    # --- verify before replacing anything ------------------------------------
    if not try_load(tmp_out):
        os.remove(tmp_out)
        sys.exit("the rebuilt file still does not load -- ask your teammate to re-send the checkpoint.")
    ck = torch.load(tmp_out, map_location="cpu", weights_only=False)
    print("verified: it loads. top-level keys:",
          list(ck.keys())[:5] if isinstance(ck, dict) else type(ck).__name__)

    if os.path.exists(target):
        moved = backup(target)
        print(f"old version moved to {os.path.basename(moved)}")
    shutil.move(tmp_out, target)
    size_mb = os.path.getsize(target) / 1e6
    print(f"done: {target} is now a single file ({size_mb:.1f} MB).")
    print("Delete the *_BACKUP folder and the .zip once everything works.")


if __name__ == "__main__":
    main()