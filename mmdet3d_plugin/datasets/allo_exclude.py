"""Skip ALLO frames flagged as camera-inconsistent.

An exclusion CSV (columns: split,seq,camera,index,...) lists frames whose
background is inconsistent across the camera's frame stack. This module drops
those frames from a loaded ``data_infos`` DataFrame so they are never sampled
during training/eval.

The CSV path is taken from the ``ALLO_EXCLUDE_CSV`` environment variable. If the
variable is unset or the file is missing, nothing is excluded -- i.e. the loader
behaves exactly as before. This keeps old runs (which don't set the variable)
bit-for-bit identical while new runs opt in by exporting ALLO_EXCLUDE_CSV.

Frame paths look like ``<data_root>/<split>/<seq>/<cam>/images/<index>_*.png``,
so each frame maps to the key ``(split, seq, cam, index)`` used to match the CSV.

Standalone check (reports how many frames each split would skip and flags
exclusion rows that match nothing on disk)::

    python -m mmdet3d_plugin.datasets.allo_exclude <data_root> --csv camera_inconsistent.csv
"""
import os
from pathlib import Path

import pandas as pd

EXCLUDE_ENV = "ALLO_EXCLUDE_CSV"


def load_exclusion_keys(csv_path):
    """Return a set of (split, seq, camera, index) string keys from the CSV."""
    df = pd.read_csv(csv_path)
    required = {"split", "seq", "camera", "index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    return {
        (str(r["split"]).strip(), str(r["seq"]).strip(),
         str(r["camera"]).strip(), str(r["index"]).strip())
        for _, r in df.iterrows()
    }


def frame_key(img_path, data_root):
    """Map a frame path to its (split, seq, camera, index) key."""
    rel = Path(img_path).relative_to(Path(data_root))
    parts = rel.parts  # (split, seq, cam, images, <index>_<name>.png)
    split, seq, cam = parts[0], parts[1], parts[2]
    index = rel.name.split("_", 1)[0]
    return (split, seq, cam, index)


def filter_excluded(data_infos, data_root, csv_path=None, verbose=True):
    """Drop rows of ``data_infos`` whose frame key is in the exclusion CSV.

    No-op (returns the input unchanged) when no CSV is configured/found.
    """
    csv_path = csv_path or os.environ.get(EXCLUDE_ENV)
    if not csv_path or not os.path.exists(csv_path):
        return data_infos
    keys = load_exclusion_keys(csv_path)
    if not keys:
        return data_infos
    mask = data_infos["img_path"].map(
        lambda p: frame_key(p, data_root) not in keys)
    kept = data_infos[mask].reset_index(drop=True)
    if verbose:
        print(f"[allo_exclude] {csv_path}: skipped {len(data_infos) - len(kept)} "
              f"/ {len(data_infos)} frames ({len(keys)} exclusion entries)")
    return kept


def _main():
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Check ALLO exclusion CSV vs a dataset root")
    ap.add_argument("data_root", help="ALLO dataset root containing <split>/<seq>/<cam>/images")
    ap.add_argument("--csv", default=os.environ.get(EXCLUDE_ENV), help="exclusion CSV path")
    ap.add_argument("--img-glob", default="*normal.png")
    args = ap.parse_args()

    if not args.csv:
        ap.error(f"no CSV given (pass --csv or set {EXCLUDE_ENV})")

    keys = load_exclusion_keys(args.csv)
    print(f"Loaded {len(keys)} exclusion entries from {args.csv}")

    root = Path(args.data_root)
    matched = set()
    total = skipped = 0
    for split in ("train", "test"):
        sp = root / split
        if not sp.is_dir():
            continue
        for f in sp.glob(f"**/images/{args.img_glob}"):
            total += 1
            k = frame_key(f, root)
            if k in keys:
                skipped += 1
                matched.add(k)
    print(f"Frames found: {total} | would skip: {skipped}")
    unmatched = keys - matched
    if unmatched:
        print(f"WARNING: {len(unmatched)} exclusion entries matched no frame on disk:")
        for k in sorted(unmatched)[:20]:
            print("   ", k)
        if len(unmatched) > 20:
            print(f"    ... and {len(unmatched) - 20} more")
    else:
        print("All exclusion entries matched a frame on disk.")


if __name__ == "__main__":
    _main()
