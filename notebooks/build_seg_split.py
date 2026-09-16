"""Build and persist a segmentation split, once, to be filtered later.

Same discipline as the classification side (docs/notes.md, item 1): the
split is computed once and written to disk, and any later subset is a
*filter* of that file rather than a fresh recompute — because
recomputing a grouped split on a reduced pool reshuffles fold membership
and silently changes what is being compared.

Grouping is by `patient_id`, and it is worth being explicit that this is
**vacuous on IDRiD**: 81 images from 81 patients, one each, so grouping
by patient and splitting by image are the same operation. The discipline
is applied anyway so the code path is identical everywhere, and because
CHASE_DB1 (14 subjects, two eyes each) is the one segmentation dataset
where it does real work.

For IDRiD the dataset's own train/test division is respected rather than
replaced — it is the split the literature reports against, and
substituting a private one would make every number incomparable. The
validation fold is carved out of the official training set, patient-grouped.

Run: python notebooks/build_seg_split.py <artifacts_dir> [--val-frac 0.2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.utils import get_logger, set_global_seed  # noqa: E402

logger = get_logger(__name__)


def build_split(manifest: pd.DataFrame, val_frac: float, seed: int) -> dict:
    from sklearn.model_selection import GroupShuffleSplit

    if "source_split" in manifest.columns and manifest["source_split"].notna().any():
        train_pool = manifest[manifest["source_split"] == "train"].reset_index(drop=True)
        test = manifest[manifest["source_split"] == "test"].reset_index(drop=True)
        logger.info(
            "Using the dataset's own train/test division (%d/%d) -- it is what "
            "published numbers are reported against",
            len(train_pool), len(test),
        )
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr_idx, te_idx = next(gss.split(manifest, groups=manifest["patient_id"]))
        train_pool = manifest.iloc[tr_idx].reset_index(drop=True)
        test = manifest.iloc[te_idx].reset_index(drop=True)
        logger.info("No dataset split shipped; made a patient-grouped %d/%d division",
                    len(train_pool), len(test))

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(gss.split(train_pool, groups=train_pool["patient_id"]))
    train, val = train_pool.iloc[tr_idx], train_pool.iloc[va_idx]

    for name, part in (("train", train), ("val", val), ("test", test)):
        logger.info("  %-5s %3d images, %3d patients", name, len(part),
                    part["patient_id"].nunique())

    overlap = (
        set(train["patient_id"]) & set(val["patient_id"])
        | set(train["patient_id"]) & set(test["patient_id"])
        | set(val["patient_id"]) & set(test["patient_id"])
    )
    if overlap:
        raise ValueError(f"patient overlap across folds: {sorted(overlap)[:10]}")
    logger.info("  patient overlap across folds: 0")

    return {
        "train": train["image_path"].tolist(),
        "val": val["image_path"].tolist(),
        "test": test["image_path"].tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed)
    artifacts = Path(args.artifacts_dir)
    manifest = pd.read_parquet(artifacts / "manifest.parquet")
    split = build_split(manifest, args.val_frac, args.seed)

    out = artifacts / "splits"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "seg_split.json", "w") as fh:
        json.dump(split, fh, indent=2)
    print(f"Wrote {out / 'seg_split.json'}")


if __name__ == "__main__":
    main()
