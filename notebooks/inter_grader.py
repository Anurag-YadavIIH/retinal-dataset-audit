"""Item 4 step 3: inter-grader agreement -- the ceiling on model performance.

Two human experts annotate the same image and do not produce the same
mask. Their agreement is the practical upper bound on what any model can
be scored at, because a model is evaluated against *one* of them: a model
that matched observer 1 perfectly would still score only the
observer-1-vs-observer-2 agreement when graded against observer 2. Any
reported Dice should be read against this number, not against 1.0.

**Reported per dataset, never pooled.** CHASE_DB1 and DRIVE were annotated
by different people, under different conventions, on images acquired
differently. Averaging them would hide exactly the kind of
between-dataset difference this project has spent its length finding --
and if the two ceilings differ materially, that is itself the finding:
"the inter-grader ceiling" would not be one number but a property of each
annotation effort.

**Provenance caveat, carried into every number below.** DRIVE's official
distribution withholds its test annotations (WALKTHROUGH.md §13), so the
second-observer masks here come from a third-party Kaggle re-upload
(`ahtcmstp/retina`) that appears to be a copy of the pre-Grand-Challenge
distribution. Provenance is unverified beyond the structural checks this
script re-runs and reports: dimension agreement, binarity, plausible
vessel fraction, non-emptiness, observers differing from each other, and
consistency with the official *training* set's first-observer
conventions. CHASE_DB1's two observers ship together in its normal
distribution and need no such caveat.

Run: python notebooks/inter_grader.py
Output: printed report + artifacts/inter_grader.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.mask_quality import load_binary_mask  # noqa: E402
from retinaprep.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

CHASE_DIR = Path("D:/retinaprep_data/chasedb1")
DRIVE_DIR = Path("D:/retinaprep_data/drive")
ARTIFACTS = REPO_ROOT / "artifacts"


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return float(2 * inter / total) if total else 1.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def _pairs_chase() -> list[tuple[str, Path, Path, Path]]:
    out = []
    for img in sorted(CHASE_DIR.glob("Image_*.jpg")):
        stem = img.stem
        out.append((stem, img, CHASE_DIR / f"{stem}_1stHO.png", CHASE_DIR / f"{stem}_2ndHO.png"))
    return out


def _pairs_drive() -> list[tuple[str, Path, Path, Path]]:
    out = []
    for i in range(1, 21):
        s = f"{i:02d}"
        out.append(
            (s, DRIVE_DIR / f"{s}_test.tif",
             DRIVE_DIR / f"{s}_manual1.gif", DRIVE_DIR / f"{s}_manual2.gif")
        )
    return out


def structural_checks(name: str, pairs) -> dict:
    """Re-run the provenance checks here rather than trusting a prior
    session, so the JSON output carries its own evidence."""
    from PIL import Image

    dims_ok, identical, empty, fracs1, fracs2 = True, 0, 0, [], []
    for _sid, img, m1, m2 in pairs:
        a, b = load_binary_mask(str(m1)), load_binary_mask(str(m2))
        with Image.open(img) as im:
            ish = np.asarray(im).shape[:2]
        dims_ok &= (ish == a.shape == b.shape)
        identical += int(np.array_equal(a, b))
        empty += int(a.sum() == 0) + int(b.sum() == 0)
        fracs1.append(a.mean())
        fracs2.append(b.mean())
    report = {
        "dims_match_all": bool(dims_ok),
        "identical_observer_pairs": identical,
        "empty_masks": empty,
        "obs1_fg_fraction_mean": float(np.mean(fracs1)),
        "obs2_fg_fraction_mean": float(np.mean(fracs2)),
    }
    logger.info("%s structural checks: %s", name, report)
    return report


def agreement(name: str, pairs) -> tuple[pd.DataFrame, dict]:
    rows = []
    for sid, _img, m1, m2 in pairs:
        a, b = load_binary_mask(str(m1)), load_binary_mask(str(m2))
        rows.append(
            {
                "dataset": name,
                "image_id": sid,
                "dice": dice(a, b),
                "iou": iou(a, b),
                "obs1_fg_fraction": float(a.mean()),
                "obs2_fg_fraction": float(b.mean()),
            }
        )
    df = pd.DataFrame(rows)
    d, j = df["dice"], df["iou"]
    # SE of the mean, to make the small-n limit explicit at the point of use
    se = d.std(ddof=1) / np.sqrt(len(d))
    summary = {
        "n_images": len(df),
        "dice_mean": float(d.mean()),
        "dice_std": float(d.std(ddof=1)),
        "dice_min": float(d.min()),
        "dice_max": float(d.max()),
        "dice_se": float(se),
        "dice_ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
        "iou_mean": float(j.mean()),
        "iou_std": float(j.std(ddof=1)),
        "iou_min": float(j.min()),
        "iou_max": float(j.max()),
    }
    return df, summary


def main() -> None:
    result = {}
    frames = []
    for name, pairs in [("CHASE_DB1", _pairs_chase()), ("DRIVE", _pairs_drive())]:
        if not pairs or not pairs[0][1].exists():
            logger.warning("%s not found on disk -- skipping", name)
            continue
        checks = structural_checks(name, pairs)
        df, summary = agreement(name, pairs)
        frames.append(df)
        result[name] = {"structural_checks": checks, "agreement": summary}

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_parquet(ARTIFACTS / "inter_grader_per_image.parquet", index=False)
    with open(ARTIFACTS / "inter_grader.json", "w") as fh:
        json.dump(result, fh, indent=2)

    print("\n=== Inter-grader agreement: the ceiling on model performance ===")
    print("Reported per dataset. NOT pooled: different annotators, conventions and cameras.\n")
    for name, r in result.items():
        a = r["agreement"]
        print(f"{name}  (n={a['n_images']})")
        print(f"  Dice  mean {a['dice_mean']:.4f}  sd {a['dice_std']:.4f}  "
              f"range {a['dice_min']:.4f}-{a['dice_max']:.4f}  "
              f"95% CI [{a['dice_ci95'][0]:.4f}, {a['dice_ci95'][1]:.4f}]")
        print(f"  IoU   mean {a['iou_mean']:.4f}  sd {a['iou_std']:.4f}  "
              f"range {a['iou_min']:.4f}-{a['iou_max']:.4f}")
        c = r["structural_checks"]
        print(f"  checks: dims_ok={c['dims_match_all']} identical_pairs="
              f"{c['identical_observer_pairs']} empty={c['empty_masks']} "
              f"fg {c['obs1_fg_fraction_mean']:.4f} vs {c['obs2_fg_fraction_mean']:.4f}")
        print()

    if len(result) == 2:
        a, b = (result[k]["agreement"]["dice_mean"] for k in result)
        print(f"Difference between the two ceilings: {abs(a-b):.4f} Dice")


if __name__ == "__main__":
    main()
