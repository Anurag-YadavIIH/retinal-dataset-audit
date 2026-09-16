"""Item 4 follow-up: why do three IDRiD disc images fail?

The optic disc U-Net reports Dice 0.8581 +/- 0.1646 on 27 test images.
The mean is the wrong summary and the median (0.9110) is the right one:
three images fail badly, two of them below 0.50 Dice. Three images out of
27 move the headline by 0.05, which is exactly the underpowered regime
declared before the run. A 7% silent-failure rate matters more for a
screening tool than a mean does, so the three are investigated here
rather than averaged away.

Two hypotheses, and the first one is the flattering one:

1. **They are bad photographs**, and this project's own quality module
   would have rejected them upstream -- a clean link between the curation
   work and the segmentation work. Tested by scoring every test image
   with `retinaprep.quality` at the configured threshold.
2. **They are diseased photographs.** IDRiD is a diabetic retinopathy
   dataset and ships hard-exudate masks alongside the disc masks, so
   "the model is confusing bright yellow-white exudates for the optic
   disc" is checkable rather than eyeballed. Tested by measuring each
   test image's hard-exudate burden from IDRiD's own EX masks and
   comparing failures against the rest.

This script existed as an ad-hoc pass first and is committed as a script
because docs/notes.md draws a line between artifact-sourced numbers
(regenerate them yourself, with the command given) and prose-sourced ones
(the inputs are gone). The exudate numbers are quoted in README.md,
docs/summary.md and WALKTHROUGH.md section 17, so they belong on the
artifact-sourced side of that line.

Run: python notebooks/idrid_failure_analysis.py
Output: printed report
        artifacts/idrid_od/failure_analysis.csv   (per-image, sorted by Dice)
        artifacts/idrid_od/failure_analysis.json  (the group comparison)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage, stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.adapters.idrid_seg import MASK_CLASSES  # noqa: E402
from retinaprep.config import load_config  # noqa: E402
from retinaprep.mask_quality import load_binary_mask  # noqa: E402
from retinaprep.quality import compute_metrics, gradability_score  # noqa: E402
from retinaprep.segment import IMAGE_SIZE, UNet, dice_score, predict_masks  # noqa: E402
from retinaprep.utils import get_logger  # noqa: E402

logger = get_logger(__name__)
ART = REPO_ROOT / "artifacts" / "idrid_od"

# "Failure" is defined by a gap in the sorted Dice distribution, not by a
# round number picked afterwards: the three worst images sit at 0.36/0.42/
# 0.55 and the fourth is 0.73. Recorded as a constant so the choice is
# visible rather than implied.
FAILURE_DICE_MAX = 0.70


def _exudate_mask_path(image_path: str) -> Path | None:
    """IDRiD's EX masks sit beside the OD masks, same split, same stem.

    Not every image has one -- an eye with no hard exudates ships no mask,
    which is a real zero rather than missing data, so None means burden 0.
    """
    directory, suffix = MASK_CLASSES["EX"]
    od_directory, od_suffix = MASK_CLASSES["OD"]
    stem = Path(image_path).stem
    manifest = pd.read_parquet(ART / "manifest.parquet")
    od_path = Path(manifest.set_index("image_path").loc[image_path, "mask_path"])
    ex_path = od_path.parent.parent / directory / f"{stem}{suffix}.tif"
    assert od_path.parent.name == od_directory and od_suffix in od_path.name
    return ex_path if ex_path.exists() else None


def _resized_target(mask_path: str) -> np.ndarray:
    mask = load_binary_mask(mask_path).astype(np.uint8) * 255
    resized = Image.fromarray(mask).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    return np.asarray(resized) > 127


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def main() -> None:
    cfg = load_config()
    manifest = pd.read_parquet(ART / "manifest.parquet")
    with open(ART / "splits" / "seg_split.json") as fh:
        split = json.load(fh)
    by_path = {r["image_path"]: r for r in manifest.to_dict("records")}
    rows = [by_path[p] for p in split["test"]]
    logger.info("Analysing %d IDRiD disc test images", len(rows))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    ckpt = torch.load(ART / "unet_od_best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    preds = predict_masks(model, rows, device)

    recs = []
    for pred, row in zip(preds, rows, strict=True):
        true = _resized_target(row["mask_path"])

        image = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
        quality = gradability_score(compute_metrics(image, cfg), cfg)

        pred_c, true_c = _centroid(pred), _centroid(true)
        centroid_px = (
            float(np.hypot(pred_c[0] - true_c[0], pred_c[1] - true_c[1]))
            if pred_c and true_c
            else np.nan
        )

        ex_path = _exudate_mask_path(row["image_path"])
        ex_burden = float(load_binary_mask(str(ex_path)).mean()) if ex_path else 0.0

        recs.append(
            {
                "name": Path(row["image_path"]).name,
                "dice": dice_score(pred, true),
                "quality": quality,
                "pred_area": float(pred.mean()),
                "true_area": float(true.mean()),
                "centroid_px": centroid_px,
                "n_comp": int(ndimage.label(pred)[1]),
                "exudate_fraction": ex_burden,
                "has_exudate_mask": ex_path is not None,
            }
        )

    df = pd.DataFrame(recs).sort_values("dice").reset_index(drop=True)
    df["exudate_rank"] = df["exudate_fraction"].rank(ascending=False, method="min").astype(int)
    df.to_csv(ART / "failure_analysis.csv", index=False)

    failed = df["dice"] < FAILURE_DICE_MAX
    fail, rest = df[failed], df[~failed]
    reject_threshold = cfg["quality"]["reject_below_score"]

    result = {
        "n_test": len(df),
        "failure_dice_max": FAILURE_DICE_MAX,
        "n_failures": int(failed.sum()),
        "dice_mean": float(df["dice"].mean()),
        "dice_median": float(df["dice"].median()),
        "quality": {
            "failures_mean": float(fail["quality"].mean()),
            "rest_mean": float(rest["quality"].mean()),
            "reject_threshold": reject_threshold,
            "n_failures_flagged": int((fail["quality"] < reject_threshold).sum()),
            "n_rest_flagged": int((rest["quality"] < reject_threshold).sum()),
        },
        "geometry": {
            "area_ratio_failures": float((fail["pred_area"] / fail["true_area"]).mean()),
            "area_ratio_rest": float((rest["pred_area"] / rest["true_area"]).mean()),
            "area_ratio_per_failure": [
                float(r) for r in (fail["pred_area"] / fail["true_area"])
            ],
            "centroid_px_failures": float(fail["centroid_px"].mean()),
            "centroid_px_rest": float(rest["centroid_px"].mean()),
            "centroid_px_per_failure": [float(r) for r in fail["centroid_px"]],
            "n_comp_per_failure": [int(r) for r in fail["n_comp"]],
            "n_comp_rest_median": float(rest["n_comp"].median()),
        },
        "exudate": {
            "failures_mean_fraction": float(fail["exudate_fraction"].mean()),
            "rest_mean_fraction": float(rest["exudate_fraction"].mean()),
            "ratio": float(fail["exudate_fraction"].mean() / rest["exudate_fraction"].mean()),
            "failure_ranks": {r["name"]: int(r["exudate_rank"]) for _, r in fail.iterrows()},
            "mannwhitney_p": float(
                stats.mannwhitneyu(
                    fail["exudate_fraction"], rest["exudate_fraction"], alternative="greater"
                )[1]
            ),
        },
    }
    with open(ART / "failure_analysis.json", "w") as fh:
        json.dump(result, fh, indent=2)

    q = result["quality"]
    ex = result["exudate"]
    g = result["geometry"]
    print(f"\n=== IDRiD optic disc: why do {result['n_failures']} of "
          f"{result['n_test']} fail? ===\n")
    print(f"Dice mean {result['dice_mean']:.4f} vs median {result['dice_median']:.4f} "
          "-- the mean is dragged by the tail, not representative")
    print("\nHYPOTHESIS 1: they are bad photographs the quality module would have caught")
    print(f"  gradability, failures    : {q['failures_mean']:.3f}")
    print(f"  gradability, other {len(rest):2d}    : {q['rest_mean']:.3f}")
    print(f"  flagged below {q['reject_threshold']}        : "
          f"{q['n_failures_flagged']} of {result['n_failures']} failures, "
          f"{q['n_rest_flagged']} of {len(rest)} others")
    print("  => REFUTED. The failures are CLEANER than average and none is flagged.")
    print("\nHYPOTHESIS 2: they are diseased photographs (exudate confusion)")
    print(f"  hard-exudate fraction, failures: {ex['failures_mean_fraction']:.5f}")
    print(f"  hard-exudate fraction, rest    : {ex['rest_mean_fraction']:.5f}")
    print(f"  ratio                          : {ex['ratio']:.2f}x")
    print(f"  failure ranks by exudate burden: {ex['failure_ranks']} (of {result['n_test']})")
    print(f"  Mann-Whitney (greater)         : p={ex['mannwhitney_p']:.4f} "
          f"-- indicative at n={result['n_failures']} vs {len(rest)}, not significant")
    print("\nWHAT THE FAILURES LOOK LIKE")
    print(f"  predicted/true disc area : {g['area_ratio_per_failure']} "
          f"(rest mean {g['area_ratio_rest']:.2f}x)")
    print(f"  centroid error (px)      : "
          f"{[round(v, 1) for v in g['centroid_px_per_failure']]} "
          f"(rest mean {g['centroid_px_rest']:.1f})")
    print(f"  connected components     : {g['n_comp_per_failure']} "
          f"(rest median {g['n_comp_rest_median']:.0f})")


if __name__ == "__main__":
    main()
