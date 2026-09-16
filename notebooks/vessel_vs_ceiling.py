"""Item 4 step 4b: score a vessel model against BOTH observers.

This is the experiment §15 proposes and the one comparison in this item
that is actually commensurable. The optic disc model cannot be compared
to the vessel ceiling -- different task, different difficulty, disc Dice
normally runs far higher. A vessel model trained on DRIVE and evaluated
on DRIVE's test split can be, because the ceiling was measured on those
exact 20 images.

Three numbers, on the same images:

    model vs observer 1   -- what a paper would report
    model vs observer 2   -- the same model, graded by the other human
    observer 1 vs 2       -- the ceiling (0.7879 from step 3)

The logic: observer 1's masks are the training target, so a model that
has learned *observer 1's conventions* should score distinctly better
against observer 1 than against observer 2, and its drop when regraded
should be comparable to the human-human disagreement. A model that has
learned *vessels* should score similarly against both. The gap between
the first two numbers is the style-fitting signal.

Run: python notebooks/vessel_vs_ceiling.py
Output: printed report + artifacts/drive_vessel/vessel_vs_ceiling.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.mask_quality import load_binary_mask  # noqa: E402
from retinaprep.segment import (  # noqa: E402
    IMAGE_SIZE,
    UNet,
    dice_score,
    iou_score,
    predict_masks,
)
from retinaprep.utils import get_logger  # noqa: E402

logger = get_logger(__name__)
ART = REPO_ROOT / "artifacts" / "drive_vessel"
DRIVE_DIR = Path("D:/retinaprep_data/drive")


def _target(mask_path: Path) -> np.ndarray:
    from PIL import Image

    mask = load_binary_mask(str(mask_path)).astype(np.uint8) * 255
    resized = Image.fromarray(mask).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    return np.asarray(resized) > 127


def main() -> None:
    manifest = pd.read_parquet(ART / "manifest.parquet")
    with open(ART / "splits" / "seg_split.json") as fh:
        split = json.load(fh)
    by_path = {r["image_path"]: r for r in manifest.to_dict("records")}
    test_rows = [by_path[p] for p in split["test"]]
    logger.info("Scoring %d DRIVE test images against both observers", len(test_rows))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    ckpt = torch.load(ART / "unet_vessel_best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    logger.info("Loaded model from epoch %d (val_dice=%.4f)", ckpt["epoch"], ckpt["val_dice"])

    preds = predict_masks(model, test_rows, device)

    rows = []
    for pred, row in zip(preds, test_rows, strict=True):
        sid = row["patient_id"]
        o1 = _target(DRIVE_DIR / f"{sid}_manual1.gif")
        o2 = _target(DRIVE_DIR / f"{sid}_manual2.gif")
        rows.append(
            {
                "image_id": sid,
                "model_vs_obs1_dice": dice_score(pred, o1),
                "model_vs_obs2_dice": dice_score(pred, o2),
                "obs1_vs_obs2_dice": dice_score(o1, o2),
                "model_vs_obs1_iou": iou_score(pred, o1),
                "model_vs_obs2_iou": iou_score(pred, o2),
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(ART / "vessel_vs_ceiling_per_image.parquet", index=False)

    def stats(col):
        s = df[col]
        se = s.std(ddof=1) / np.sqrt(len(s))
        return {
            "mean": float(s.mean()), "std": float(s.std(ddof=1)), "se": float(se),
            "ci95": [float(s.mean() - 1.96 * se), float(s.mean() + 1.96 * se)],
        }

    result = {
        "n_test": len(df),
        "model_vs_obs1": stats("model_vs_obs1_dice"),
        "model_vs_obs2": stats("model_vs_obs2_dice"),
        "ceiling_obs1_vs_obs2": stats("obs1_vs_obs2_dice"),
        "style_gap_obs1_minus_obs2": float(
            df["model_vs_obs1_dice"].mean() - df["model_vs_obs2_dice"].mean()
        ),
        "n_images_model_above_ceiling": int(
            (df["model_vs_obs1_dice"] > df["obs1_vs_obs2_dice"]).sum()
        ),
    }
    with open(ART / "vessel_vs_ceiling.json", "w") as fh:
        json.dump(result, fh, indent=2)

    print("\n=== DRIVE vessels: model vs both observers (commensurable comparison) ===")
    for key, label in [
        ("model_vs_obs1", "model vs observer 1 (the training target)"),
        ("model_vs_obs2", "model vs observer 2 (regraded)"),
        ("ceiling_obs1_vs_obs2", "observer 1 vs observer 2 (the ceiling)"),
    ]:
        s = result[key]
        print(f"  {label:42s} Dice {s['mean']:.4f} +/- {s['std']:.4f}  "
              f"95% CI [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]")
    print(f"\n  style gap (obs1 - obs2): {result['style_gap_obs1_minus_obs2']:+.4f}")
    print(f"  images where model beat the ceiling: "
          f"{result['n_images_model_above_ceiling']}/{result['n_test']}")


if __name__ == "__main__":
    main()
