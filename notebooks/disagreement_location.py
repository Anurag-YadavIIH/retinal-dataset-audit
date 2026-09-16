"""Item 4 follow-up: WHERE the model disagrees, not just how much.

Aggregate Dice can match while the errors sit somewhere completely
different. §15 found the model's agreement with observer 1 (0.7884)
statistically indistinguishable from observer 2's (0.7882), which is
consistent with three quite different situations that a scalar cannot
separate:

1. The model learned the easy consensus and left the contested pixels
   contested -- its errors concentrate exactly where the two humans
   already disagree. Benign, and the most interesting outcome.
2. Its errors sit elsewhere -- it is making its own distinct mistakes
   that happen to cost the same amount of Dice.
3. Its errors sit in contested territory AND systematically take
   observer 1's side. That is style-fitting, and the aggregate score
   missed it.

Two quantities separate them, on the 20 DRIVE test images.

**Concentration.** D_human = XOR(obs1, obs2), the pixels the humans
dispute. D_model = XOR(model, obs2), the pixels where the model departs
from observer 2. Report |D_model ∩ D_human| / |D_model|: the share of
the model's disagreements that land in already-contested territory.

The null: disagreements can only occur where at least one annotator
marked something, so the containing region is U = obs1 ∪ obs2 ∪ model.
If D_model were scattered at random within U, the expected share landing
in D_human is |D_human| / |U|. Observed divided by that is an enrichment
factor -- 1.0 means "no better than chance", higher means the model's
errors are concentrated where humans already argue.

**Side-taking.** Within the contested pixels, which human does the model
agree with? Note this follows from binarity rather than needing a
separate measurement: if obs1 ≠ obs2 (contested) and model ≠ obs2, then
model = obs1 necessarily. So |D_model ∩ D_human| / |D_human| *is* the
fraction of contested pixels where the model sides with observer 1. Near
1.0 means it systematically backs its training annotator (style-fitting);
near 0.0 means it backs observer 2; near 0.5 means it splits them.

Run: python notebooks/disagreement_location.py
Output: printed report + artifacts/drive_vessel/disagreement_location.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.mask_quality import load_binary_mask  # noqa: E402
from retinaprep.segment import IMAGE_SIZE, UNet, predict_masks  # noqa: E402
from retinaprep.utils import get_logger  # noqa: E402

logger = get_logger(__name__)
ART = REPO_ROOT / "artifacts" / "drive_vessel"
DRIVE_DIR = Path("D:/retinaprep_data/drive")


def _target(path: Path) -> np.ndarray:
    mask = load_binary_mask(str(path)).astype(np.uint8) * 255
    resized = Image.fromarray(mask).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    return np.asarray(resized) > 127


def main() -> None:
    manifest = pd.read_parquet(ART / "manifest.parquet")
    with open(ART / "splits" / "seg_split.json") as fh:
        split = json.load(fh)
    by_path = {r["image_path"]: r for r in manifest.to_dict("records")}
    rows = [by_path[p] for p in split["test"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(
        torch.load(ART / "unet_vessel_best.pt", map_location=device)["state_dict"]
    )
    preds = predict_masks(model, rows, device)

    recs = []
    for pred, row in zip(preds, rows, strict=True):
        sid = row["patient_id"]
        o1 = _target(DRIVE_DIR / f"{sid}_manual1.gif")
        o2 = _target(DRIVE_DIR / f"{sid}_manual2.gif")

        d_human = np.logical_xor(o1, o2)
        d_model = np.logical_xor(pred, o2)
        union = o1 | o2 | pred
        both = d_human & d_model

        concentration = both.sum() / d_model.sum() if d_model.sum() else np.nan
        null = d_human.sum() / union.sum() if union.sum() else np.nan
        side_with_obs1 = both.sum() / d_human.sum() if d_human.sum() else np.nan

        recs.append(
            {
                "image_id": sid,
                "n_d_human": int(d_human.sum()),
                "n_d_model": int(d_model.sum()),
                "n_union": int(union.sum()),
                "concentration": float(concentration),
                "null_expectation": float(null),
                "enrichment": float(concentration / null) if null else np.nan,
                "frac_contested_taking_obs1_side": float(side_with_obs1),
            }
        )

    df = pd.DataFrame(recs)
    df.to_parquet(ART / "disagreement_location_per_image.parquet", index=False)

    def _ci(s):
        se = s.std(ddof=1) / np.sqrt(len(s))
        return [float(s.mean() - 1.96 * se), float(s.mean() + 1.96 * se)]

    summary = {
        "n_images": len(df),
        "concentration_mean": float(df["concentration"].mean()),
        "concentration_ci95": _ci(df["concentration"]),
        "null_expectation_mean": float(df["null_expectation"].mean()),
        "enrichment_mean": float(df["enrichment"].mean()),
        "enrichment_ci95": _ci(df["enrichment"]),
        "side_with_obs1_mean": float(df["frac_contested_taking_obs1_side"].mean()),
        "side_with_obs1_ci95": _ci(df["frac_contested_taking_obs1_side"]),
        "mean_d_human_px": float(df["n_d_human"].mean()),
        "mean_d_model_px": float(df["n_d_model"].mean()),
    }
    with open(ART / "disagreement_location.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Where does the model disagree? (DRIVE, 20 test images) ===\n")
    print("CONCENTRATION -- are the model's errors in already-contested territory?")
    print(f"  observed  : {summary['concentration_mean']:.4f} of model-vs-obs2 disagreements "
          f"fall inside human-contested pixels")
    print(f"              95% CI [{summary['concentration_ci95'][0]:.4f}, "
          f"{summary['concentration_ci95'][1]:.4f}]")
    print(f"  null      : {summary['null_expectation_mean']:.4f} "
          f"(|D_human| / |obs1 u obs2 u model|, i.e. spatially independent)")
    print(f"  enrichment: {summary['enrichment_mean']:.2f}x  "
          f"95% CI [{summary['enrichment_ci95'][0]:.2f}, {summary['enrichment_ci95'][1]:.2f}]")
    print()
    print("SIDE-TAKING -- within contested pixels, whose side does the model take?")
    print(f"  fraction siding with observer 1 (its training target): "
          f"{summary['side_with_obs1_mean']:.4f}")
    print(f"              95% CI [{summary['side_with_obs1_ci95'][0]:.4f}, "
          f"{summary['side_with_obs1_ci95'][1]:.4f}]")
    print("  (1.0 = always backs observer 1 => style-fitting; 0.5 = splits them; "
          "0.0 = always backs observer 2)")
    print()
    print(f"  mean contested pixels per image : {summary['mean_d_human_px']:.0f}")
    print(f"  mean model-vs-obs2 disagreements: {summary['mean_d_model_px']:.0f}")


if __name__ == "__main__":
    main()
