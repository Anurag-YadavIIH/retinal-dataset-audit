"""Step 9 — mask and annotation QC.

The segmentation counterpart to quality.py. Same stance: classical
checks, no ML, every rejection logged with a reason rather than dropped
quietly, and thresholds that say *why* a mask failed rather than emitting
one opaque verdict.

What is checked, and why each one earns its place:

- **Dimension agreement.** A mask whose shape differs from its image
  cannot be overlaid at all. Hard reject, not a score.
- **Empty masks.** Zero foreground means no annotation. Distinguish
  carefully from an *absent* mask: for lesion classes most eyes simply do
  not have the lesion, and the adapter drops those rows rather than
  emitting empty ones (idrid_seg.py). An empty mask that survives to here
  is a real defect; a missing lesion is not, and conflating them would
  manufacture a reject rate out of healthy anatomy.
- **Connected components.** An optic disc is one structure, so a
  multi-component disc mask is either a stray blob or a mis-annotation.
  Lesion classes are legitimately multi-component (that is what
  microaneurysms *are*), so this check is only applied where a single
  component is expected.
- **Area outliers.** Optic disc area varies within a narrow anatomical
  band; a mask far outside the distribution is suspect even if it is
  binary, non-empty and single-component. Flagged on log-area, since
  areas are right-skewed.
- **Alignment beyond dimensions.** Matching shapes do not prove a mask
  belongs to its image. Fundus photographs have a large black surround
  outside the circular field of view, and a correctly-placed mask sits on
  retina, not on that surround. The fraction of mask foreground landing
  on near-black pixels is a cheap, honest alignment proxy: high values
  mean the mask is displaced, mirrored, or from another image.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from retinaprep.config import resolve_path
from retinaprep.utils import get_logger, save_json

logger = get_logger(__name__)

# A pixel this dark is outside the retinal field of view, not tissue.
FOV_DARK_MAX = 25
# Above this share of mask foreground sitting on the black surround, the
# mask is not plausibly aligned to its image.
MISALIGNED_FG_ON_DARK_MAX = 0.20
# Area outliers: |z| on log10(foreground fraction) across the dataset.
AREA_OUTLIER_Z = 3.0
EMPTY_FG_MAX = 1e-6


def load_binary_mask(path: str) -> np.ndarray:
    """Foreground boolean array, robust to how the mask was encoded.

    Encodings are not consistent even within one dataset: DRIVE's first
    observer ships mode 'L' with values {0, 255} while its second ships
    mode 'P' with values {0, 1}, so a decoder that assumes 255 means
    foreground silently reads the second observer as ~0.03% vessel
    instead of ~8.7%. Thresholding at >0 is the only assumption that
    holds across both, and it is why this helper exists rather than
    inlining np.asarray at each call site.
    """
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def compute_mask_metrics(image_path: str, mask_path: str, expect_single_component: bool) -> dict:
    """Per-mask geometry and agreement metrics. No thresholds applied here."""
    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"))
    raw = np.asarray(Image.open(mask_path))
    mask = load_binary_mask(mask_path)

    img_h, img_w = image.shape[:2]
    m_h, m_w = mask.shape[:2]
    dims_match = (img_h, img_w) == (m_h, m_w)

    fg = int(mask.sum())
    fg_fraction = float(mask.mean())

    n_components = 0
    largest_component_fraction = 0.0
    if fg > 0:
        labelled, n_components = ndimage.label(mask)
        if n_components > 0:
            sizes = ndimage.sum(mask, labelled, range(1, n_components + 1))
            largest_component_fraction = float(sizes.max() / fg)

    # Alignment: how much of the annotation sits on the black surround.
    fg_on_dark = np.nan
    if dims_match and fg > 0:
        grey = image.mean(axis=2)
        fg_on_dark = float((grey[mask] < FOV_DARK_MAX).mean())

    return {
        "image_path": image_path,
        "mask_path": mask_path,
        "img_h": img_h,
        "img_w": img_w,
        "mask_h": m_h,
        "mask_w": m_w,
        "dims_match": dims_match,
        "n_mask_values": int(len(np.unique(raw))),
        "fg_pixels": fg,
        "fg_fraction": fg_fraction,
        "n_components": int(n_components),
        "largest_component_fraction": largest_component_fraction,
        "fg_on_dark_fraction": fg_on_dark,
        "expect_single_component": expect_single_component,
    }


def flag_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Apply thresholds to computed metrics, one boolean column per reason.

    Area outliers are computed on log10 foreground fraction *within this
    run*, so the reference distribution is the dataset's own anatomy
    rather than a constant carried over from another dataset -- the
    transfer failure that cost this project a retraction on the fundus
    side (docs/notes.md, item 3).
    """
    out = df.copy()
    out["reject_dim_mismatch"] = ~out["dims_match"]
    out["reject_empty"] = out["fg_fraction"] <= EMPTY_FG_MAX

    out["flag_non_binary"] = out["n_mask_values"] > 2
    out["flag_multi_component"] = out["expect_single_component"] & (out["n_components"] > 1)
    out["flag_misaligned"] = out["fg_on_dark_fraction"] > MISALIGNED_FG_ON_DARK_MAX

    usable = out["fg_fraction"] > EMPTY_FG_MAX
    out["area_z"] = np.nan
    if usable.sum() > 2:
        log_area = np.log10(out.loc[usable, "fg_fraction"])
        z = (log_area - log_area.mean()) / log_area.std(ddof=1)
        out.loc[usable, "area_z"] = z
    out["flag_area_outlier"] = out["area_z"].abs() > AREA_OUTLIER_Z

    reject_cols = [c for c in out.columns if c.startswith("reject_")]
    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    out["rejected"] = out[reject_cols].any(axis=1)
    out["flagged"] = out[flag_cols].any(axis=1)

    def _reasons(row) -> str:
        hits = [c.replace("reject_", "").replace("flag_", "")
                for c in reject_cols + flag_cols if row[c]]
        return ",".join(hits) if hits else ""

    out["reasons"] = out.apply(_reasons, axis=1)
    return out


def run_mask_qc(cfg: dict) -> pd.DataFrame:
    """Score every mask in a segmentation manifest; write
    artifacts/mask_quality.parquet and mask_rejects.csv."""
    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    manifest = pd.read_parquet(artifacts_dir / "manifest.parquet")
    if "mask_path" not in manifest.columns:
        raise ValueError("manifest has no mask_path column -- this is not a segmentation manifest")

    mask_class = cfg["dataset"].get("mask_class", "OD")
    expect_single = mask_class == "OD"
    logger.info(
        "Mask QC on %d masks (class=%s, single-component expected=%s)",
        len(manifest), mask_class, expect_single,
    )

    rows = [
        compute_mask_metrics(r.image_path, r.mask_path, expect_single)
        for r in manifest.itertuples(index=False)
    ]
    metrics = flag_masks(pd.DataFrame(rows))

    out_path = artifacts_dir / "mask_quality.parquet"
    metrics.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)

    n = len(metrics)
    n_rej = int(metrics["rejected"].sum())
    n_flag = int(metrics["flagged"].sum())
    logger.info(
        "Rejected %d/%d masks (%.1f%%); flagged %d/%d (%.1f%%)",
        n_rej, n, n_rej / n * 100, n_flag, n, n_flag / n * 100,
    )
    for col in [c for c in metrics.columns if c.startswith(("reject_", "flag_"))]:
        k = int(metrics[col].sum())
        if k:
            logger.info("  %-26s %d", col, k)

    problems = metrics[metrics["rejected"] | metrics["flagged"]]
    rejects_path = artifacts_dir / "mask_rejects.csv"
    problems[["image_path", "mask_path", "reasons", "fg_fraction",
              "n_components", "fg_on_dark_fraction", "area_z"]].to_csv(rejects_path, index=False)
    logger.info("Wrote %s (%d rows)", rejects_path, len(problems))

    summary = {
        "n_masks": n,
        "n_rejected": n_rej,
        "reject_rate": n_rej / n,
        "n_flagged": n_flag,
        "flag_rate": n_flag / n,
        "by_reason": {
            c: int(metrics[c].sum())
            for c in metrics.columns if c.startswith(("reject_", "flag_"))
        },
        "fg_fraction": {
            "mean": float(metrics["fg_fraction"].mean()),
            "median": float(metrics["fg_fraction"].median()),
            "min": float(metrics["fg_fraction"].min()),
            "max": float(metrics["fg_fraction"].max()),
        },
        "mask_class": mask_class,
    }
    save_json(summary, artifacts_dir / "mask_quality_summary.json")
    return metrics
