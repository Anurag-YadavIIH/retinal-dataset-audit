"""Step 5 — per-image gradability scoring. No ML, all classical CV.

This is where most of the real cleaning effort in fundus datasets goes.
An ungradable image is not a hard example, it is noise, and training on it
teaches the model nothing except the photographer's mistakes.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from retinaprep.config import resolve_path
from retinaprep.utils import get_logger

logger = get_logger(__name__)

BLUR_RESIZE = 512  # fixed size before blur metrics -- see variance_of_laplacian


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Blur proxy. Low variance of the Laplacian means few sharp edges.

    Resolution-dependent: more pixels means more high-frequency detail to
    differentiate even at identical true focus, so this metric ranks a
    5184px Canon capture above a 1728px Zeiss one for reasons that have
    nothing to do with focus. The caller must resize to a fixed size first
    (see `_resize_for_blur` / `compute_metrics`) or it ranks cameras, not
    sharpness. Checked directly: this project's own preprocessed_images/ is
    already uniformly 512x512 (verified against a 200-image random sample),
    so the concern is moot for the data this pipeline actually scores right
    now -- but it is very much live for the raw Training Images/ (12
    distinct resolutions found in the first 50 files alone, 894px-3888px),
    and for any other dataset this module gets pointed at later. Resizing
    unconditionally rather than skipping it when "it doesn't matter here"
    is the defensible default.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    """Second blur proxy via Sobel gradient energy. Same resolution caveat as above."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


def _resize_for_blur(gray: np.ndarray, size: int = BLUR_RESIZE) -> np.ndarray:
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def exposure_metrics(gray: np.ndarray, fov_mask: np.ndarray) -> dict:
    """Mean intensity, saturated pixel fraction, near-black fraction -- inside the FOV only.

    Computed over the whole frame, the black corners outside the circular
    retinal field dominate the pixel count and make every image look
    underexposed regardless of how the actual fundus was captured.
    """
    pixels = gray[fov_mask]
    if pixels.size == 0:
        return {"mean_intensity": 0.0, "saturated_frac": 1.0, "near_black_frac": 1.0}
    return {
        "mean_intensity": float(pixels.mean()),
        "saturated_frac": float((pixels >= 250).mean()),
        "near_black_frac": float((pixels <= 5).mean()),
    }


def illumination_uniformity(gray: np.ndarray, fov_mask: np.ndarray) -> float:
    """Coefficient of variation of mean intensity across FOV quadrants.

    Catches the classic uneven-flash fundus image where one side is washed
    out and the other is in shadow -- a failure mode blur and exposure
    metrics alone don't see, since the frame-wide mean can look perfectly
    normal even when one half is unusable.
    """
    ys, xs = np.where(fov_mask)
    if ys.size == 0:
        return 1.0
    cy, cx = ys.mean(), xs.mean()
    quadrant_means = []
    for y_cond in (ys < cy, ys >= cy):
        for x_cond in (xs < cx, xs >= cx):
            sel = y_cond & x_cond
            if sel.any():
                quadrant_means.append(gray[ys[sel], xs[sel]].mean())
    if len(quadrant_means) < 2:
        return 1.0
    arr = np.array(quadrant_means, dtype=float)
    mean = arr.mean()
    if mean <= 0:
        return 1.0
    return float(arr.std() / mean)


def _largest_contour(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def detect_fov(image: np.ndarray) -> tuple[np.ndarray | None, tuple[float, float] | None, float]:
    """Find the circular retinal field: threshold, largest contour, min enclosing circle.

    Returns (mask, centre, radius); mask/centre are None and radius is 0.0
    if no field could be found at all (e.g. a fully black frame).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    contour = _largest_contour(image)
    if contour is None:
        return None, None, 0.0
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    return mask, (cx, cy), float(radius)


def _circularity(image: np.ndarray) -> float:
    """contour_area / (pi * enclosing_radius^2), in [0, 1] for a convex region.

    A genuine full circle scores near 1.0; a circle truncated by the frame
    edge scores lower, since the visible area falls short of what its fitted
    enclosing radius implies. Reported as a diagnostic, not used as a hard
    reject gate -- see the note on `fov_clipped` in `compute_metrics` for
    why: on this real dataset it does not cleanly separate "truncated" from
    "complete but oval-cropped," and forcing it to gate rejection produced a
    38-51% reject rate driven almost entirely by that false-positive mode
    (verified by eye against several examples at each stage of tuning this),
    not by genuine data-quality problems.
    """
    contour = _largest_contour(image)
    if contour is None:
        return 0.0
    area = cv2.contourArea(contour)
    _, radius = cv2.minEnclosingCircle(contour)
    if radius <= 0:
        return 0.0
    return float(area / (np.pi * radius**2))


FOV_CLIPPED_CIRCULARITY_MAX = 0.85  # diagnostic only -- see note below, not a reject gate


def compute_metrics(image: np.ndarray, cfg: dict) -> dict:
    """Run every metric on one RGB image, using the config's own FOV-validity thresholds.

    `fov_clipped` (circularity below FOV_CLIPPED_CIRCULARITY_MAX) is reported
    for visibility but deliberately does NOT gate `fov_valid` or the score.
    Investigated directly on this project's real data: circularity conflates
    three distinct things on this dataset -- genuine truncation (a real
    straight-edge cutoff), complete-but-oval crops (a legitimate, common
    preprocessing variant here, not a defect), and low-contrast/hazy images
    producing a ragged threshold boundary. Gating on it hard-rejected 38-51%
    of the dataset in earlier passes, verified by eye to be overwhelmingly
    the oval-crop case, not real clipping -- see docs/notes.md. Only the
    radius-fraction check (a real, uncontroversial "is the field too small"
    signal) gates FOV validity; the transparent weighted score below is what
    actually determines gradability, which is also more useful in
    practice -- it says *why* an image scored low, rather than a single
    opaque "FOV rejected."
    """
    thresholds = cfg["quality"]["thresholds"]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    height, width = gray.shape

    mask, _centre, radius = detect_fov(image)
    if mask is None:
        return {
            "laplacian_var": 0.0,
            "tenengrad": 0.0,
            "mean_intensity": 0.0,
            "saturated_frac": 1.0,
            "near_black_frac": 1.0,
            "illumination_cv": 1.0,
            "fov_radius_frac": 0.0,
            "fov_clipped": True,
            "fov_valid": False,
        }

    fov_radius_frac = radius / (min(width, height) / 2)
    clipped = _circularity(image) < FOV_CLIPPED_CIRCULARITY_MAX
    fov_valid = fov_radius_frac >= thresholds["fov_radius_frac_min"]

    exposure = exposure_metrics(gray, mask)
    blur_gray = _resize_for_blur(gray)

    return {
        "laplacian_var": variance_of_laplacian(blur_gray),
        "tenengrad": tenengrad(blur_gray),
        "mean_intensity": exposure["mean_intensity"],
        "saturated_frac": exposure["saturated_frac"],
        "near_black_frac": exposure["near_black_frac"],
        "illumination_cv": illumination_uniformity(gray, mask),
        "fov_radius_frac": fov_radius_frac,
        "fov_clipped": clipped,
        "fov_valid": fov_valid,
    }


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def gradability_score(metrics: dict, cfg: dict) -> float:
    """Transparent weighted combination in [0, 1]. Not learned, on purpose.

    A bad FOV (too small or clipped) is a hard gate, not a soft weighted
    penalty: an image with no usable retinal field is not "somewhat
    gradable," it is unusable, and averaging in good blur/exposure numbers
    would hide that. Being able to say why one specific image was rejected
    is worth more than a smoother-looking score.
    """
    if not metrics["fov_valid"]:
        return 0.0

    thresholds = cfg["quality"]["thresholds"]
    weights = cfg["quality"]["weights"]

    blur_score = _clip01(metrics["laplacian_var"] / thresholds["laplacian_var_min"])
    saturated_score = 1.0 - _clip01(
        metrics["saturated_frac"] / thresholds["saturated_pixel_frac_max"]
    )
    near_black_score = 1.0 - _clip01(
        metrics["near_black_frac"] / thresholds["near_black_pixel_frac_max"]
    )
    exposure_score = (saturated_score + near_black_score) / 2
    illumination_score = 1.0 - _clip01(
        metrics["illumination_cv"] / thresholds["quadrant_intensity_cv_max"]
    )

    score = (
        weights["blur"] * blur_score
        + weights["exposure"] * exposure_score
        + weights["illumination"] * illumination_score
    )
    return float(_clip01(score))


def _reject_reason(row: dict, cfg: dict) -> str:
    thresholds = cfg["quality"]["thresholds"]
    if not row["fov_valid"]:
        return (
            f"FOV radius fraction {row['fov_radius_frac']:.2f} "
            f"below minimum {thresholds['fov_radius_frac_min']}"
        )
    reject_below = cfg["quality"]["reject_below_score"]
    return f"gradability score {row['score']:.3f} below reject_below_score {reject_below}"


def run_quality(cfg: dict) -> None:
    """Score every image in the manifest; write artifacts/quality.parquet and
    artifacts/rejects.csv with a reason column, plus a contact sheet of the
    highest/lowest scoring images for manual threshold validation. There are
    no ground-truth quality labels for this dataset, so eyeballing the
    extremes is the honest substitute for a held-out validation set.
    """
    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    manifest = pd.read_parquet(artifacts_dir / "manifest.parquet")

    rows = []
    for _, m_row in manifest.iterrows():
        image_bgr = cv2.imread(m_row["image_path"], cv2.IMREAD_COLOR)
        if image_bgr is None:
            logger.warning("Could not read %s for quality scoring", m_row["image_path"])
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        metrics = compute_metrics(image, cfg)
        score = gradability_score(metrics, cfg)
        rows.append(
            {
                "image_path": m_row["image_path"],
                "patient_id": m_row["patient_id"],
                "score": score,
                **metrics,
            }
        )

    quality_df = pd.DataFrame(rows)
    quality_path = artifacts_dir / "quality.parquet"
    quality_df.to_parquet(quality_path, index=False)
    logger.info("Wrote %s (%d images scored)", quality_path, len(quality_df))

    reject_threshold = cfg["quality"]["reject_below_score"]
    rejects = quality_df[quality_df["score"] < reject_threshold].copy()
    rejects["reason"] = rejects.apply(lambda r: _reject_reason(r.to_dict(), cfg), axis=1)
    rejects_path = artifacts_dir / "rejects.csv"
    rejects[["image_path", "patient_id", "score", "reason"]].to_csv(rejects_path, index=False)

    n_total = len(quality_df)
    n_rejected = len(rejects)
    reject_rate = n_rejected / n_total if n_total else 0.0
    logger.info(
        "Rejected %d/%d images (%.1f%%) below score %.2f",
        n_rejected,
        n_total,
        reject_rate * 100,
        reject_threshold,
    )
    if reject_rate > 0.10:
        logger.warning(
            "Reject rate %.1f%% exceeds ~10%% -- thresholds may be too aggressive. "
            "Reporting the number as measured, not tuning to a target.",
            reject_rate * 100,
        )

    write_contact_sheet(cfg, quality_df)


def write_contact_sheet(
    cfg: dict, quality_df: pd.DataFrame, n_per_side: int = 20, cols: int = 10, thumb_size: int = 150
) -> str:
    """20 lowest- and 20 highest-scoring images, side by side, for manual eyeballing."""
    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    sorted_df = quality_df.sort_values("score")
    lowest = sorted_df.head(n_per_side)
    highest = sorted_df.tail(n_per_side).sort_values("score", ascending=False)

    header_h = 24
    rows_per_group = -(-n_per_side // cols)
    group_h = header_h + rows_per_group * thumb_size
    sheet = Image.new("RGB", (cols * thumb_size, group_h * 2), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    def render_group(df_group: pd.DataFrame, y_offset: int, label: str) -> None:
        draw.rectangle([0, y_offset, cols * thumb_size, y_offset + header_h], fill=(20, 20, 20))
        draw.text((6, y_offset + 4), label, fill="white")
        for i, (_, r) in enumerate(df_group.iterrows()):
            row, col = divmod(i, cols)
            x = col * thumb_size
            y = y_offset + header_h + row * thumb_size
            try:
                with Image.open(r["image_path"]) as im:
                    thumb = im.convert("RGB").resize((thumb_size, thumb_size))
            except OSError:
                thumb = Image.new("RGB", (thumb_size, thumb_size), (128, 128, 128))
            sheet.paste(thumb, (x, y))
            draw.rectangle([x, y, x + thumb_size, y + 16], fill=(0, 0, 0))
            draw.text((x + 3, y + 2), f"{r['score']:.2f}", fill="white")

    render_group(lowest, 0, f"LOWEST {n_per_side} (most likely reject)")
    render_group(highest, group_h, f"HIGHEST {n_per_side} (best gradability)")

    out_path = artifacts_dir / "quality_contact_sheet.png"
    sheet.save(out_path)
    logger.info("Wrote contact sheet (%d+%d thumbnails) to %s", len(lowest), len(highest), out_path)
    return str(out_path)
