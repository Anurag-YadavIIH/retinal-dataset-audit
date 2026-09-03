"""Step 5 — per-image gradability scoring. No ML, all classical CV.

This is where most of the real cleaning effort in fundus datasets goes.
An ungradable image is not a hard example, it is noise, and training on it
teaches the model nothing except the photographer's mistakes.
"""

from __future__ import annotations


def variance_of_laplacian(gray) -> float:
    """Blur proxy. Low variance of the Laplacian means few sharp edges.

    TODO(claude-code): implement with cv2.Laplacian(gray, cv2.CV_64F).var().
    Note in WALKTHROUGH.md that this metric is resolution-dependent, so it must
    be computed after resizing to a fixed size or it will rank a 5184px Canon
    image above a 1728px Zeiss image for reasons that have nothing to do with
    focus. ODIR-5K mixes camera models, so this matters here.
    """
    raise NotImplementedError


def tenengrad(gray) -> float:
    """Second blur proxy via Sobel gradient energy. TODO(claude-code)."""
    raise NotImplementedError


def exposure_metrics(gray) -> dict:
    """Mean intensity, saturated pixel fraction, near-black fraction.

    TODO(claude-code): compute inside the detected FOV only. Counting the black
    corners outside the circular retinal field will make every image look
    underexposed.
    """
    raise NotImplementedError


def illumination_uniformity(gray, fov_mask) -> float:
    """Coefficient of variation of mean intensity across FOV quadrants.

    TODO(claude-code): catches the classic uneven-flash fundus image where one
    side is washed out and the other is in shadow.
    """
    raise NotImplementedError


def detect_fov(image):
    """Find the circular retinal field: threshold, largest contour, min enclosing circle.

    TODO(claude-code): return (mask, centre, radius). Flag images where the
    circle is clipped by the frame edge or where radius / frame_width falls
    below cfg.quality.thresholds.fov_radius_frac_min.
    """
    raise NotImplementedError


def gradability_score(metrics: dict, cfg: dict) -> float:
    """Transparent weighted combination in [0, 1]. Not learned, on purpose.

    TODO(claude-code): normalise each metric to [0, 1] against the configured
    thresholds, then apply cfg.quality.weights. Keep it explainable — being able
    to say why a specific image was rejected is worth more than a better score.
    """
    raise NotImplementedError


def run_quality(cfg: dict) -> None:
    """TODO(claude-code): score every image in the manifest, write
    artifacts/quality.parquet and artifacts/rejects.csv with a reason column."""
    raise NotImplementedError
