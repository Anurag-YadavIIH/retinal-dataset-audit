"""These two tests are the project's core claim, expressed as code."""

import pytest

from retinaprep.adapters.odir5k import build_manifest
from retinaprep.splits import (
    filter_split_to_manifest,
    image_random_split,
    load_persisted_split,
    patient_group_split,
    patient_overlap,
)


def test_patient_group_split_has_zero_patient_overlap(synthetic_fundus_dir, synthetic_cfg):
    """No patient_id may appear in more than one fold."""
    manifest = build_manifest(synthetic_cfg)
    split = patient_group_split(manifest, synthetic_cfg)
    overlap = patient_overlap(split, manifest)

    assert overlap["n_patients_in_more_than_one_fold"] == 0
    assert all(count == 0 for count in overlap["pairwise_overlap_counts"].values())


def test_image_random_split_does_leak_patients(synthetic_fundus_dir, synthetic_cfg):
    """Asserting the wrong method IS wrong, so the comparison is not vacuous.

    With two eyes per patient and a random image split, the probability of
    zero overlap is negligible. Assert overlap > 0.
    """
    manifest = build_manifest(synthetic_cfg)
    split = image_random_split(manifest, synthetic_cfg)
    overlap = patient_overlap(split, manifest)

    assert overlap["n_patients_in_more_than_one_fold"] > 0


def test_split_fractions_are_respected(synthetic_fundus_dir, synthetic_cfg):
    manifest = build_manifest(synthetic_cfg)
    test_frac = synthetic_cfg["split"]["test_frac"]
    val_frac = synthetic_cfg["split"]["val_frac"]
    n = len(manifest)

    random_split = image_random_split(manifest, synthetic_cfg)
    assert len(random_split["train"]) + len(random_split["val"]) + len(random_split["test"]) == n
    assert abs(len(random_split["test"]) / n - test_frac) < 0.08
    assert abs(len(random_split["val"]) / n - val_frac) < 0.08

    group_split = patient_group_split(manifest, synthetic_cfg)
    assert len(group_split["train"]) + len(group_split["val"]) + len(group_split["test"]) == n
    # Group-constrained stratification is only approximate on a dataset this
    # small (15 patients, one of them single-eye), so allow a wider tolerance
    # than the unconstrained split.
    assert abs(len(group_split["test"]) / n - test_frac) < 0.15
    assert abs(len(group_split["val"]) / n - val_frac) < 0.15


def test_filter_split_to_manifest_preserves_fold_assignment():
    """The fix for the split-then-curate ordering defect: filtering a base
    split down to a curated manifest must never move a surviving image to
    a different fold, only remove images that are no longer present."""
    import pandas as pd

    base_split = {
        "train": ["a.jpg", "b.jpg", "c.jpg"],
        "val": ["d.jpg"],
        "test": ["e.jpg", "f.jpg"],
    }
    # curated manifest drops b.jpg (rejected) and e.jpg (rejected)
    manifest = pd.DataFrame({"image_path": ["a.jpg", "c.jpg", "d.jpg", "f.jpg"]})

    filtered = filter_split_to_manifest(base_split, manifest)

    assert filtered == {"train": ["a.jpg", "c.jpg"], "val": ["d.jpg"], "test": ["f.jpg"]}


def test_filter_split_to_manifest_on_real_split_keeps_original_fold(
    synthetic_fundus_dir, synthetic_cfg
):
    """Same property, exercised against a real StratifiedGroupKFold split
    rather than a hand-built dict, and against every surviving image."""
    manifest = build_manifest(synthetic_cfg)
    base_split = patient_group_split(manifest, synthetic_cfg)
    fold_of = {p: fold for fold, paths in base_split.items() for p in paths}

    # Simulate curation dropping a few images (analogous to quality rejects).
    dropped = set(manifest["image_path"].iloc[:3])
    curated_manifest = manifest[~manifest["image_path"].isin(dropped)]

    filtered = filter_split_to_manifest(base_split, curated_manifest)

    for fold, paths in filtered.items():
        for p in paths:
            assert p not in dropped
            assert fold_of[p] == fold, f"{p} moved from {fold_of[p]} to {fold} after filtering"

    # Every surviving image is accounted for exactly once, nothing added.
    total_filtered = sum(len(paths) for paths in filtered.values())
    assert total_filtered == len(curated_manifest)


def test_load_persisted_split_fails_loudly_when_missing(tmp_path):
    """No silent fallback to a fresh recompute -- CLAUDE.md rule 7."""
    cfg = {"paths": {"artifacts": str(tmp_path)}}
    with pytest.raises(FileNotFoundError, match="retinaprep split"):
        load_persisted_split(cfg, "patient_group")
