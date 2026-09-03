"""These two tests are the project's core claim, expressed as code."""

from retinaprep.adapters.odir5k import build_manifest
from retinaprep.splits import image_random_split, patient_group_split, patient_overlap


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
    # small (14 patients), so allow a wider tolerance than the unconstrained split.
    assert abs(len(group_split["test"]) / n - test_frac) < 0.15
    assert abs(len(group_split["val"]) / n - val_frac) < 0.15
