"""The manifest contract is the interface every other module depends on."""

import pytest

from retinaprep.adapters.odir5k import build_manifest
from retinaprep.utils import MANIFEST_COLUMNS, validate_manifest


def test_manifest_has_required_columns(synthetic_fundus_dir, synthetic_cfg):
    _, metadata = synthetic_fundus_dir
    manifest = build_manifest(synthetic_cfg)

    assert list(manifest.columns) == MANIFEST_COLUMNS
    validate_manifest(manifest)  # must not raise

    n_patients = metadata["ID"].nunique()
    assert manifest["patient_id"].nunique() == n_patients
    assert len(manifest) == 2 * n_patients  # one row per eye, no missing files in the fixture
    assert set(manifest["label"].unique()) <= {0, 1}


def test_manifest_rejects_missing_image_paths(synthetic_fundus_dir, synthetic_cfg):
    manifest = build_manifest(synthetic_cfg)

    broken = manifest.copy()
    first = broken.index[0]
    broken.loc[first, "image_path"] = str(broken.loc[first, "image_path"]) + "_does_not_exist.jpg"

    with pytest.raises(ValueError, match="nonexistent file"):
        validate_manifest(broken)


def test_eye_column_only_L_or_R(synthetic_fundus_dir, synthetic_cfg):
    manifest = build_manifest(synthetic_cfg)

    assert set(manifest["eye"].unique()) == {"L", "R"}

    broken = manifest.copy()
    broken.loc[broken.index[0], "eye"] = "left"

    with pytest.raises(ValueError, match="'eye' has values outside"):
        validate_manifest(broken)
