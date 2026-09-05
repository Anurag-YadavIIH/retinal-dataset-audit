"""The manifest contract is the interface every other module depends on."""

import pandas as pd
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
    assert len(manifest) == len(metadata)  # long format: one manifest row per metadata row
    assert set(manifest["label"].unique()) <= {0, 1}

    # Not every patient has two eyes -- the fixture includes a single-eye
    # patient (ID 15) specifically so this is never silently assumed.
    images_per_patient = manifest.groupby("patient_id").size()
    assert (images_per_patient == 1).sum() >= 1
    assert (images_per_patient == 2).sum() >= 1

    single_eye_rows = manifest[manifest["patient_id"] == "15"]
    assert len(single_eye_rows) == 1
    assert single_eye_rows.iloc[0]["eye"] == "R"


def test_unrecognized_filename_pattern_fails_loudly(synthetic_fundus_dir, synthetic_cfg):
    data_root, _ = synthetic_fundus_dir
    csv_path = data_root / "full_df.csv"
    raw = pd.read_csv(csv_path)
    raw.loc[raw.index[0], "filename"] = "0_center.jpg"  # neither _left nor _right
    raw.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="do not match the expected"):
        build_manifest(synthetic_cfg)


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
