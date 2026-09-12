"""artifacts/runs/ must never overwrite a previous run, and cohorts from
different configs must never silently blend into one average."""

import copy

from retinaprep.utils import append_run_index, config_hash, load_current_run_metrics, load_run_index


def test_config_hash_ignores_config_path(synthetic_cfg):
    """_config_path is an absolute filesystem path, not config content --
    two checkouts of the same config on different machines/paths must hash
    the same."""
    cfg_a = copy.deepcopy(synthetic_cfg)
    cfg_b = copy.deepcopy(synthetic_cfg)
    cfg_b["_config_path"] = "/some/other/machine/configs/default.yaml"

    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_config_hash_changes_with_content(synthetic_cfg):
    cfg_a = copy.deepcopy(synthetic_cfg)
    cfg_b = copy.deepcopy(synthetic_cfg)
    cfg_b["train"]["epochs"] = cfg_b["train"]["epochs"] + 1

    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_load_run_index_empty_when_missing(tmp_path):
    assert load_run_index(tmp_path) == []


def test_append_run_index_accumulates(tmp_path):
    append_run_index(tmp_path, {"run_name": "A_seed42", "arm": "A"})
    append_run_index(tmp_path, {"run_name": "A_seed43", "arm": "A"})

    entries = load_run_index(tmp_path)
    assert len(entries) == 2
    assert [e["run_name"] for e in entries] == ["A_seed42", "A_seed43"]


def _entry(*, arm, seed, config_hash, timestamp, auroc=0.8):
    return {
        "run_name": f"{arm}_seed{seed}",
        "arm": arm,
        "seed": seed,
        "split_name": "image_random",
        "config_hash": config_hash,
        "timestamp": timestamp,
        "run_dir": f"{arm}_seed{seed}_{timestamp}_{config_hash}",
        "n_train": 100,
        "auroc": auroc,
        "auprc": auroc,
        "sens_95_spec": auroc,
    }


def test_load_current_run_metrics_excludes_superseded_cohort(tmp_path):
    """Two cohorts for arm A, different config_hash. Only the newer
    (later-timestamp) cohort should count as current -- otherwise a rerun
    under a changed config would silently blend into the old average."""
    old_cohort = [
        _entry(arm="A", seed=s, config_hash="old111", timestamp="20260101T000000Z")
        for s in (42, 43)
    ]
    new_cohort = [
        _entry(arm="A", seed=s, config_hash="new222", timestamp="20260901T000000Z")
        for s in (42, 43)
    ]
    for e in old_cohort + new_cohort:
        append_run_index(tmp_path, e)

    current = load_current_run_metrics(tmp_path)

    assert set(current["config_hash"]) == {"new222"}
    assert len(current) == 2


def test_load_current_run_metrics_keeps_other_arms_independent(tmp_path):
    """Arm A getting a fresh cohort must not affect arm B's own current cohort."""
    entries = [
        _entry(arm="A", seed=42, config_hash="old", timestamp="20260101T000000Z"),
        _entry(arm="A", seed=42, config_hash="new", timestamp="20260901T000000Z"),
        _entry(arm="B", seed=42, config_hash="stable", timestamp="20260101T000000Z"),
    ]
    for e in entries:
        append_run_index(tmp_path, e)

    current = load_current_run_metrics(tmp_path)

    by_arm = current.set_index("arm")["config_hash"].to_dict()
    assert by_arm == {"A": "new", "B": "stable"}


def test_load_current_run_metrics_empty_when_no_index(tmp_path):
    assert load_current_run_metrics(tmp_path).empty
