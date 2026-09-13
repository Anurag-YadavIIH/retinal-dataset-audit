"""Step 2 — the heart of the experiment: two ways to split, one right, one wrong."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from retinaprep.config import resolve_path
from retinaprep.utils import get_logger, save_json, set_global_seed

logger = get_logger(__name__)


def image_random_split(manifest: pd.DataFrame, cfg: dict) -> dict:
    """Naive stratified split over images, ignoring patient identity.

    This is the WRONG way and it is implemented on purpose. It is what a large
    share of published fundus work does, and reproducing it is how we measure
    the cost.
    """
    split_cfg = cfg["split"]
    test_frac = split_cfg["test_frac"]
    val_frac = split_cfg["val_frac"]
    stratify_col = split_cfg["stratify_on"]
    seed = cfg["seed"]

    train_val_df, test_df = train_test_split(
        manifest, test_size=test_frac, stratify=manifest[stratify_col], random_state=seed
    )
    remaining_val_frac = val_frac / (1.0 - test_frac)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=remaining_val_frac,
        stratify=train_val_df[stratify_col],
        random_state=seed,
    )

    return {
        "train": train_df["image_path"].tolist(),
        "val": val_df["image_path"].tolist(),
        "test": test_df["image_path"].tolist(),
    }


def patient_group_split(manifest: pd.DataFrame, cfg: dict) -> dict:
    """Grouped split on patient_id, stratified on label.

    Uses StratifiedGroupKFold so both eyes of one patient always land in the
    same fold. Perfect stratification is impossible under a group constraint,
    so the achieved class balance per fold is logged rather than assumed to
    match the configured fractions exactly.
    """
    split_cfg = cfg["split"]
    test_frac = split_cfg["test_frac"]
    val_frac = split_cfg["val_frac"]
    stratify_col = split_cfg["stratify_on"]
    group_col = split_cfg["group_on"]
    seed = cfg["seed"]

    manifest = manifest.reset_index(drop=True)
    train_val_df, test_df = _group_holdout(manifest, test_frac, stratify_col, group_col, seed)

    remaining_val_frac = val_frac / (1.0 - test_frac)
    train_df, val_df = _group_holdout(
        train_val_df, remaining_val_frac, stratify_col, group_col, seed
    )

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        balance = df[stratify_col].value_counts(normalize=True).sort_index().round(3).to_dict()
        logger.info(
            "patient_group %s fold: %d rows, %d patients, class balance (label -> fraction) %s",
            name,
            len(df),
            df[group_col].nunique(),
            balance,
        )

    return {
        "train": train_df["image_path"].tolist(),
        "val": val_df["image_path"].tolist(),
        "test": test_df["image_path"].tolist(),
    }


def _group_holdout(
    df: pd.DataFrame, holdout_frac: float, stratify_col: str, group_col: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Peel off one StratifiedGroupKFold fold (~holdout_frac of rows) as the holdout."""
    n_groups = df[group_col].nunique()
    n_splits = max(2, min(round(1.0 / holdout_frac), n_groups))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = df[stratify_col].to_numpy()
    groups = df[group_col].to_numpy()
    keep_pos, holdout_pos = next(sgkf.split(df, labels, groups))

    keep_df = df.iloc[keep_pos].reset_index(drop=True)
    holdout_df = df.iloc[holdout_pos].reset_index(drop=True)
    return keep_df, holdout_df


def load_persisted_split(cfg: dict, split_name: str) -> dict:
    """Load a previously-computed split from artifacts/splits/<split_name>.json.

    Written once by `retinaprep split`, on the raw/uncurated manifest --
    this is the base split that filter_split_to_manifest filters down for
    curated arms (see its docstring for why that matters). Fails loudly
    if `retinaprep split` hasn't been run yet, rather than silently
    falling back to a fresh recompute.
    """
    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    split_path = artifacts_dir / "splits" / f"{split_name}.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} does not exist -- run `retinaprep split` first to "
            "produce the base split that curated arms filter down from."
        )
    with open(split_path) as fh:
        return json.load(fh)


def filter_split_to_manifest(base_split: dict, manifest: pd.DataFrame) -> dict:
    """Filter a persisted split down to the images present in `manifest`.

    The fix for the split-then-curate ordering defect (docs/notes.md,
    "Why curation costs AUROC"): recomputing `patient_group_split` fresh
    on a curated (reduced) pool lets `StratifiedGroupKFold` reshuffle a
    large fraction of fold membership from even a small change to its
    input -- removing 43 images (0.7%) changed ~31% of the training set
    in the original measurement, confounding curation's effect with the
    split algorithm's sensitivity to perturbation, not curation itself.

    Filtering a fixed base split instead keeps every surviving image's
    fold assignment exactly as it was on the raw pool -- the only
    variable left between arms is which images were removed, which is
    what a curation comparison is actually supposed to isolate. This
    function only ever removes paths; it never reassigns a surviving
    image to a different fold.
    """
    valid_paths = set(manifest["image_path"])
    return {fold: [p for p in paths if p in valid_paths] for fold, paths in base_split.items()}


def run_split(cfg: dict) -> None:
    """Write artifacts/splits/{image_random,patient_group}.json and print an overlap report."""
    set_global_seed(cfg["seed"])

    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    manifest = pd.read_parquet(artifacts_dir / "manifest.parquet")

    splits_dir = artifacts_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "image_random": image_random_split(manifest, cfg),
        "patient_group": patient_group_split(manifest, cfg),
    }

    for name, split in splits.items():
        save_json(split, splits_dir / f"{name}.json")

    print("Patient-overlap report:")
    for name, split in splits.items():
        overlap = patient_overlap(split, manifest)
        print(f"  {name}: {overlap}")


def patient_overlap(split: dict, manifest: pd.DataFrame) -> dict:
    """Count patients appearing in more than one fold. Zero for patient_group."""
    path_to_patient = manifest.set_index("image_path")["patient_id"]
    fold_patients = {fold: set(path_to_patient.loc[paths]) for fold, paths in split.items()}

    fold_names = list(fold_patients)
    pairwise: dict[str, int] = {}
    overlapping_patients: set = set()
    for i in range(len(fold_names)):
        for j in range(i + 1, len(fold_names)):
            a, b = fold_names[i], fold_names[j]
            shared = fold_patients[a] & fold_patients[b]
            pairwise[f"{a}_vs_{b}"] = len(shared)
            overlapping_patients |= shared

    return {
        "pairwise_overlap_counts": pairwise,
        "n_patients_in_more_than_one_fold": len(overlapping_patients),
    }
