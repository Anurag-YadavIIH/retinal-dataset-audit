"""Adapter contract.

Every adapter takes a config and returns a DataFrame with exactly the columns
in utils.MANIFEST_COLUMNS. Nothing downstream knows which dataset it came from.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from retinaprep.utils import get_logger

logger = get_logger(__name__)

ADAPTERS: dict[str, Callable] = {}


def register(name: str):
    def _wrap(fn):
        ADAPTERS[name] = fn
        return fn

    return _wrap


def get_adapter(name: str) -> Callable:
    if name not in ADAPTERS:
        raise KeyError(f"No adapter for {name!r}. Registered: {sorted(ADAPTERS)}")
    return ADAPTERS[name]


def subsample_by_patient(df: pd.DataFrame, subsample_n: int | None, seed: int) -> pd.DataFrame:
    """Cap a manifest-shaped DataFrame to roughly `subsample_n` rows by
    keeping whole patients.

    Subsampling by row would silently split a patient's eyes across the
    subsample boundary and undermine the whole leakage experiment, so
    patients are drawn whole (be it one eye or two) and accumulated until
    the row-count target is reached (the last patient added may push the
    total slightly over subsample_n).

    Promoted here from odir5k.py (originally private, `_subsample_by_patient`)
    when eyepacs.py needed the exact same logic verbatim -- the second adapter
    is the concrete test of whether one shared helper belongs here instead of
    two private copies drifting apart; it does.
    """
    if subsample_n is None or len(df) <= subsample_n:
        return df

    patient_sizes = df.groupby("patient_id").size()
    rng = np.random.default_rng(seed)
    shuffled_patients = rng.permutation(patient_sizes.index.to_numpy())

    kept_patients = []
    running_total = 0
    for pid in shuffled_patients:
        if running_total >= subsample_n:
            break
        kept_patients.append(pid)
        running_total += int(patient_sizes[pid])

    logger.info(
        "Subsampled to %d whole patients (%d rows), target was %d rows",
        len(kept_patients),
        running_total,
        subsample_n,
    )
    return df.loc[df["patient_id"].isin(kept_patients)].reset_index(drop=True)
