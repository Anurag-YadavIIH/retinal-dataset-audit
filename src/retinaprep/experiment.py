"""Step 4 + 7 — run the arms and build the comparison table.

| arm | data                     | split         |
|-----|--------------------------|---------------|
| A   | raw                      | image_random  |
| B   | raw                      | patient_group |
| C   | quality-curated          | patient_group |
| D   | curated + deduplicated   | patient_group |

Same seed, same hyperparameters, matched training-set sizes. If the sizes are
not matched, the table measures dataset size rather than curation.

Curation removes images (quality: ~0.7% of the dataset; quality+dedupe: an
additional ~0.2%), so C and D's natural pools are smaller than A/B's raw
one. Every arm is capped to the smallest natural train size across all
FOUR configured arms, not just the ones sharing a curation level -- matching
sizes within "raw" alone while leaving curated arms uncapped would let a
curated arm's smaller pool masquerade as a curation effect when it's really
just less data.
"""

from __future__ import annotations

import json

import pandas as pd

from retinaprep.config import resolve_path
from retinaprep.utils import get_logger, load_current_run_metrics

logger = get_logger(__name__)

ARMS = {
    "A": {"curation": "raw", "split": "image_random"},
    "B": {"curation": "raw", "split": "patient_group"},
    "C": {"curation": "quality", "split": "patient_group"},
    "D": {"curation": "quality+dedupe", "split": "patient_group"},
    # roadmap item 2 follow-up: holds out entire putative sites (camera/
    # centre proxy, notebooks/domain_shift_audit.py) rather than
    # patients. Requires artifacts/splits/site_group.json (retinaprep
    # split, after running the domain-shift audit once to derive
    # site_labels.parquet) -- see splits.site_group_split.
    "E": {"curation": "raw", "split": "site_group"},
}


def build_curated_manifest(cfg: dict, curation: str) -> pd.DataFrame:
    """The manifest for one curation level: raw, quality, or quality+dedupe.

    "quality" drops every image below cfg.quality.reject_below_score.
    "quality+dedupe" additionally drops all but one image per verified
    duplicate cluster (artifacts/duplicates.parquet), keeping the
    higher quality-scored image of each pair -- a duplicate already
    quality-rejected on one side needs no further action, since it's down
    to a single surviving copy already.
    """
    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    manifest = pd.read_parquet(artifacts_dir / "manifest.parquet")

    if curation == "raw":
        return manifest

    quality = pd.read_parquet(artifacts_dir / "quality.parquet")
    reject_below = cfg["quality"]["reject_below_score"]
    rejected = set(quality.loc[quality["score"] < reject_below, "image_path"])
    curated = manifest[~manifest["image_path"].isin(rejected)].reset_index(drop=True)

    if curation == "quality":
        return curated

    if curation == "quality+dedupe":
        dup = pd.read_parquet(artifacts_dir / "duplicates.parquet")
        quality_by_path = quality.set_index("image_path")["score"]
        curated_paths = set(curated["image_path"])
        drop: set[str] = set()
        for _cluster_id, group in dup[dup["is_duplicate"]].groupby("cluster_id"):
            survivors = [p for p in group["image_path"] if p in curated_paths]
            if len(survivors) < 2:
                continue  # already resolved by quality curation
            keep = max(survivors, key=lambda p: quality_by_path.get(p, 0.0))
            drop.update(p for p in survivors if p != keep)
        return curated[~curated["image_path"].isin(drop)].reset_index(drop=True)

    raise ValueError(f"Unknown curation {curation!r}")


def build_arm_split(cfg: dict, arm: str) -> tuple[pd.DataFrame, dict]:
    """The (manifest, split) pair for one arm.

    `experiment.split_mode` controls how curated arms (C, D) get their
    `patient_group` split:

    - "persisted_base" (default -- the fix): load the base split computed
      once on the raw pool by `retinaprep split`, then filter it down to
      this arm's curated manifest (`splits.filter_split_to_manifest`), so
      every surviving image keeps the fold it was originally assigned.
      The only variable between B/C/D is which images curation removed,
      not how the remaining ones got reshuffled.
    - "recompute_per_arm" (legacy): `patient_group_split` recomputed
      fresh on each arm's own curated pool. This is the split-then-curate
      ordering defect documented in docs/notes.md ("Why curation costs
      AUROC") -- kept, named, and fully working, because the project's
      existing writeup reports numbers produced this way and those
      numbers must stay reproducible on demand.

    `image_random` (arm A only) and `site_group` (arm E only) are
    unaffected by either mode -- neither is ever curated, so there is
    nothing to filter; E always loads the persisted split directly
    (same persisted-split discipline as the fix, applied uniformly).
    """
    from retinaprep.splits import (
        filter_split_to_manifest,
        image_random_split,
        load_persisted_split,
        patient_group_split,
    )

    spec = ARMS[arm]
    manifest = build_curated_manifest(cfg, spec["curation"])

    if spec["split"] == "image_random":
        return manifest, image_random_split(manifest, cfg)
    if spec["split"] == "site_group":
        return manifest, load_persisted_split(cfg, "site_group")
    if spec["split"] != "patient_group":
        raise ValueError(f"Unknown split {spec['split']!r} for arm {arm}")

    split_mode = cfg["experiment"].get("split_mode", "persisted_base")
    if split_mode == "persisted_base":
        base_split = load_persisted_split(cfg, "patient_group")
        split = filter_split_to_manifest(base_split, manifest)
    elif split_mode == "recompute_per_arm":
        split = patient_group_split(manifest, cfg)
    else:
        raise ValueError(f"Unknown experiment.split_mode {split_mode!r}")
    return manifest, split


def _compute_train_size_cap(cfg: dict) -> int | None:
    """Smallest natural train-set size across all FOUR configured arms --
    only meaningful under split_mode "recompute_per_arm".

    Under "persisted_base" (the fix), arms are DELIBERATELY left at their
    natural, different sizes: matching size is exactly what forced a
    fresh per-arm split recompute in the old design (StratifiedGroupKFold
    reshuffles ~31% of the training set from even a small pool change,
    docs/notes.md), which is the defect persisted_base exists to
    eliminate. The size difference between arms *is* the treatment under
    this mode (fewer images = what curation actually removed), not a
    confound to correct for -- capping it back to a common size would
    silently reintroduce the very resampling this mode exists to avoid.
    Always returns None here, regardless of experiment.match_arm_sizes's
    configured value.

    Recomputes each arm's manifest and split fresh (under
    "recompute_per_arm") rather than trusting whatever is on disk from a
    previous run, so the cap is always correct for the current
    quality/dedupe state, not stale.
    """
    split_mode = cfg["experiment"].get("split_mode", "persisted_base")
    if split_mode == "persisted_base":
        if cfg["experiment"].get("match_arm_sizes", True):
            logger.info(
                "split_mode=persisted_base: ignoring experiment.match_arm_sizes "
                "-- arm sizes are intentionally left to differ under this mode"
            )
        return None

    if not cfg["experiment"].get("match_arm_sizes", True):
        return None

    arms = [a for a in cfg["experiment"]["arms"] if a in ARMS]
    sizes = {}
    for a in arms:
        _manifest, split = build_arm_split(cfg, a)
        sizes[a] = len(split["train"])

    if not sizes:
        return None
    cap = min(sizes.values())
    logger.info("Natural train sizes %s -> cap %d", sizes, cap)
    return cap


def run_experiment(
    cfg: dict,
    arm: str | None = None,
    *,
    run_name: str | None = None,
    seed_override: int | None = None,
) -> list[dict]:
    """Run one arm across cfg.experiment.n_seeds seeds, write the results table.

    Seeds are `[cfg["seed"], cfg["seed"] + 1, ..., cfg["seed"] + n_seeds - 1]`
    unless `seed_override` pins a single specific seed (e.g. for an ad hoc
    rerun) -- in which case exactly one run happens, at that seed, and
    `run_name` is used as given rather than seed-suffixed.

    Returns a list of the run_train() result dicts, one per seed, in seed
    order -- for aggregating mean/std across seeds without re-reading every
    metrics.json back off disk.
    """
    if arm is None:
        raise SystemExit(f"Specify --arm (one of {sorted(ARMS)}).")
    if arm not in ARMS:
        raise SystemExit(f"Unknown arm {arm!r}. Choose from {sorted(ARMS)}.")

    spec = ARMS[arm]
    from retinaprep.train import run_train

    if seed_override is not None:
        seeds = [seed_override]
        run_names = [run_name or f"{arm}_seed{seed_override}"]
    else:
        n_seeds = cfg["experiment"].get("n_seeds", 1)
        seeds = [cfg["seed"] + i for i in range(n_seeds)]
        base_name = run_name or arm
        run_names = [f"{base_name}_seed{s}" for s in seeds]

    manifest, split = build_arm_split(cfg, arm)
    train_size_cap = _compute_train_size_cap(cfg)
    results = [
        run_train(
            cfg,
            split_name=spec["split"],
            run_name=rn,
            arm=arm,
            seed_override=seed,
            train_size_cap=train_size_cap,
            manifest=manifest,
            split=split,
        )
        for seed, rn in zip(seeds, run_names, strict=True)
    ]

    write_results_table(cfg)
    return results


def write_results_table(cfg: dict) -> None:
    """Write the comparison table for each arm's current run cohort.

    Idempotent and order-independent: always reflects whatever has actually
    been run, not just the run that just finished. "Current" means, per
    arm, only the runs sharing that arm's most recently recorded
    config_hash (utils.load_current_run_metrics) -- artifacts/runs/ never
    overwrites, so older cohorts (a previous config, an earlier point in
    the project's history) stay on disk and in index.json for provenance,
    but must not silently blend into this table's per-arm averages.
    """
    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    current = load_current_run_metrics(artifacts_dir)

    rows = []
    for run_dir_name in current["run_dir"] if not current.empty else []:
        metrics_path = artifacts_dir / "runs" / run_dir_name / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as fh:
                rows.append(json.load(fh))

    lines = [
        "| Run | Arm | Split | Seed | N train | AUROC | AUPRC | Sens@95%Spec "
        "| Confusion [[tn,fp],[fn,tp]] | Wall-clock (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tm = r["test_metrics"]
        lines.append(
            f"| {r['run_name']} | {r.get('arm') or '-'} | {r['split_name']} | {r['seed']} | "
            f"{r['n_train']} | {tm['auroc']:.3f} | {tm['auprc']:.3f} | "
            f"{tm['sensitivity_at_95_specificity']:.3f} | {tm['confusion_matrix']} | "
            f"{r['wall_clock_seconds']:.1f} |"
        )

    out_path = artifacts_dir / "results_table.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d runs)", out_path, len(rows))
