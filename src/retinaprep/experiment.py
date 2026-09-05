"""Step 4 + 7 — run the arms and build the comparison table.

| arm | data                     | split         |
|-----|--------------------------|---------------|
| A   | raw                      | image_random  |
| B   | raw                      | patient_group |
| C   | quality-curated          | patient_group |
| D   | curated + deduplicated   | patient_group |

Same seed, same hyperparameters, matched training-set sizes. If the sizes are
not matched, the table measures dataset size rather than curation.

Only A and B are runnable: C and D need quality/dedupe, which are later
build-order steps that don't exist yet. Requesting them raises rather than
silently skipping or running on the wrong data.
"""

from __future__ import annotations

import json

from retinaprep.config import resolve_path
from retinaprep.utils import get_logger

logger = get_logger(__name__)

ARMS = {
    "A": {"curation": "raw", "split": "image_random"},
    "B": {"curation": "raw", "split": "patient_group"},
    "C": {"curation": "quality", "split": "patient_group"},
    "D": {"curation": "quality+dedupe", "split": "patient_group"},
}


def _compute_train_size_cap(cfg: dict) -> int | None:
    """Smallest natural train-set size among the currently-runnable arms.

    Only considers arms with "raw" curation (A/B) -- C/D aren't implemented,
    so they can't be part of the size-matching pool yet. Reads straight from
    the split JSONs (unaffected by which specific arm/seed is being trained),
    so every run in a session agrees on the same cap.
    """
    if not cfg["experiment"].get("match_arm_sizes", True):
        return None

    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    runnable = [a for a in cfg["experiment"]["arms"] if ARMS.get(a, {}).get("curation") == "raw"]

    sizes = []
    for a in runnable:
        split_path = artifacts_dir / "splits" / f"{ARMS[a]['split']}.json"
        if not split_path.exists():
            continue
        with open(split_path) as fh:
            sizes.append(len(json.load(fh)["train"]))

    if not sizes:
        return None
    logger.info("Train-size cap across runnable arms %s: %d", runnable, min(sizes))
    return min(sizes)


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
    if spec["curation"] != "raw":
        raise NotImplementedError(
            f"Arm {arm} needs {spec['curation']!r} curation, which isn't built yet "
            "(quality/dedupe are later build-order steps). Only A and B are runnable."
        )

    from retinaprep.train import run_train

    if seed_override is not None:
        seeds = [seed_override]
        run_names = [run_name or f"{arm}_seed{seed_override}"]
    else:
        n_seeds = cfg["experiment"].get("n_seeds", 1)
        seeds = [cfg["seed"] + i for i in range(n_seeds)]
        base_name = run_name or arm
        run_names = [f"{base_name}_seed{s}" for s in seeds]

    train_size_cap = _compute_train_size_cap(cfg)
    results = [
        run_train(
            cfg,
            split_name=spec["split"],
            run_name=rn,
            arm=arm,
            seed_override=seed,
            train_size_cap=train_size_cap,
        )
        for seed, rn in zip(seeds, run_names, strict=True)
    ]

    write_results_table(cfg)
    return results


def write_results_table(cfg: dict) -> None:
    """Scan every artifacts/runs/*/metrics.json and (re)write the comparison table.

    Idempotent and order-independent: always reflects whatever has actually
    been run, not just the run that just finished.
    """
    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    runs_dir = artifacts_dir / "runs"

    rows = []
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir()):
            metrics_path = run_dir / "metrics.json"
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
