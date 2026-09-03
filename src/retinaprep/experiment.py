"""Step 4 + 7 — run the arms and build the comparison table.

| arm | data                     | split         |
|-----|--------------------------|---------------|
| A   | raw                      | image_random  |
| B   | raw                      | patient_group |
| C   | quality-curated          | patient_group |
| D   | curated + deduplicated   | patient_group |

Same seed, same hyperparameters, matched training-set sizes. If the sizes are
not matched, the table measures dataset size rather than curation.
"""

from __future__ import annotations

ARMS = {
    "A": {"curation": "raw", "split": "image_random"},
    "B": {"curation": "raw", "split": "patient_group"},
    "C": {"curation": "quality", "split": "patient_group"},
    "D": {"curation": "quality+dedupe", "split": "patient_group"},
}


def run_experiment(cfg: dict, arm: str | None = None) -> None:
    """TODO(claude-code): run one arm or all of them, then write
    artifacts/results_table.md and artifacts/results.json.

    STOP after arms A and B and report those two numbers to the user before
    building anything else. That delta is the project's headline.
    """
    raise NotImplementedError
