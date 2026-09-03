"""Config loading. Everything tunable lives in configs/default.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load the YAML config and apply dotted-key overrides from the CLI.

    Overrides use dotted keys so a caller can do --set train.epochs=2 without
    editing the file. Keeping this in one place means every run can log the
    fully resolved config alongside its metrics, which is what makes results
    reproducible months later.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(cfg, dotted, value)
    return cfg


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def resolve_path(cfg: dict, *parts: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    return REPO_ROOT.joinpath(*parts)
