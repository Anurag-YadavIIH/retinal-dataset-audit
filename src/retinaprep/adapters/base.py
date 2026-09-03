"""Adapter contract.

Every adapter takes a config and returns a DataFrame with exactly the columns
in utils.MANIFEST_COLUMNS. Nothing downstream knows which dataset it came from.
"""

from __future__ import annotations

from collections.abc import Callable

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
