"""Dataset adapters. Adding a dataset must mean writing one function here."""

from __future__ import annotations

import importlib
import pkgutil

from retinaprep.adapters.base import ADAPTERS, get_adapter

# Each adapter module registers itself via the @register decorator as a side
# effect of being imported. Discovering and importing every submodule here
# means a new adapter file really is the whole change -- nothing in this
# file needs to be touched to pick it up.
for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name != "base":
        importlib.import_module(f"{__name__}.{_module_info.name}")

__all__ = ["ADAPTERS", "get_adapter"]
