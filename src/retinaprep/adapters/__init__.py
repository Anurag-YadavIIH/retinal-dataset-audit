"""Dataset adapters. Adding a dataset must mean writing one function here."""

from retinaprep.adapters.base import ADAPTERS, get_adapter

__all__ = ["ADAPTERS", "get_adapter"]
