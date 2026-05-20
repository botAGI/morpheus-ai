"""Legacy Morpheus training helpers.

New learning work should use ``morpheus.core.learning`` and the ``morpheus learn``
CLI group, which only trains from reviewed source-backed datasets. These
helpers remain for compatibility and dry-run script generation.
"""
from .consolidate import consolidate_sessions
from .train import train, generate_training_script
from .eval import run_eval, load_adapter

__all__ = [
    "consolidate_sessions",
    "train",
    "generate_training_script",
    "run_eval",
    "load_adapter",
]
