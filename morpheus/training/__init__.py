"""
Morpheus Training — Daily LoRA fine-tuning pipeline.

Modules:
    consolidate: Convert OpenClaw sessions to training dataset
    train: QLoRA fine-tuning script
    eval: Evaluation harness
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
