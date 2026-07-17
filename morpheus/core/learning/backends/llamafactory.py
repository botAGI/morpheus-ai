"""LlamaFactory training command renderer."""

from morpheus.core.learning.backends.base import RenderedTrainingCommand, TrainingBackend
from morpheus.core.learning.training_runtime import shell_quote_training_argument


class LlamaFactoryBackend(TrainingBackend):
    name = "llamafactory"
    supported_methods = {"qlora", "lora"}

    def render_command(self, config: dict, *, dry_run: bool) -> RenderedTrainingCommand:
        quantization_args = ["--quantization_bit", "4"] if config["method"] == "qlora" else []
        args = [
            "llamafactory-cli",
            "train",
            "--stage",
            "sft",
            "--model_name_or_path",
            config["base_model"],
            "--template",
            "qwen",
            "--dataset_dir",
            config["dataset_dir"],
            "--dataset",
            config["dataset_name"],
            "--output_dir",
            config["output_dir"],
            "--finetuning_type",
            "lora",
            "--lora_rank",
            str(config["rank"]),
            "--lora_alpha",
            str(config["alpha"]),
            "--lora_dropout",
            str(config["dropout"]),
            "--num_train_epochs",
            str(config["epochs"]),
            "--learning_rate",
            str(config["learning_rate"]),
            "--cutoff_len",
            str(config["max_seq_length"]),
            *quantization_args,
        ]
        notes = ["dry-run only; no model download or GPU work"] if dry_run else []
        return RenderedTrainingCommand(command=_shell_script(args, dry_run=dry_run), backend_notes=notes)


def _shell_script(args: list[object], *, dry_run: bool) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    if dry_run:
        lines.append("# Dry-run scaffold. Remove --dry-run and run through Morpheus execution guards.")
    lines.append(" ".join(shell_quote_training_argument(arg) for arg in args))
    return "\n".join(lines) + "\n"
