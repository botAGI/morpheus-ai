"""PEFT training command renderer."""
import shlex

from morpheus.core.learning.backends.base import RenderedTrainingCommand, TrainingBackend


class PeftBackend(TrainingBackend):
    name = "peft"
    supported_methods = {"lora", "qlora"}

    def render_command(self, config: dict, *, dry_run: bool) -> RenderedTrainingCommand:
        args = [
            "python",
            "-m",
            "morpheus.learning_peft_train",
            "--dataset",
            config["dataset_path"],
            "--base-model",
            config["base_model"],
            "--method",
            config["method"],
            "--output-dir",
            config["output_dir"],
            "--rank",
            str(config["rank"]),
            "--alpha",
            str(config["alpha"]),
            "--dropout",
            str(config["dropout"]),
            "--epochs",
            str(config["epochs"]),
            "--learning-rate",
            str(config["learning_rate"]),
            "--max-seq-length",
            str(config["max_seq_length"]),
        ]
        if dry_run:
            args.append("--dry-run")
        return RenderedTrainingCommand(
            command=_shell_script(args),
            backend_notes=["dry-run PEFT scaffold; runner module is intentionally not executed"],
        )


def _shell_script(args: list[str]) -> str:
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        " ".join(shlex.quote(arg) for arg in args),
        "",
    ])
