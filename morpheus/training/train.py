"""
QLoRA Fine-tuning Script for Morpheus.

Fine-tunes a base model (Qwen2.5-7B or similar) with LoRA adapter
on consolidated session data for "weights-as-memory" effect.
"""
import json
import subprocess
from pathlib import Path
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CONFIG = {
    "base_model": "qwen2.5:7b",
    "output_dir": "./morpheus_adapters",
    "dataset": "./dataset.jsonl",
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "epochs": 3,
    "warmup_steps": 100,
    "save_steps": 500,
    "eval_steps": 500,
    "quantization": "4bit",
    "lora_target": "all",  # or ["q_proj", "k_proj", "v_proj", "o_proj"]
}

TRAINING_SCRIPT = """#!/bin/bash
# Morpheus QLoRA Training Script
# Generated automatically

set -e

BASE_MODEL="{base_model}"
OUTPUT_DIR="{output_dir}"
DATASET="{dataset}"

# LlamaFactory training
llamafactory-cli train \\
    --stage sft \\
    --model_name_or_path $BASE_MODEL \\
    --template qwen2 \\
    --dataset_dir $(dirname $DATASET) \\
    --dataset $(basename $DATASET) \\
    --output_dir $OUTPUT_DIR \\
    --overwrite_cache \\
    --do_train \\
    --finetuning_type lora \\
    --lora_rank {lora_rank} \\
    --lora_alpha {lora_alpha} \\
    --lora_dropout {lora_dropout} \\
    --lora_target {lora_target} \\
    --quantization_bit {quantization_bit} \\
    --bf16 True \\
    --batch_size {batch_size} \\
    --learning_rate {learning_rate} \\
    --num_train_epochs {epochs} \\
    --warmup_steps {warmup_steps} \\
    --save_steps {save_steps} \\
    --eval_steps {eval_steps} \\
    --log_steps 10 \\
    --logging_steps 10

echo "Training complete! Adapter saved to $OUTPUT_DIR"
"""


def generate_training_script(config: dict, output_path: Path):
    """Generate LlamaFactory training script from config."""
    # Determine quantization bit
    quant_bit = 4 if config.get("quantization") == "4bit" else 8 if config.get("quantization") == "8bit" else 0
    
    # Handle lora_target (list or "all")
    lora_target = config.get("lora_target", "all")
    if isinstance(lora_target, list):
        lora_target = ",".join(lora_target)
    
    script_content = TRAINING_SCRIPT.format(
        base_model=config.get("base_model", "qwen2.5:7b"),
        output_dir=config.get("output_dir", "./morpheus_adapters"),
        dataset=config.get("dataset", "./dataset.jsonl"),
        lora_rank=config.get("lora_rank", 64),
        lora_alpha=config.get("lora_alpha", 128),
        lora_dropout=config.get("lora_dropout", 0.05),
        lora_target=lora_target,
        quantization_bit=quant_bit,
        batch_size=config.get("batch_size", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        epochs=config.get("epochs", 3),
        warmup_steps=config.get("warmup_steps", 100),
        save_steps=config.get("save_steps", 500),
        eval_steps=config.get("eval_steps", 500),
    )
    
    output_path.write_text(script_content)
    output_path.chmod(0o755)
    return output_path


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    required = ["llamafactory-cli", "python3"]
    missing = []
    
    for cmd in required:
        result = subprocess.run(
            ["which", cmd],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            missing.append(cmd)
    
    return len(missing) == 0, missing


def train(
    base_model: str = typer.Option("qwen2.5:7b", help="Base model name"),
    dataset: Path = typer.Option(Path("dataset.jsonl"), help="Training dataset"),
    output_dir: Path = typer.Option(Path("morpheus_adapters"), help="Output directory"),
    lora_rank: int = typer.Option(64, help="LoRA rank"),
    lora_alpha: int = typer.Option(128, help="LoRA alpha"),
    epochs: int = typer.Option(3, help="Training epochs"),
    dry_run: bool = typer.Option(False, help="Generate script without running"),
):
    """Run QLoRA fine-tuning on session dataset.
    
    Requires:
    - llamafactory-cli installed (pip install llamafactory)
    - Sufficient GPU VRAM (7B model ≈ 6-8GB with 4bit)
    """
    config = {
        "base_model": base_model,
        "dataset": str(dataset.absolute()),
        "output_dir": str(output_dir.absolute()),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.05,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "epochs": epochs,
        "warmup_steps": 100,
        "save_steps": 500,
        "eval_steps": 500,
        "quantization": "4bit",
        "lora_target": "all",
    }
    
    # Check dependencies
    ok, missing = check_dependencies()
    if not ok:
        console.print(f"[red]Missing dependencies: {', '.join(missing)}[/red]")
        console.print("[yellow]Install with: pip install llamafactory[/yellow]")
        raise typer.Exit(1)
    
    # Check dataset exists
    if not dataset.exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        console.print("[yellow]Run 'morpheus consolidate' first[/yellow]")
        raise typer.Exit(1)
    
    # Generate training script
    script_path = Path("morpheus_train.sh")
    generate_training_script(config, script_path)
    
    console.print(Panel.fit(
        f"[green]Training script generated[/green]\n"
        f"Model: {base_model}\n"
        f"Dataset: {dataset.name}\n"
        f"Output: {output_dir}/\n"
        f"LoRA rank: {lora_rank}",
        title="Morpheus Training"
    ))
    
    if dry_run:
        console.print(f"[yellow]Dry run - script saved to {script_path}[/yellow]")
        return
    
    # Run training
    console.print("[blue]Starting training... (this may take 30-60 minutes)[/blue]")
    
    result = subprocess.run(["bash", str(script_path)])
    
    if result.returncode == 0:
        console.print("[green]✓ Training complete![/green]")
        console.print(f"[green]Adapter: {output_dir}/[/green]")
    else:
        console.print("[red]✗ Training failed[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(train)
