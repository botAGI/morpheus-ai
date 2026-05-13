"""
Evaluation Harness for Morpheus Adapter.

Tests the fine-tuned adapter on held-out session data to measure
if the model has learned project context.
"""
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

console = Console()

EVAL_PROMPT = """You are Morpheus, an AI assistant that remembers everything about the project.

Based on your training, answer this question about the project:

Question: {question}

Answer what you know. If you don't know, say you don't have that information."""


def load_adapter(adapter_path: Path, base_model: str = "qwen2.5:7b") -> bool:
    """Check if adapter exists and is loadable."""
    if not adapter_path.exists():
        return False
    
    # Check for adapter files
    adapter_files = list(adapter_path.glob("*.safetensors"))
    return len(adapter_files) > 0


def query_model(
    prompt: str,
    base_model: str = "qwen2.5:7b",
    adapter_path: Optional[Path] = None,
    temperature: float = 0.7,
    max_tokens: int = 500
) -> str:
    """Query the model with optional LoRA adapter."""
    # Ollama command
    cmd = ["ollama", "generate", base_model, prompt]
    
    if adapter_path and adapter_path.exists():
        console.print(f"[yellow]Note: Adapter {adapter_path} not auto-loaded in Ollama[/yellow]")
        console.print("[yellow]Load manually: ollama run qwen2.5:7b --adapter {adapter_path}[/yellow]")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    
    return result.stdout.strip()


def run_eval(
    adapter_path: Path = typer.Option(Path("morpheus_adapters"), help="LoRA adapter path"),
    base_model: str = typer.Option("qwen2.5:7b", help="Base model"),
    test_file: Path = typer.Option(Path("eval_questions.jsonl"), help="Test questions"),
    output: Path = typer.Option(Path("eval_results.jsonl"), help="Results output"),
):
    """Run evaluation on held-out questions.
    
    Test file format (JSONL):
    {"question": "...", "expected_keywords": ["keyword1", "keyword2"]}
    """
    if not load_adapter(adapter_path, base_model):
        console.print(f"[red]Adapter not found: {adapter_path}[/red]")
        console.print("[yellow]Run 'morpheus train' first[/yellow]")
        raise typer.Exit(1)
    
    if not test_file.exists():
        console.print(f"[red]Test file not found: {test_file}[/red]")
        console.print("[yellow]Create eval_questions.jsonl with test questions[/yellow]")
        raise typer.Exit(1)
    
    results = []
    total = 0
    passed = 0
    
    with open(test_file) as f:
        lines = f.readlines()
    
    console.print(f"[blue]Running evaluation on {len(lines)} questions...[/blue]\n")
    
    for i, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
            question = item["question"]
            expected = item.get("expected_keywords", [])
        except json.JSONDecodeError:
            continue
        
        total += 1
        prompt = EVAL_PROMPT.format(question=question)
        
        console.print(f"[cyan]Q{i}:[/cyan] {question[:80]}...")
        answer = query_model(prompt, base_model, adapter_path)
        
        # Simple keyword matching
        keywords_found = sum(1 for kw in expected if kw.lower() in answer.lower())
        score = keywords_found / len(expected) if expected else 0
        passed += 1 if score >= 0.5 else 0
        
        results.append({
            "question": question,
            "answer": answer,
            "expected": expected,
            "keywords_found": keywords_found,
            "score": score
        })
        
        console.print(f"   [green]Answer:[/green] {answer[:150]}...")
        console.print(f"   [yellow]Score: {score:.0%} ({keywords_found}/{len(expected)})[/yellow]\n")
    
    # Summary
    success_rate = (passed / total * 100) if total > 0 else 0
    
    table = Table(title="Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Questions", str(total))
    table.add_row("Passed (≥50% keywords)", str(passed))
    table.add_row("Success Rate", f"{success_rate:.1f}%")
    
    console.print("\n")
    console.print(table)
    
    # Save results
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    console.print(f"\n[green]✓ Results saved to {output}[/green]")


def create_sample_eval():
    """Create sample evaluation file."""
    sample = [
        {"question": "What is the Morpheus project about?", "expected_keywords": ["state", "compiler", "agent"]},
        {"question": "What integrations are supported?", "expected_keywords": ["gmail", "github", "calendar"]},
        {"question": "How does provenance work?", "expected_keywords": ["receipt", "signature", "evidence"]},
    ]
    
    output_path = Path("eval_questions.jsonl")
    with open(output_path, "w") as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    console.print(f"[green]✓ Created sample evaluation: {output_path}[/green]")


if __name__ == "__main__":
    typer.run(run_eval)
