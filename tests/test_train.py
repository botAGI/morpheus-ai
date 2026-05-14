"""
Tests for morpheus.training.train.
"""
import importlib
import shlex

import click
import pytest

from morpheus.training.train import generate_training_script

train_module = importlib.import_module("morpheus.training.train")


def test_generate_training_script_quotes_shell_path_variables(tmp_path):
    script_path = tmp_path / "morpheus_train.sh"

    generate_training_script(
        {
            "base_model": "models/qwen 2.5",
            "dataset": str(tmp_path / "data set" / "dataset.jsonl"),
            "output_dir": str(tmp_path / "adapter output"),
        },
        script_path,
    )

    script = script_path.read_text()
    assert '--model_name_or_path "$BASE_MODEL"' in script
    assert '--dataset_dir "$(dirname "$DATASET")"' in script
    assert '--dataset "$(basename "$DATASET")"' in script
    assert '--output_dir "$OUTPUT_DIR"' in script


def test_generate_training_script_shell_quotes_config_values(tmp_path):
    script_path = tmp_path / "morpheus_train.sh"
    config = {
        "base_model": "models/qwen $(touch model_pwned)",
        "dataset": str(tmp_path / "data $(touch dataset_pwned)" / "dataset.jsonl"),
        "output_dir": str(tmp_path / "adapter $(touch output_pwned)"),
    }

    generate_training_script(config, script_path)

    script = script_path.read_text()
    assert f"BASE_MODEL={shlex.quote(config['base_model'])}" in script
    assert f"DATASET={shlex.quote(config['dataset'])}" in script
    assert f"OUTPUT_DIR={shlex.quote(config['output_dir'])}" in script


def test_generate_training_script_shell_quotes_scalar_training_values(tmp_path):
    script_path = tmp_path / "morpheus_train.sh"
    config = {"lora_rank": "64 $(touch rank_pwned)"}

    generate_training_script(config, script_path)

    script = script_path.read_text()
    assert f"--lora_rank {shlex.quote(config['lora_rank'])}" in script
    assert "--lora_rank 64 $(touch rank_pwned)" not in script


def test_generate_training_script_accepts_non_string_lora_target_items(tmp_path):
    script_path = tmp_path / "morpheus_train.sh"

    generate_training_script({"lora_target": ["q_proj", 123]}, script_path)

    script = script_path.read_text()
    assert "--lora_target q_proj,123" in script


def test_generate_training_script_creates_parent_directory(tmp_path):
    script_path = tmp_path / "scripts" / "morpheus_train.sh"

    generate_training_script({}, script_path)

    assert script_path.exists()


def test_train_dry_run_generates_script_without_dependency_check(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instruction":"Q","output":"A"}\n')
    monkeypatch.chdir(tmp_path)

    def fail_dependency_check():
        raise AssertionError("dry-run should not check runtime training dependencies")

    monkeypatch.setattr(train_module, "check_dependencies", fail_dependency_check)

    train_module.train(
        base_model="qwen2.5:7b",
        dataset=dataset,
        output_dir=tmp_path / "adapter",
        lora_rank=64,
        lora_alpha=128,
        epochs=3,
        dry_run=True,
    )

    assert (tmp_path / "morpheus_train.sh").exists()


def test_train_non_dry_run_checks_dependencies_first(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instruction":"Q","output":"A"}\n')
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(train_module, "check_dependencies", lambda: (False, ["llamafactory-cli"]))

    with pytest.raises(click.exceptions.Exit):
        train_module.train(
            base_model="qwen2.5:7b",
            dataset=dataset,
            output_dir=tmp_path / "adapter",
            lora_rank=64,
            lora_alpha=128,
            epochs=3,
            dry_run=False,
        )
    assert not (tmp_path / "morpheus_train.sh").exists()


def test_train_rejects_dataset_directory(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(click.exceptions.Exit):
        train_module.train(
            base_model="qwen2.5:7b",
            dataset=dataset,
            output_dir=tmp_path / "adapter",
            lora_rank=64,
            lora_alpha=128,
            epochs=3,
            dry_run=True,
        )

    assert not (tmp_path / "morpheus_train.sh").exists()


def test_train_rejects_symlinked_dataset(monkeypatch, tmp_path):
    external_dataset = tmp_path / "external-dataset.jsonl"
    external_dataset.write_text('{"instruction":"Q","output":"A"}\n')
    dataset = tmp_path / "dataset.jsonl"
    try:
        dataset.symlink_to(external_dataset)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(click.exceptions.Exit):
        train_module.train(
            base_model="qwen2.5:7b",
            dataset=dataset,
            output_dir=tmp_path / "adapter",
            lora_rank=64,
            lora_alpha=128,
            epochs=3,
            dry_run=True,
        )

    assert not (tmp_path / "morpheus_train.sh").exists()


def test_train_rejects_symlinked_output_dir(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instruction":"Q","output":"A"}\n')
    external_adapter_dir = tmp_path / "external-adapter"
    external_adapter_dir.mkdir()
    output_dir = tmp_path / "adapter"
    try:
        output_dir.symlink_to(external_adapter_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(click.exceptions.Exit):
        train_module.train(
            base_model="qwen2.5:7b",
            dataset=dataset,
            output_dir=output_dir,
            lora_rank=64,
            lora_alpha=128,
            epochs=3,
            dry_run=True,
        )

    assert not (tmp_path / "morpheus_train.sh").exists()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("lora_rank", 0),
        ("lora_alpha", 0),
        ("epochs", 0),
    ],
)
def test_train_rejects_non_positive_numeric_options(monkeypatch, tmp_path, option, value):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instruction":"Q","output":"A"}\n')
    monkeypatch.chdir(tmp_path)
    kwargs = {
        "base_model": "qwen2.5:7b",
        "dataset": dataset,
        "output_dir": tmp_path / "adapter",
        "lora_rank": 64,
        "lora_alpha": 128,
        "epochs": 3,
        "dry_run": True,
    }
    kwargs[option] = value

    with pytest.raises(click.exceptions.Exit):
        train_module.train(**kwargs)

    assert not (tmp_path / "morpheus_train.sh").exists()


def test_train_rejects_blank_base_model(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instruction":"Q","output":"A"}\n')
    monkeypatch.chdir(tmp_path)

    with pytest.raises(click.exceptions.Exit):
        train_module.train(
            base_model="   ",
            dataset=dataset,
            output_dir=tmp_path / "adapter",
            lora_rank=64,
            lora_alpha=128,
            epochs=3,
            dry_run=True,
        )

    assert not (tmp_path / "morpheus_train.sh").exists()


def test_train_reports_unwritable_training_script(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instruction":"Q","output":"A"}\n')
    monkeypatch.chdir(tmp_path)
    (tmp_path / "morpheus_train.sh").mkdir()

    with pytest.raises(click.exceptions.Exit):
        train_module.train(
            base_model="qwen2.5:7b",
            dataset=dataset,
            output_dir=tmp_path / "adapter",
            lora_rank=64,
            lora_alpha=128,
            epochs=3,
            dry_run=True,
        )


def test_train_rejects_symlinked_training_script(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instruction":"Q","output":"A"}\n')
    external_script = tmp_path / "external-train.sh"
    external_script.write_text("do not modify")
    (tmp_path / "morpheus_train.sh").symlink_to(external_script)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(click.exceptions.Exit):
        train_module.train(
            base_model="qwen2.5:7b",
            dataset=dataset,
            output_dir=tmp_path / "adapter",
            lora_rank=64,
            lora_alpha=128,
            epochs=3,
            dry_run=True,
        )

    assert external_script.read_text() == "do not modify"
