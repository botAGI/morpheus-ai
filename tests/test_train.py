"""
Tests for morpheus.training.train.
"""
import shlex

from morpheus.training.train import generate_training_script


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
