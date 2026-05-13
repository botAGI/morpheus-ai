"""
Tests for morpheus.training.train.
"""
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
