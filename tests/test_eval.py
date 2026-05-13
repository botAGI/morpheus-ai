"""
Tests for morpheus.training.eval.
"""
import importlib
import json


eval_module = importlib.import_module("morpheus.training.eval")


def test_run_eval_skips_malformed_question_rows(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter.safetensors").write_text("stub")
    test_file = tmp_path / "eval_questions.jsonl"
    output = tmp_path / "eval_results.jsonl"
    test_file.write_text(
        '{"expected_keywords":["ignored"]}\n'
        '{not json}\n'
        '{"question":"How is provenance verified?","expected_keywords":["receipt"]}\n'
    )

    monkeypatch.setattr(eval_module, "query_model", lambda *args, **kwargs: "receipt chain")

    eval_module.run_eval(
        adapter_path=adapter_dir,
        base_model="qwen2.5:7b",
        test_file=test_file,
        output=output,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["question"] for row in rows] == ["How is provenance verified?"]
    assert rows[0]["score"] == 1
