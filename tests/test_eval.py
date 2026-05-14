"""
Tests for morpheus.training.eval.
"""
import importlib
import json

import click
import pytest


eval_module = importlib.import_module("morpheus.training.eval")


def test_load_adapter_rejects_symlinked_adapter_directory(tmp_path):
    outside_adapter = tmp_path / "outside-adapter"
    outside_adapter.mkdir()
    (outside_adapter / "adapter.safetensors").write_text("stub")
    adapter_dir = tmp_path / "adapter"
    try:
        adapter_dir.symlink_to(outside_adapter, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    assert not eval_module.load_adapter(adapter_dir)


def test_load_adapter_ignores_symlinked_adapter_files(tmp_path):
    outside_adapter_file = tmp_path / "outside-adapter.safetensors"
    outside_adapter_file.write_text("stub")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    try:
        (adapter_dir / "adapter.safetensors").symlink_to(outside_adapter_file)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    assert not eval_module.load_adapter(adapter_dir)


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


def test_run_eval_exits_when_no_valid_questions(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter.safetensors").write_text("stub")
    test_file = tmp_path / "eval_questions.jsonl"
    output = tmp_path / "eval_results.jsonl"
    test_file.write_text('{"expected_keywords":["ignored"]}\n{not json}\n')

    with pytest.raises(click.exceptions.Exit):
        eval_module.run_eval(
            adapter_path=adapter_dir,
            base_model="qwen2.5:7b",
            test_file=test_file,
            output=output,
        )

    assert not output.exists()


def test_run_eval_exits_when_question_file_unreadable(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter.safetensors").write_text("stub")
    test_file = tmp_path / "eval_questions.jsonl"
    output = tmp_path / "eval_results.jsonl"
    test_file.mkdir()

    with pytest.raises(click.exceptions.Exit):
        eval_module.run_eval(
            adapter_path=adapter_dir,
            base_model="qwen2.5:7b",
            test_file=test_file,
            output=output,
        )

    assert not output.exists()


def test_run_eval_exits_when_model_query_errors(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter.safetensors").write_text("stub")
    test_file = tmp_path / "eval_questions.jsonl"
    output = tmp_path / "eval_results.jsonl"
    test_file.write_text('{"question":"What changed?","expected_keywords":["receipt"]}\n')
    monkeypatch.setattr(
        eval_module,
        "query_model",
        lambda *args, **kwargs: "Error: ollama executable not found",
    )

    with pytest.raises(click.exceptions.Exit):
        eval_module.run_eval(
            adapter_path=adapter_dir,
            base_model="qwen2.5:7b",
            test_file=test_file,
            output=output,
        )

    assert not output.exists()


def test_run_eval_exits_when_output_file_unwritable(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter.safetensors").write_text("stub")
    test_file = tmp_path / "eval_questions.jsonl"
    output = tmp_path / "eval_results.jsonl"
    test_file.write_text('{"question":"What changed?","expected_keywords":["receipt"]}\n')
    output.mkdir()
    monkeypatch.setattr(eval_module, "query_model", lambda *args, **kwargs: "receipt chain")

    with pytest.raises(click.exceptions.Exit):
        eval_module.run_eval(
            adapter_path=adapter_dir,
            base_model="qwen2.5:7b",
            test_file=test_file,
            output=output,
        )


def test_run_eval_exits_when_output_file_is_symlink(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter.safetensors").write_text("stub")
    test_file = tmp_path / "eval_questions.jsonl"
    output = tmp_path / "eval_results.jsonl"
    external_output = tmp_path / "external-results.jsonl"
    test_file.write_text('{"question":"What changed?","expected_keywords":["receipt"]}\n')
    external_output.write_text("do not modify")
    output.symlink_to(external_output)
    monkeypatch.setattr(eval_module, "query_model", lambda *args, **kwargs: "receipt chain")

    with pytest.raises(click.exceptions.Exit):
        eval_module.run_eval(
            adapter_path=adapter_dir,
            base_model="qwen2.5:7b",
            test_file=test_file,
            output=output,
        )

    assert external_output.read_text() == "do not modify"


def test_query_model_reports_missing_ollama(monkeypatch):
    def raise_missing_executable(*args, **kwargs):
        raise FileNotFoundError("ollama")

    monkeypatch.setattr(eval_module.subprocess, "run", raise_missing_executable)

    result = eval_module.query_model("prompt")

    assert result == "Error: ollama executable not found"


def test_query_model_reports_timeout(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise eval_module.subprocess.TimeoutExpired(cmd=["ollama"], timeout=60)

    monkeypatch.setattr(eval_module.subprocess, "run", raise_timeout)

    result = eval_module.query_model("prompt")

    assert result == "Error: model query timed out after 60s"


def test_query_model_rejects_blank_base_model_without_subprocess(monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("blank base_model should not invoke ollama")

    monkeypatch.setattr(eval_module.subprocess, "run", fail_run)

    result = eval_module.query_model("prompt", base_model="   ")

    assert result == "Error: base_model must not be blank"


def test_query_model_uses_ollama_run(monkeypatch):
    calls = []

    class Completed:
        returncode = 0
        stdout = "answer\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return Completed()

    monkeypatch.setattr(eval_module.subprocess, "run", fake_run)

    result = eval_module.query_model("prompt", base_model="qwen2.5:7b")

    assert result == "answer"
    assert calls == [
        (
            ["ollama", "run", "qwen2.5:7b", "prompt"],
            {"capture_output": True, "text": True, "timeout": 60},
        )
    ]


def test_query_model_prints_adapter_path_in_manual_load_hint(monkeypatch, tmp_path, capsys):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    class Completed:
        returncode = 0
        stdout = "answer\n"
        stderr = ""

    monkeypatch.setattr(eval_module.subprocess, "run", lambda *args, **kwargs: Completed())

    eval_module.query_model("prompt", adapter_path=adapter_dir)

    captured = capsys.readouterr().out
    assert str(adapter_dir) in captured.replace("\n", "")
    assert "Load manually:" in captured
    assert "{adapter_path}" not in captured


def test_create_sample_eval_exits_when_output_file_unwritable(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "eval_questions.jsonl").mkdir()

    with pytest.raises(click.exceptions.Exit):
        eval_module.create_sample_eval()


def test_create_sample_eval_rejects_symlinked_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    external_output = tmp_path / "external-eval-questions.jsonl"
    external_output.write_text("do not modify")
    (tmp_path / "eval_questions.jsonl").symlink_to(external_output)

    with pytest.raises(click.exceptions.Exit):
        eval_module.create_sample_eval()

    assert external_output.read_text() == "do not modify"
