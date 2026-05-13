"""
Tests for morpheus.cli.
"""
import json
from pathlib import Path

from typer.testing import CliRunner

import morpheus.cli as cli_module
from morpheus.cli import app
from morpheus.core.provenance import build_receipt, compute_sha256_file, receipt_file_name


def write_out_of_filename_order_receipt_chain(morpheus_dir: Path):
    private_key_path = morpheus_dir / "keys" / "local.key"
    receipts_dir = morpheus_dir / "receipts"
    root = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="1" * 64,
        sources_data=[],
        private_key_path=private_key_path,
        receipt_id="rcpt_z_root",
    )
    root_path = receipts_dir / receipt_file_name(root["receipt_id"])
    root_path.write_text(json.dumps(root, default=str))

    tail = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="2" * 64,
        sources_data=[],
        private_key_path=private_key_path,
        prev_hash=compute_sha256_file(root_path),
        receipt_id="rcpt_a_tail",
    )
    tail_path = receipts_dir / receipt_file_name(tail["receipt_id"])
    tail_path.write_text(json.dumps(tail, default=str))

    return root_path, tail_path


def write_unlinked_receipts(morpheus_dir: Path):
    private_key_path = morpheus_dir / "keys" / "local.key"
    receipts_dir = morpheus_dir / "receipts"
    for receipt_id, wake_sha in [
        ("rcpt_a_root", "1" * 64),
        ("rcpt_b_root", "2" * 64),
    ]:
        receipt = build_receipt(
            state_dict={"claims": [], "evidence": []},
            wake_md_sha=wake_sha,
            sources_data=[],
            private_key_path=private_key_path,
            receipt_id=receipt_id,
        )
        (receipts_dir / receipt_file_name(receipt["receipt_id"])).write_text(
            json.dumps(receipt, default=str)
        )


def test_init_creates_morpheus_state(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0, result.output
        project_root = Path.cwd()
        assert (project_root / ".morpheus" / "morpheus.toml").exists()
        assert (project_root / ".morpheus" / "keys" / "local.key").exists()
        assert (project_root / ".morpheus" / "keys" / "local.pub").exists()


def test_compile_preserves_receipts_with_same_timestamp(tmp_path, monkeypatch):
    runner = CliRunner()
    receipt_ids = [
        "rcpt_20260513T114006Z_first",
        "rcpt_20260513T114006Z_second",
    ]

    def build_receipt_with_same_timestamp(*args, **kwargs):
        receipt_id = receipt_ids.pop(0)
        return {
            "receipt_id": receipt_id,
            "claim_count": {"active": 1, "superseded": 0, "unverified": 0},
            "issued_at": "2026-05-13T11:40:06Z",
            "signature": {"algo": "ed25519", "key_id": "local", "signature_b64": "sig"},
        }

    monkeypatch.setattr(cli_module, "build_receipt", build_receipt_with_same_timestamp)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: preserve receipt files\n")

        init_result = runner.invoke(app, ["init"])
        first_compile = runner.invoke(app, ["compile"])
        second_compile = runner.invoke(app, ["compile"])

        assert init_result.exit_code == 0, init_result.output
        assert first_compile.exit_code == 0, first_compile.output
        assert second_compile.exit_code == 0, second_compile.output
        receipt_files = sorted((Path.cwd() / ".morpheus" / "receipts").glob("receipt_*.json"))
        assert [path.name for path in receipt_files] == [
            "receipt_rcpt_20260513T114006Z_first.json",
            "receipt_rcpt_20260513T114006Z_second.json",
        ]


def test_compile_receipt_hashes_final_wake_file(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: hash final wake\n")

        init_result = runner.invoke(app, ["init"])
        compile_result = runner.invoke(app, ["compile"])

        assert init_result.exit_code == 0, init_result.output
        assert compile_result.exit_code == 0, compile_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        receipt_path = next((morpheus_dir / "receipts").glob("receipt_*.json"))
        receipt = json.loads(receipt_path.read_text())

        assert receipt["wake_md_sha256"] == compute_sha256_file(morpheus_dir / "WAKE.md")


def test_compile_receipt_hashes_final_state_file(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: hash final state\n")

        init_result = runner.invoke(app, ["init"])
        compile_result = runner.invoke(app, ["compile"])

        assert init_result.exit_code == 0, init_result.output
        assert compile_result.exit_code == 0, compile_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        receipt_path = next((morpheus_dir / "receipts").glob("receipt_*.json"))
        receipt = json.loads(receipt_path.read_text())

        assert receipt["state_json_sha256"] == compute_sha256_file(morpheus_dir / "state.json")


def test_compile_state_records_receipt_id(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: record receipt id\n")

        init_result = runner.invoke(app, ["init"])
        compile_result = runner.invoke(app, ["compile"])

        assert init_result.exit_code == 0, init_result.output
        assert compile_result.exit_code == 0, compile_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        receipt_path = next((morpheus_dir / "receipts").glob("receipt_*.json"))
        receipt = json.loads(receipt_path.read_text())
        state = json.loads((morpheus_dir / "state.json").read_text())

        assert state["receipt_id"] == receipt["receipt_id"]


def test_compile_receipt_hashes_final_evidence_file(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: hash final evidence\n")

        init_result = runner.invoke(app, ["init"])
        compile_result = runner.invoke(app, ["compile"])

        assert init_result.exit_code == 0, init_result.output
        assert compile_result.exit_code == 0, compile_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        receipt_path = next((morpheus_dir / "receipts").glob("receipt_*.json"))
        receipt = json.loads(receipt_path.read_text())

        assert (morpheus_dir / "evidence.jsonl").exists()
        assert receipt["evidence_jsonl_sha256"] == compute_sha256_file(
            morpheus_dir / "evidence.jsonl"
        )


def test_compile_reports_invalid_config_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        (Path.cwd() / ".morpheus" / "morpheus.toml").write_text("{not toml")

        result = runner.invoke(app, ["compile"])

        assert result.exit_code == 1
        assert "Config invalid" in result.output


def test_verify_quick_reports_receipt_chain_tail_not_filename_latest(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        write_out_of_filename_order_receipt_chain(morpheus_dir)

        result = runner.invoke(app, ["verify"])

        assert result.exit_code == 0, result.output
        assert "rcpt_a_tail" in result.output
        assert "rcpt_z_root" not in result.output


def test_status_reports_receipt_chain_tail_not_filename_latest(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        (morpheus_dir / "state.json").write_text(
            json.dumps({
                "sources": [],
                "claims": [],
                "evidence": [],
                "compiled_at": "2026-05-13T00:00:00+00:00",
            })
        )
        write_out_of_filename_order_receipt_chain(morpheus_dir)

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "rcpt_a_tail" in result.output
        assert "rcpt_z_root" not in result.output


def test_verify_quick_reports_invalid_receipt_chain_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        write_unlinked_receipts(Path.cwd() / ".morpheus")

        result = runner.invoke(app, ["verify"])

        assert result.exit_code == 1
        assert "Receipt chain invalid" in result.output
        assert "expected exactly one receipt chain tail" in result.output


def test_status_reports_invalid_receipt_chain_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        (morpheus_dir / "state.json").write_text(
            json.dumps({
                "sources": [],
                "claims": [],
                "evidence": [],
                "compiled_at": "2026-05-13T00:00:00+00:00",
            })
        )
        write_unlinked_receipts(morpheus_dir)

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 1
        assert "Receipt chain invalid" in result.output
        assert "expected exactly one receipt chain tail" in result.output


def test_status_reports_invalid_state_json_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        (morpheus_dir / "state.json").write_text("{not json")

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 1
        assert "State file invalid" in result.output


def test_train_dry_run_skips_cli_dependency_check(tmp_path, monkeypatch):
    runner = CliRunner()

    def fail_dependency_check():
        raise AssertionError("dry-run should not check runtime training dependencies")

    monkeypatch.setattr(cli_module, "check_dependencies", fail_dependency_check)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("dataset.jsonl").write_text('{"instruction":"Q","output":"A"}\n')

        result = runner.invoke(app, ["train", "--dataset", "dataset.jsonl", "--dry-run"])

        assert result.exit_code == 0, result.output
        assert Path("morpheus_train.sh").exists()
        script = Path("morpheus_train.sh").read_text()
        assert "OptionInfo" not in script
        assert "--lora_alpha 128" in script


def test_train_dry_run_accepts_lora_alpha_option(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(cli_module, "check_dependencies", lambda: (_ for _ in ()).throw(
        AssertionError("dry-run should not check runtime training dependencies")
    ))

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("dataset.jsonl").write_text('{"instruction":"Q","output":"A"}\n')

        result = runner.invoke(
            app,
            ["train", "--dataset", "dataset.jsonl", "--lora-alpha", "256", "--dry-run"],
        )

        assert result.exit_code == 0, result.output
        assert "--lora_alpha 256" in Path("morpheus_train.sh").read_text()
