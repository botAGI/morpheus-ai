"""
Tests for morpheus.cli.
"""
import json
from pathlib import Path

from typer.testing import CliRunner

import morpheus.cli as cli_module
from morpheus.cli import app
from morpheus.core.provenance import compute_sha256_file


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
