"""
Tests for morpheus.cli.
"""
import json
from pathlib import Path

import pytest
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


def test_init_rejects_morpheus_state_file_without_force_hint(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path(".morpheus").write_text("not a directory")

        result = runner.invoke(app, ["init"])

        assert result.exit_code == 1
        assert ".morpheus path is not a directory" in result.output
        assert "Use --force" not in result.output


def test_init_force_rejects_morpheus_state_file_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path(".morpheus").write_text("not a directory")

        result = runner.invoke(app, ["init", "--force"])

        assert result.exit_code == 1
        assert ".morpheus path is not a directory" in result.output
        assert "Initialization failed" not in result.output


def test_init_force_rejects_morpheus_symlink_without_writing_target(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        outside = tmp_path / "outside-morpheus"
        outside.mkdir()
        Path(".morpheus").symlink_to(outside, target_is_directory=True)

        result = runner.invoke(app, ["init", "--force"])

        assert result.exit_code == 1
        assert ".morpheus path must not be a symlink" in result.output
        assert not (outside / "morpheus.toml").exists()
        assert not (outside / "keys").exists()


def test_init_force_reports_key_generation_failures_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        local_key = Path.cwd() / ".morpheus" / "keys" / "local.key"
        local_key.mkdir(parents=True)

        result = runner.invoke(app, ["init", "--force"])

        assert result.exit_code == 1
        assert "Initialization failed" in result.output


def test_init_force_reports_invalid_public_key_path_without_success(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        public_key = Path.cwd() / ".morpheus" / "keys" / "local.pub"
        public_key.unlink()
        public_key.mkdir()

        result = runner.invoke(app, ["init", "--force"])

        assert result.exit_code == 1
        assert "Initialization failed" in result.output
        assert "local.pub" in result.output


def test_init_force_reports_invalid_config_path_without_success(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        config_path = Path.cwd() / ".morpheus" / "morpheus.toml"
        config_path.mkdir(parents=True)

        result = runner.invoke(app, ["init", "--force"])

        assert result.exit_code == 1
        assert "Initialization failed" in result.output
        assert "morpheus.toml" in result.output


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


def test_compile_rejects_morpheus_state_file_without_signing_error(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path(".morpheus").write_text("not a directory")

        result = runner.invoke(app, ["compile"])

        assert result.exit_code == 1
        assert "Not initialized" in result.output
        assert "Signing failed" not in result.output


def test_compile_rejects_morpheus_symlink_without_signing_error(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: compile with symlinked state\n")
        outside = tmp_path / "outside-morpheus"
        outside.mkdir()
        Path(".morpheus").symlink_to(outside, target_is_directory=True)

        result = runner.invoke(app, ["compile"])

        assert result.exit_code == 1
        assert "Not initialized" in result.output
        assert "Signing failed" not in result.output


def test_compile_reports_invalid_config_types_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        (Path.cwd() / ".morpheus" / "morpheus.toml").write_text("watch_dirs = 123")

        result = runner.invoke(app, ["compile"])

        assert result.exit_code == 1
        assert "Config invalid" in result.output


def test_compile_reports_unreadable_config_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        config_path = Path.cwd() / ".morpheus" / "morpheus.toml"
        config_path.unlink()
        config_path.mkdir()

        result = runner.invoke(app, ["compile"])

        assert result.exit_code == 1
        assert "Config unreadable" in result.output


def test_compile_reports_previous_receipt_hash_failures_without_traceback(
    monkeypatch,
    tmp_path,
):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: compile with racy previous receipt\n")
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output

        def fake_latest_receipt(receipts_dir):
            return receipts_dir / "receipt_old.json"

        def fail_sha256(path):
            raise OSError("permission denied")

        monkeypatch.setattr(cli_module, "latest_receipt_file", fake_latest_receipt)
        monkeypatch.setattr(cli_module, "compute_sha256_file", fail_sha256)

        result = runner.invoke(app, ["compile"])

        assert result.exit_code == 1
        assert "Receipt chain invalid" in result.output
        assert "permission denied" in result.output


def test_compile_rejects_receipts_path_file_before_signing(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: compile with invalid receipts path\n")
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        receipts_dir = Path.cwd() / ".morpheus" / "receipts"
        receipts_dir.rmdir()
        receipts_dir.write_text("not a directory")

        result = runner.invoke(app, ["compile"])

        assert result.exit_code == 1
        assert "Receipt chain invalid" in result.output
        assert "receipts path is not a directory" in result.output
        assert "Signing failed" not in result.output
        assert "Output write failed" not in result.output


def test_compile_reports_invalid_signing_key_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: compile with corrupted key\n")
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        (Path.cwd() / ".morpheus" / "keys" / "local.key").write_bytes(b"bad")

        result = runner.invoke(app, ["compile"])

        assert result.exit_code == 1
        assert "Signing failed" in result.output


def test_compile_rejects_symlinked_signing_key_without_creating_receipt(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: compile with symlinked signing key\n")
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        external_key = tmp_path / "external-local.key"
        local_key = morpheus_dir / "keys" / "local.key"
        local_key.replace(external_key)
        local_key.symlink_to(external_key)

        result = runner.invoke(app, ["compile"])

        assert result.exit_code == 1
        assert "Signing failed" in result.output
        assert "private signing key must not be a symlink" in result.output
        assert not list((morpheus_dir / "receipts").glob("receipt_*.json"))


def test_compile_reports_output_write_failures_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: compile with blocked output path\n")
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        (Path.cwd() / ".morpheus" / "WAKE.md").mkdir()

        result = runner.invoke(app, ["compile"])

        assert result.exit_code == 1
        assert "Output write failed" in result.output


@pytest.mark.parametrize(
    "relative_path",
    ["WAKE.md", "state.json", "evidence.jsonl", "receipts/audit.log"],
)
def test_compile_rejects_symlinked_output_artifacts_before_writing(tmp_path, relative_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("README.md").write_text("TODO: compile with symlinked output artifact\n")
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        external_output = tmp_path / "external-output"
        external_output.write_text("do not modify")
        output_path = morpheus_dir / relative_path
        if output_path.exists():
            output_path.unlink()
        output_path.symlink_to(external_output)

        result = runner.invoke(app, ["compile"])

        assert result.exit_code == 1
        assert "Output write failed" in result.output
        assert "must not be a symlink" in result.output
        assert external_output.read_text() == "do not modify"
        assert not list((morpheus_dir / "receipts").glob("receipt_*.json"))


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


def test_verify_rejects_morpheus_state_file_without_receipt_error(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path(".morpheus").write_text("not a directory")

        result = runner.invoke(app, ["verify"])

        assert result.exit_code == 1
        assert "Not initialized" in result.output
        assert "No receipts found" not in result.output


def test_verify_rejects_receipts_path_file_without_no_receipts_message(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        receipts_dir = Path.cwd() / ".morpheus" / "receipts"
        receipts_dir.rmdir()
        receipts_dir.write_text("not a directory")

        result = runner.invoke(app, ["verify"])

        assert result.exit_code == 1
        assert "Receipt chain invalid" in result.output
        assert "receipts path is not a directory" in result.output
        assert "No receipts found" not in result.output


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


def test_status_rejects_morpheus_state_file_without_compile_message(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path(".morpheus").write_text("not a directory")

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "Not initialized" in result.output
        assert "No compilation yet" not in result.output


def test_status_rejects_morpheus_symlink_without_reading_target(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        outside = tmp_path / "outside-morpheus"
        outside.mkdir()
        (outside / "state.json").write_text(
            json.dumps(
                {
                    "sources": [],
                    "claims": [],
                    "evidence": [],
                    "compiled_at": "2026-05-13T00:00:00Z",
                }
            )
        )
        Path(".morpheus").symlink_to(outside, target_is_directory=True)

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "Not initialized" in result.output
        assert "Project Status" not in result.output


def test_verify_quick_reports_invalid_receipt_chain_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        write_unlinked_receipts(Path.cwd() / ".morpheus")

        result = runner.invoke(app, ["verify"])

        assert result.exit_code == 1
        assert "Receipt chain invalid" in result.output
        assert "expected exactly one receipt chain root" in result.output


def test_verify_verbose_handles_non_collection_receipt_fields(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        receipt = {
            "receipt_id": "rcpt_bad_fields",
            "issued_at": "2026-05-13T00:00:00Z",
            "claim_count": None,
            "sources": None,
        }
        receipts_dir = Path.cwd() / ".morpheus" / "receipts"
        (receipts_dir / "receipt_rcpt_bad_fields.json").write_text(json.dumps(receipt))

        result = runner.invoke(app, ["verify", "--verbose"])

        assert result.exit_code == 0, result.output
        assert "rcpt_bad_fields" in result.output


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
        assert "expected exactly one receipt chain root" in result.output


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


def test_status_handles_non_string_compiled_at_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        (morpheus_dir / "state.json").write_text(
            json.dumps(
                {
                    "sources": [],
                    "claims": [],
                    "evidence": [],
                    "compiled_at": 1234567890,
                }
            )
        )

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "1234567890" in result.output


def test_status_handles_non_list_state_collections_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        morpheus_dir = Path.cwd() / ".morpheus"
        (morpheus_dir / "state.json").write_text(
            json.dumps(
                {
                    "sources": None,
                    "claims": "not a list",
                    "evidence": {"not": "a list"},
                    "compiled_at": "2026-05-13T00:00:00Z",
                }
            )
        )

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        assert "Project Status" in result.output


def test_wake_reports_unreadable_wake_file_without_traceback(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        init_result = runner.invoke(app, ["init"])
        assert init_result.exit_code == 0, init_result.output
        (Path.cwd() / ".morpheus" / "WAKE.md").mkdir()

        result = runner.invoke(app, ["wake"])

        assert result.exit_code == 1
        assert "WAKE.md unreadable" in result.output


def test_wake_rejects_morpheus_state_file_without_missing_wake_message(tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path(".morpheus").write_text("not a directory")

        result = runner.invoke(app, ["wake"])

        assert result.exit_code == 1
        assert "Not initialized" in result.output
        assert "No WAKE.md found" not in result.output


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


def test_eval_command_forwards_options(monkeypatch):
    runner = CliRunner()
    calls = []

    def fake_run_eval(adapter_path, base_model, test_file, output):
        calls.append(
            {
                "adapter_path": adapter_path,
                "base_model": base_model,
                "test_file": test_file,
                "output": output,
            }
        )

    import morpheus.training.eval as eval_module

    monkeypatch.setattr(eval_module, "run_eval", fake_run_eval)

    result = runner.invoke(
        app,
        [
            "eval",
            "--adapter-path",
            "adapter-dir",
            "--base-model",
            "qwen2.5:14b",
            "--test-file",
            "questions.jsonl",
            "--output",
            "results.jsonl",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "adapter_path": Path("adapter-dir"),
            "base_model": "qwen2.5:14b",
            "test_file": Path("questions.jsonl"),
            "output": Path("results.jsonl"),
        }
    ]


def test_consolidate_command_forwards_min_pairs_option(monkeypatch):
    runner = CliRunner()
    calls = []

    def fake_consolidate_sessions(
        sessions_dir,
        output_path,
        days,
        min_pairs,
        stats_output_path,
        verbose,
    ):
        calls.append(
            {
                "sessions_dir": sessions_dir,
                "output_path": output_path,
                "days": days,
                "min_pairs": min_pairs,
                "stats_output_path": stats_output_path,
                "verbose": verbose,
            }
        )

    monkeypatch.setattr(cli_module, "consolidate_sessions", fake_consolidate_sessions)

    result = runner.invoke(
        app,
        [
            "consolidate",
            "--sessions-dir",
            "sessions",
            "--output",
            "dataset.jsonl",
            "--days",
            "14",
            "--min-pairs",
            "3",
            "--stats-output",
            "stats.json",
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "sessions_dir": Path("sessions"),
            "output_path": Path("dataset.jsonl"),
            "days": 14,
            "min_pairs": 3,
            "stats_output_path": Path("stats.json"),
            "verbose": True,
        }
    ]


def test_integrate_list_does_not_require_service_argument():
    runner = CliRunner()

    result = runner.invoke(app, ["integrate", "--list"])

    assert result.exit_code == 0, result.output
    assert "Available Integrations" in result.output


def test_integrate_unknown_service_exits_with_error():
    runner = CliRunner()

    result = runner.invoke(app, ["integrate", "slack"])

    assert result.exit_code == 1
    assert "Unknown integration service" in result.output


def test_integrate_github_rejects_token_directory(monkeypatch, tmp_path):
    runner = CliRunner()
    token_path = tmp_path / ".morpheus" / "github_token.txt"
    token_path.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = runner.invoke(app, ["integrate", "github"])

    assert result.exit_code == 1
    assert "GitHub token path is not a file" in result.output
    assert "already configured" not in result.output
