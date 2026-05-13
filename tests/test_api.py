"""
Tests for morpheus.api.server.
"""
import json

import pytest

from morpheus.core.config import MorpheusConfig
from morpheus.core.provenance import build_receipt, compute_sha256_file, receipt_file_name

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient",
    reason="API tests require fastapi to be installed",
)
TestClient = fastapi_testclient.TestClient


def api_client(raise_server_exceptions: bool = True) -> TestClient:
    from morpheus.api.server import app

    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def write_unlinked_receipts(morpheus_dir):
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


def test_health_returns_version():
    client = api_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_cors_preflight_does_not_allow_credentials_for_wildcard_origins():
    client = api_client()

    response = client.options(
        "/health",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


def test_compile_persists_state_and_receipt_for_status_and_verify(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: compile through API\n")
    client = api_client()

    compile_response = client.post("/compile", json={"project_root": str(tmp_path)})
    status_response = client.get("/status", params={"project_root": str(tmp_path)})
    verify_response = client.post("/verify", params={"project_root": str(tmp_path)})

    assert compile_response.status_code == 200
    compile_payload = compile_response.json()
    assert compile_payload["source_count"] == 1
    assert compile_payload["claim_count"]["active"] == 1
    assert compile_payload["receipt_id"].startswith("rcpt_")
    assert (tmp_path / ".morpheus" / "WAKE.md").exists()
    assert (tmp_path / ".morpheus" / "state.json").exists()
    assert len(list((tmp_path / ".morpheus" / "receipts").glob("receipt_*.json"))) == 1

    assert status_response.status_code == 200
    assert status_response.json()["initialized"] is True
    assert status_response.json()["claims"] == 1

    assert verify_response.status_code == 200
    assert verify_response.json()["valid"] is True
    assert verify_response.json()["receipt_id"] == compile_payload["receipt_id"]


def test_compile_receipt_hashes_final_wake_file(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: hash final API wake\n")
    client = api_client()

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 200
    morpheus_dir = tmp_path / ".morpheus"
    receipt_path = next((morpheus_dir / "receipts").glob("receipt_*.json"))
    receipt = json.loads(receipt_path.read_text())
    assert receipt["wake_md_sha256"] == compute_sha256_file(morpheus_dir / "WAKE.md")


def test_compile_receipt_hashes_final_state_file(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: hash final API state\n")
    client = api_client()

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 200
    morpheus_dir = tmp_path / ".morpheus"
    receipt_path = next((morpheus_dir / "receipts").glob("receipt_*.json"))
    receipt = json.loads(receipt_path.read_text())
    assert receipt["state_json_sha256"] == compute_sha256_file(morpheus_dir / "state.json")


def test_compile_state_records_receipt_id(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: record API receipt id\n")
    client = api_client()

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 200
    morpheus_dir = tmp_path / ".morpheus"
    receipt_path = next((morpheus_dir / "receipts").glob("receipt_*.json"))
    receipt = json.loads(receipt_path.read_text())
    state = json.loads((morpheus_dir / "state.json").read_text())
    assert state["receipt_id"] == receipt["receipt_id"]


def test_compile_receipt_hashes_final_evidence_file(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: hash final API evidence\n")
    client = api_client()

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 200
    morpheus_dir = tmp_path / ".morpheus"
    receipt_path = next((morpheus_dir / "receipts").glob("receipt_*.json"))
    receipt = json.loads(receipt_path.read_text())
    assert (morpheus_dir / "evidence.jsonl").exists()
    assert receipt["evidence_jsonl_sha256"] == compute_sha256_file(
        morpheus_dir / "evidence.jsonl"
    )


def test_verify_returns_invalid_response_for_broken_receipt_chain(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    morpheus_dir = tmp_path / ".morpheus"
    private_key_path = morpheus_dir / "keys" / "local.key"
    receipts_dir = morpheus_dir / "receipts"
    first = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="1" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    second = build_receipt(
        state_dict={"claims": [], "evidence": []},
        wake_md_sha="2" * 64,
        sources_data=[],
        private_key_path=private_key_path,
    )
    (receipts_dir / receipt_file_name(first["receipt_id"])).write_text(json.dumps(first))
    (receipts_dir / receipt_file_name(second["receipt_id"])).write_text(json.dumps(second))
    client = api_client()

    response = client.post("/verify", params={"project_root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    assert payload["receipt_id"] == "none"
    assert any("expected exactly one root receipt" in error for error in payload["errors"])


def test_verify_returns_bad_request_when_latest_receipt_tail_cannot_be_loaded(
    monkeypatch,
    tmp_path,
):
    MorpheusConfig(project_root=tmp_path).init_default()

    import morpheus.api.server as server_module
    import morpheus.core.verify as verify_module

    monkeypatch.setattr(verify_module, "verify_receipt_chain", lambda morpheus_dir: (True, []))

    def fail_latest_receipt(receipts_dir):
        raise ValueError("tail receipt changed during verification")

    monkeypatch.setattr(server_module, "latest_receipt_file", fail_latest_receipt)
    client = api_client(raise_server_exceptions=False)

    response = client.post("/verify", params={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Receipt chain invalid" in response.json()["detail"]
    assert "tail receipt changed during verification" in response.json()["detail"]


def test_verify_returns_bad_request_for_morpheus_state_file(tmp_path):
    (tmp_path / ".morpheus").write_text("not a directory")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/verify", params={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Not initialized" in response.json()["detail"]


def test_compile_returns_bad_request_for_broken_receipt_chain(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: do not compile onto broken chain\n")
    write_unlinked_receipts(tmp_path / ".morpheus")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Receipt chain invalid" in response.json()["detail"]
    assert "expected exactly one receipt chain root" in response.json()["detail"]


def test_compile_returns_bad_request_for_morpheus_state_file(tmp_path):
    (tmp_path / ".morpheus").write_text("not a directory")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Not initialized" in response.json()["detail"]
    assert "Signing failed" not in response.json()["detail"]


def test_compile_returns_bad_request_when_previous_receipt_hash_fails(
    monkeypatch,
    tmp_path,
):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: compile with racy previous receipt\n")

    import morpheus.api.server as server_module

    def fake_latest_receipt(receipts_dir):
        return receipts_dir / "receipt_old.json"

    def fail_sha256(path):
        raise OSError("permission denied")

    monkeypatch.setattr(server_module, "latest_receipt_file", fake_latest_receipt)
    monkeypatch.setattr(server_module, "compute_sha256_file", fail_sha256)
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Receipt chain invalid" in response.json()["detail"]
    assert "permission denied" in response.json()["detail"]


def test_compile_returns_bad_request_for_invalid_config(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / ".morpheus" / "morpheus.toml").write_text("{not toml")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Config invalid" in response.json()["detail"]


def test_compile_returns_bad_request_for_invalid_config_types(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / ".morpheus" / "morpheus.toml").write_text("watch_dirs = 123")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Config invalid" in response.json()["detail"]


def test_compile_returns_bad_request_for_unreadable_config(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    config_path = tmp_path / ".morpheus" / "morpheus.toml"
    config_path.unlink()
    config_path.mkdir()
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Config unreadable" in response.json()["detail"]


def test_compile_returns_bad_request_for_invalid_signing_key(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: compile through API with corrupted key\n")
    (tmp_path / ".morpheus" / "keys" / "local.key").write_bytes(b"bad")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Signing failed" in response.json()["detail"]


def test_compile_returns_bad_request_for_output_write_failures(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: compile API with blocked output path\n")
    (tmp_path / ".morpheus" / "WAKE.md").mkdir()
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Output write failed" in response.json()["detail"]


def test_get_wake_rejects_project_path_traversal(tmp_path, monkeypatch):
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    (tmp_path / "WAKE.md").write_text("secret parent wake")
    monkeypatch.chdir(safe_dir)
    client = api_client()

    response = client.get("/wake/%2E%2E")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid project name"


def test_get_wake_returns_bad_request_for_unreadable_wake_file(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "WAKE.md").mkdir()
    monkeypatch.chdir(tmp_path)
    client = api_client(raise_server_exceptions=False)

    response = client.get("/wake/project")

    assert response.status_code == 400
    assert "WAKE.md unreadable" in response.json()["detail"]


def test_status_returns_bad_request_for_invalid_state_json(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    (morpheus_dir / "state.json").write_text("{not json")
    client = api_client(raise_server_exceptions=False)

    response = client.get("/status", params={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "State file invalid" in response.json()["detail"]


def test_status_counts_only_list_state_collections(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
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
    client = api_client(raise_server_exceptions=False)

    response = client.get("/status", params={"project_root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"] == 0
    assert payload["claims"] == 0
    assert payload["evidence"] == 0
