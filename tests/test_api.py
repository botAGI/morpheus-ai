"""
Tests for morpheus.api.server.
"""
import json

import pytest

from morpheus.core.config import MorpheusConfig
from morpheus.core.provenance import compute_sha256_file

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient",
    reason="API tests require fastapi to be installed",
)
TestClient = fastapi_testclient.TestClient


def api_client() -> TestClient:
    from morpheus.api.server import app

    return TestClient(app)


def test_health_returns_version():
    client = api_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


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
