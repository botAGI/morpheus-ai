"""
Tests for morpheus.api.server.
"""
from fastapi.testclient import TestClient

from morpheus.api.server import app
from morpheus.core.config import MorpheusConfig


def test_health_returns_version():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_compile_persists_state_and_receipt_for_status_and_verify(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: compile through API\n")
    client = TestClient(app)

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
