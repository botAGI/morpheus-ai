"""
Tests for morpheus.api.server.
"""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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


def test_well_known_morpheus_manifest_exposes_agent_connect_url():
    client = api_client()

    response = client.get("/.well-known/morpheus.json")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "morpheus"
    assert payload["version"] == "0.1.0"
    assert payload["connect_url"] == "http://testserver/agent/connect"
    assert payload["handoff_url"] == "http://testserver/agent/handoff"
    assert payload["handoff_markdown_url"] == "http://testserver/agent/handoff.md"
    assert payload["agent_card_url"] == "http://testserver/.well-known/agent-card.json"
    assert payload["mcp_url"] == "http://testserver/mcp"
    assert payload["quickstart_url"] == "http://testserver/quickstart"
    assert payload["docs"]["human_quickstart"] == "README.md"


def test_quickstart_endpoint_returns_human_and_agent_launch_plan(tmp_path):
    client = api_client(raise_server_exceptions=False)

    response = client.get("/quickstart", params={"project_root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "morpheus"
    assert payload["project_root"] == str(tmp_path)
    assert payload["human_path"][0]["id"] == "install"
    assert payload["commands"]["install"][0] == "uvx --from morpheus-wake morpheus wake ."
    assert payload["commands"]["install"][1] == (
        "pipx run --spec morpheus-wake morpheus wake ."
    )
    assert "pip install -e" in payload["commands"]["development_install"][2]
    assert payload["commands"]["run_local"] == (
        "morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173"
    )
    assert payload["commands"]["run_lan"] == (
        "morpheus serve --ui --host 0.0.0.0 --port 8000 --ui-port 5173"
    )
    assert payload["connect"]["native"] == (
        f"http://testserver/agent/connect?project_root={str(tmp_path).replace('/', '%2F')}"
    )
    assert payload["connect"]["a2a_agent_card"] == "http://testserver/.well-known/agent-card.json"
    assert payload["connect"]["mcp"] == "http://testserver/mcp"
    assert payload["agent_path"][0]["id"] == "discover"
    assert "morpheus prepare-agent" in payload["copy_paste_agent_prompt"]


def test_quickstart_endpoint_uses_configured_ui_port(tmp_path, monkeypatch):
    monkeypatch.setenv("MORPHEUS_UI_PORT", "5179")
    client = api_client(raise_server_exceptions=False)

    response = client.get(
        "/quickstart",
        params={"project_root": str(tmp_path)},
        headers={"host": "192.0.2.24:8123"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ui_url"] == "http://192.0.2.24:5179/ui/index.html"
    assert payload["connect"]["ui"] == "http://192.0.2.24:5179/ui/index.html"


def test_a2a_agent_card_endpoint_describes_morpheus_interfaces():
    client = api_client()

    response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    assert "max-age" in response.headers["cache-control"]
    payload = response.json()
    assert payload["name"] == "Morpheus AI"
    assert payload["version"] == "0.1.0"
    assert payload["capabilities"]["streaming"] is False
    assert payload["defaultInputModes"] == ["application/json", "text/plain"]
    assert payload["defaultOutputModes"] == ["application/json", "text/markdown", "text/plain"]
    interfaces = {item["protocolBinding"]: item for item in payload["supportedInterfaces"]}
    assert interfaces["https://morpheus.ai/protocols/agent-connect/v1"]["url"] == (
        "http://testserver/agent/connect"
    )
    assert interfaces["MCP"]["url"] == "http://testserver/mcp"
    assert {skill["id"] for skill in payload["skills"]} >= {
        "compile-project-state",
        "build-agent-handoff",
        "inspect-integrations",
        "smoke-test-local-model",
    }


def test_mcp_endpoint_rejects_untrusted_origin():
    client = api_client(raise_server_exceptions=False)

    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        headers={"origin": "http://evil.example"},
    )

    assert response.status_code == 403


def test_mcp_initialize_returns_tools_capability():
    client = api_client(raise_server_exceptions=False)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == 1
    assert payload["result"]["protocolVersion"] == "2025-11-25"
    assert payload["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert payload["result"]["serverInfo"]["name"] == "morpheus"


def test_mcp_initialize_rejects_non_object_params():
    client = api_client(raise_server_exceptions=False)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": "bad-params",
            "method": "initialize",
            "params": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == "bad-params"
    assert payload["error"]["code"] == -32602
    assert payload["error"]["message"] == "MCP params must be a JSON object"


def test_mcp_tools_list_exposes_morpheus_tools():
    client = api_client(raise_server_exceptions=False)

    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "tools", "method": "tools/list"},
    )

    assert response.status_code == 200
    payload = response.json()
    tool_names = {tool["name"] for tool in payload["result"]["tools"]}
    assert tool_names >= {
        "morpheus_status",
        "morpheus_diagnostics",
        "morpheus_integrations",
        "morpheus_model_smoke",
    }


def test_mcp_tools_call_returns_integration_manifest():
    client = api_client(raise_server_exceptions=False)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": "call",
            "method": "tools/call",
            "params": {"name": "morpheus_integrations", "arguments": {}},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "call"
    assert payload["result"]["isError"] is False
    content = payload["result"]["content"]
    assert content[0]["type"] == "text"
    assert '"services"' in content[0]["text"]


def test_mcp_model_smoke_tool_uses_eval_query(monkeypatch):
    client = api_client(raise_server_exceptions=False)

    def fake_query_model(prompt, base_model="qwen2.5:7b", adapter_path=None, **kwargs):
        return f"{base_model}: {prompt}"

    import morpheus.training.eval as eval_module

    monkeypatch.setattr(eval_module, "query_model", fake_query_model)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": "smoke",
            "method": "tools/call",
            "params": {
                "name": "morpheus_model_smoke",
                "arguments": {"base_model": "qwen2.5:0.5b", "prompt": "ping"},
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["isError"] is False
    assert "qwen2.5:0.5b: ping" in payload["result"]["content"][0]["text"]


def test_agent_connect_guides_uninitialized_project(tmp_path):
    client = api_client(raise_server_exceptions=False)

    response = client.get("/agent/connect", params={"project_root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_base"] == "http://testserver"
    assert payload["project_root"] == str(tmp_path)
    assert payload["state"] == {
        "initialized": False,
        "compiled": False,
        "sources": 0,
        "claims": 0,
        "evidence": 0,
        "compiled_at": None,
    }
    assert payload["endpoints"]["initialize"]["method"] == "POST"
    assert payload["endpoints"]["initialize"]["json"] == {"project_root": str(tmp_path)}
    assert payload["endpoints"]["prepare_agent"]["method"] == "POST"
    assert payload["endpoints"]["prepare_agent"]["json"] == {"project_root": str(tmp_path)}
    config_url = urlparse(payload["endpoints"]["config"]["url"])
    assert config_url.path == "/config"
    assert parse_qs(config_url.query) == {"project_root": [str(tmp_path)]}
    assert payload["endpoints"]["integrations"]["method"] == "GET"
    assert urlparse(payload["endpoints"]["integrations"]["url"]).path == "/integrations"
    assert payload["integrations"]["service"] == "morpheus"
    assert {service["id"] for service in payload["integrations"]["services"]} >= {
        "github",
        "slack",
        "linear",
    }
    status_url = urlparse(payload["endpoints"]["status"]["url"])
    assert status_url.path == "/status"
    assert parse_qs(status_url.query) == {"project_root": [str(tmp_path)]}
    assert payload["sequence"][0]["id"] == "status"
    assert payload["sequence"][1]["id"] == "prepare_agent"
    assert payload["next_action"]["id"] == "prepare_agent"
    assert payload["next_action"]["command"] == "morpheus prepare-agent"
    assert payload["next_action"]["request"] == payload["endpoints"]["prepare_agent"]
    assert payload["cli"]["agent_connect"] == "morpheus agent-connect --json"
    assert (
        payload["cli"]["serve_ui"]
        == "morpheus serve --ui --host 0.0.0.0 --port 8000 --ui-port 5173"
    )
    assert "Fetch the connect manifest" in payload["agent_prompt"]


def test_integrations_endpoint_reports_services_for_agents(monkeypatch, tmp_path):
    morpheus_home = tmp_path / ".morpheus"
    morpheus_home.mkdir()
    (morpheus_home / "linear_cache.json").write_text("[]")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    client = api_client(raise_server_exceptions=False)

    response = client.get("/integrations")

    assert response.status_code == 200
    payload = response.json()
    services = {service["id"]: service for service in payload["services"]}
    assert payload["service"] == "morpheus"
    assert payload["home"] == str(morpheus_home)
    assert services["linear"]["status"] == "cache_ready"
    assert services["linear"]["setup_command"] == "morpheus integrate linear"
    assert services["slack"]["status"] == "not_configured"
    assert services["github"]["auth"] == "cache + PAT"
    assert services["github"]["cache_path"] == str(morpheus_home / "github_cache.json")


def test_model_smoke_endpoint_queries_configured_model(monkeypatch):
    client = api_client(raise_server_exceptions=False)
    calls = []

    def fake_query_model(prompt, base_model="qwen2.5:7b", adapter_path=None, **kwargs):
        calls.append(
            {
                "prompt": prompt,
                "base_model": base_model,
                "adapter_path": adapter_path,
                "kwargs": kwargs,
            }
        )
        return "model is alive"

    import morpheus.training.eval as eval_module

    monkeypatch.setattr(eval_module, "query_model", fake_query_model)

    response = client.post(
        "/models/smoke",
        json={"base_model": "qwen2.5:0.5b", "prompt": "ping"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "base_model": "qwen2.5:0.5b",
        "prompt": "ping",
        "answer": "model is alive",
        "error": None,
    }
    assert calls == [
        {
            "prompt": "ping",
            "base_model": "qwen2.5:0.5b",
            "adapter_path": None,
            "kwargs": {},
        }
    ]


def test_model_smoke_endpoint_reports_model_error(monkeypatch):
    client = api_client(raise_server_exceptions=False)

    def fake_query_model(prompt, base_model="qwen2.5:7b", adapter_path=None, **kwargs):
        return "Error: ollama executable not found"

    import morpheus.training.eval as eval_module

    monkeypatch.setattr(eval_module, "query_model", fake_query_model)

    response = client.post("/models/smoke", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["base_model"] == "qwen2.5:0.5b"
    assert payload["answer"] == ""
    assert payload["error"] == "Error: ollama executable not found"


def test_agent_connect_reports_compiled_project(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: agent self-connect smoke\n")
    client = api_client(raise_server_exceptions=False)
    compile_response = client.post("/compile", json={"project_root": str(tmp_path)})

    response = client.get("/agent/connect", params={"project_root": str(tmp_path)})

    assert compile_response.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"]["initialized"] is True
    assert payload["state"]["compiled"] is True
    assert payload["state"]["sources"] == 1
    assert payload["state"]["claims"] == 1
    assert payload["next_action"]["id"] == "prepare_agent"
    wake_url = urlparse(payload["endpoints"]["wake"]["url"])
    assert wake_url.path == "/wake"
    assert parse_qs(wake_url.query) == {"project_root": [str(tmp_path)]}
    assert "morpheus wake" in payload["cli"]["read_wake"]


def test_diagnostics_reports_project_setup_steps(tmp_path):
    client = api_client(raise_server_exceptions=False)

    response = client.get("/diagnostics", params={"project_root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_base"] == "http://testserver"
    assert payload["project_root"] == str(tmp_path)
    assert payload["state"]["initialized"] is False
    checks = {check["id"]: check for check in payload["checks"]}
    assert checks["backend"]["ok"] is True
    assert checks["project_root"]["ok"] is True
    assert checks["initialized"]["ok"] is False
    assert checks["compiled"]["ok"] is False
    assert checks["wake"]["ok"] is False
    assert checks["agent_bootstrap"]["ok"] is False
    assert checks["agent_bootstrap"]["detail"] == "Run Bootstrap AGENTS.md"
    assert payload["next_action"]["id"] == "prepare_agent"
    assert payload["next_action"]["label"] == "Prepare Agent"
    assert payload["next_action"]["command"] == "morpheus prepare-agent"
    assert payload["next_action"]["request"]["method"] == "POST"
    assert payload["next_action"]["request"]["url"] == "http://testserver/agent/prepare"
    assert payload["next_action"]["request"]["json"] == {"project_root": str(tmp_path)}
    assert payload["agent_connect_url"].endswith("/agent/connect?project_root=" + str(tmp_path).replace("/", "%2F"))
    assert payload["commands"]["agent_connect"] == "morpheus agent-connect --json"
    assert payload["commands"]["serve"] == "morpheus serve --host 0.0.0.0 --port 8000"
    assert (
        payload["commands"]["serve_ui"]
        == "morpheus serve --ui --host 0.0.0.0 --port 8000 --ui-port 5173"
    )


def test_diagnostics_recommends_setting_project_root_for_missing_path(tmp_path):
    missing_root = tmp_path / "missing"
    client = api_client(raise_server_exceptions=False)

    response = client.get("/diagnostics", params={"project_root": str(missing_root)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"]["initialized"] is False
    checks = {check["id"]: check for check in payload["checks"]}
    assert checks["project_root"]["ok"] is False
    assert payload["next_action"] == {
        "id": "set_project_root",
        "label": "Set Project Root",
        "detail": "Choose an existing safe project directory.",
        "command": None,
        "request": None,
    }


def test_agent_connect_recommends_setting_project_root_for_missing_path(tmp_path):
    missing_root = tmp_path / "missing"
    client = api_client(raise_server_exceptions=False)

    response = client.get("/agent/connect", params={"project_root": str(missing_root)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"]["initialized"] is False
    assert payload["next_action"]["id"] == "set_project_root"
    assert payload["next_action"]["request"] is None


def test_diagnostics_reports_current_agent_bootstrap(tmp_path):
    client = api_client(raise_server_exceptions=False)
    bootstrap_response = client.post("/agent/bootstrap", json={"project_root": str(tmp_path)})

    response = client.get("/diagnostics", params={"project_root": str(tmp_path)})

    assert bootstrap_response.status_code == 200
    assert response.status_code == 200
    checks = {check["id"]: check for check in response.json()["checks"]}
    assert checks["agent_bootstrap"]["ok"] is True
    assert checks["agent_bootstrap"]["detail"] == "AGENTS.md is current"


def test_project_config_reports_default_watch_dirs_without_initializing(tmp_path):
    client = api_client(raise_server_exceptions=False)

    response = client.get("/config", params={"project_root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["project_root"] == str(tmp_path)
    assert payload["initialized"] is False
    assert payload["watch_dirs"] == ["."]
    assert payload["watch_paths"] == [
        {
            "path": ".",
            "absolute_path": str(tmp_path),
            "exists": True,
            "kind": "directory",
            "valid": True,
            "detail": "directory",
        }
    ]


def test_project_config_updates_watch_dirs_and_compile_uses_multiple_paths(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "todo.md").write_text("TODO: source task\n")
    (tmp_path / "docs" / "decision.md").write_text("DECISION: document architecture\n")
    (tmp_path / "ignored.md").write_text("TODO: ignore root\n")
    client = api_client(raise_server_exceptions=False)

    config_response = client.post(
        "/config",
        json={"project_root": str(tmp_path), "watch_dirs": ["src", "docs"]},
    )
    compile_response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert config_response.status_code == 200
    assert compile_response.status_code == 200
    assert config_response.json()["initialized"] is True
    assert config_response.json()["watch_dirs"] == ["src", "docs"]
    assert 'watch_dirs = [ "src", "docs",]' in (
        tmp_path / ".morpheus" / "morpheus.toml"
    ).read_text()
    assert compile_response.json()["source_count"] == 2
    wake_md = compile_response.json()["wake_md"]
    assert "TODO: source task" in wake_md
    assert "DECISION: document architecture" in wake_md
    assert "TODO: ignore root" not in wake_md


def test_project_config_rejects_watch_dir_outside_project(tmp_path):
    outside = tmp_path.parent / "outside-project"
    client = api_client(raise_server_exceptions=False)

    response = client.post(
        "/config",
        json={"project_root": str(tmp_path), "watch_dirs": [str(outside)]},
    )

    assert response.status_code == 400
    assert "Watch path must stay inside project root" in response.json()["detail"]
    assert not (tmp_path / ".morpheus").exists()


def test_diagnostics_recommends_handoff_after_prepare_agent(tmp_path):
    (tmp_path / "README.md").write_text("TODO: ready for handoff\n")
    client = api_client(raise_server_exceptions=False)
    prepare_response = client.post("/agent/prepare", json={"project_root": str(tmp_path)})

    response = client.get("/diagnostics", params={"project_root": str(tmp_path)})

    assert prepare_response.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert payload["next_action"]["id"] == "handoff"
    assert payload["next_action"]["label"] == "Build Handoff"
    assert payload["next_action"]["command"] == "morpheus handoff"
    handoff_url = urlparse(payload["next_action"]["request"]["url"])
    assert payload["next_action"]["request"]["method"] == "GET"
    assert handoff_url.path == "/agent/handoff"
    assert parse_qs(handoff_url.query) == {"project_root": [str(tmp_path)]}


def test_agent_handoff_returns_bundle_for_uninitialized_project(tmp_path):
    client = api_client(raise_server_exceptions=False)

    response = client.get("/agent/handoff", params={"project_root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "morpheus"
    assert payload["project_root"] == str(tmp_path)
    assert payload["manifest"]["service"] == "morpheus"
    assert payload["diagnostics"]["project_root"] == str(tmp_path)
    assert payload["agent_bootstrap_preview"]["path"] == str(tmp_path / "AGENTS.md")
    assert "morpheus handoff" in payload["agent_bootstrap_preview"]["content"]
    assert "morpheus prepare-agent" in payload["agent_bootstrap_preview"]["content"]
    assert "morpheus agent-connect --json" in payload["agent_bootstrap_preview"]["content"]
    assert payload["wake_md"] is None
    assert payload["commands"]["handoff"] == "morpheus handoff"
    assert payload["commands"]["prepare_agent"] == "morpheus prepare-agent"
    assert "# Morpheus Agent Handoff" in payload["markdown"]
    assert "morpheus prepare-agent" in payload["markdown"]
    assert "morpheus bootstrap-agent --dry-run" in payload["markdown"]
    assert not (tmp_path / "AGENTS.md").exists()


def test_agent_handoff_markdown_endpoint_returns_plain_markdown(tmp_path):
    client = api_client(raise_server_exceptions=False)

    response = client.get("/agent/handoff.md", params={"project_root": str(tmp_path)})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text.startswith("# Morpheus Agent Handoff")
    assert "morpheus handoff" in response.text
    assert "morpheus bootstrap-agent --dry-run" in response.text
    assert not (tmp_path / "AGENTS.md").exists()


def test_agent_handoff_includes_wake_when_project_is_compiled(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: handoff includes wake\n")
    client = api_client(raise_server_exceptions=False)
    compile_response = client.post("/compile", json={"project_root": str(tmp_path)})

    response = client.get("/agent/handoff", params={"project_root": str(tmp_path)})

    assert compile_response.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert payload["wake_md"] is not None
    assert "TODO: handoff includes wake" in payload["wake_md"]
    assert "## WAKE.md" in payload["markdown"]


def test_agent_prepare_initializes_compiles_bootstraps_verifies_and_returns_handoff(tmp_path):
    (tmp_path / "README.md").write_text("TODO: prepare one click handoff\n")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/agent/prepare", json={"project_root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "morpheus"
    assert payload["project_root"] == str(tmp_path)
    assert [step["id"] for step in payload["steps"]] == [
        "initialize",
        "compile",
        "bootstrap_agent",
        "verify",
        "handoff",
    ]
    assert all(step["ok"] for step in payload["steps"])
    assert payload["initialized"]["initialized"] is True
    assert payload["compiled"]["source_count"] == 1
    assert payload["verified"]["valid"] is True
    assert "morpheus handoff" in payload["bootstrapped"]["content"]
    assert "morpheus prepare-agent" in payload["bootstrapped"]["content"]
    assert payload["handoff"]["wake_md"] is not None
    assert "TODO: prepare one click handoff" in payload["handoff"]["wake_md"]
    assert (tmp_path / ".morpheus" / "WAKE.md").exists()
    assert (tmp_path / "AGENTS.md").exists()


def test_agent_prepare_rejects_symlinked_agents_file_without_writing_target(tmp_path):
    (tmp_path / "README.md").write_text("TODO: prepare rejects symlink\n")
    outside = tmp_path / "outside.md"
    outside.write_text("do not overwrite\n")
    try:
        (tmp_path / "AGENTS.md").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/agent/prepare", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "AGENTS.md must not be a symlink" in response.json()["detail"]
    assert outside.read_text() == "do not overwrite\n"


def test_agent_prepare_writes_stable_local_bootstrap_when_requested_through_lan(tmp_path):
    (tmp_path / "README.md").write_text("TODO: prepare over lan\n")
    client = api_client(raise_server_exceptions=False)

    response = client.post(
        "/agent/prepare",
        json={"project_root": str(tmp_path)},
        headers={"host": "192.168.1.54:8000"},
    )

    assert response.status_code == 200
    payload = response.json()
    content = (tmp_path / "AGENTS.md").read_text()
    assert payload["bootstrapped"]["agent_connect_url"].startswith(
        "http://192.168.1.54:8000/agent/connect"
    )
    assert "http://127.0.0.1:8000/agent/connect" in content
    assert "http://192.168.1.54:8000/agent/connect" not in content


def test_diagnostics_agent_bootstrap_is_not_tied_to_request_host(tmp_path):
    agents_path = tmp_path / "AGENTS.md"
    agents_path.write_text(
        "# AGENTS.md\n\n"
        "<!-- MORPHEUS:BEGIN -->\n"
        "## Morpheus Bootstrap\n\n"
        "Fetch the Morpheus manifest before making changes:\n\n"
        "- Connect manifest: `http://127.0.0.1:8000/agent/connect?project_root=x`\n"
        "- One-command prepare: `morpheus prepare-agent`.\n"
        "- Local handoff bundle: `morpheus handoff`.\n"
        "- Local CLI manifest: `morpheus agent-connect --json`.\n"
        "- Local diagnostics: `morpheus diagnostics --json`.\n"
        "- Read `WAKE.md` before edits.\n"
        "- Run compile and verify after meaningful changes.\n"
        "- If the API/UI are unavailable, start them with "
        "`morpheus serve --ui --host 0.0.0.0 --port 8000 --ui-port 5173`.\n"
        "<!-- MORPHEUS:END -->\n"
    )
    client = api_client(raise_server_exceptions=False)

    response = client.get(
        "/diagnostics",
        params={"project_root": str(tmp_path)},
        headers={"host": "192.168.1.54:8000"},
    )

    assert response.status_code == 200
    checks = {check["id"]: check for check in response.json()["checks"]}
    assert checks["agent_bootstrap"]["ok"] is True
    assert checks["agent_bootstrap"]["detail"] == "AGENTS.md is current"


def test_agent_bootstrap_creates_agents_md_without_initializing_project(tmp_path):
    client = api_client(raise_server_exceptions=False)

    response = client.post("/agent/bootstrap", json={"project_root": str(tmp_path)})

    agents_path = tmp_path / "AGENTS.md"
    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == str(agents_path)
    assert payload["created"] is True
    assert payload["updated"] is True
    assert payload["project_root"] == str(tmp_path)
    assert agents_path.exists()
    content = agents_path.read_text()
    assert "<!-- MORPHEUS:BEGIN -->" in content
    assert "Fetch the Morpheus manifest before making changes" in content
    assert "morpheus handoff" in content
    assert "http://127.0.0.1:8000/agent/connect" in content
    assert not (tmp_path / ".morpheus").exists()


def test_agent_bootstrap_preview_does_not_write_agents_md(tmp_path):
    client = api_client(raise_server_exceptions=False)

    response = client.post("/agent/bootstrap/preview", json={"project_root": str(tmp_path)})

    agents_path = tmp_path / "AGENTS.md"
    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == str(agents_path)
    assert payload["created"] is True
    assert payload["updated"] is True
    assert payload["project_root"] == str(tmp_path)
    assert "<!-- MORPHEUS:BEGIN -->" in payload["content"]
    assert "morpheus handoff" in payload["content"]
    assert "morpheus agent-connect --json" in payload["content"]
    assert not agents_path.exists()


def test_agent_bootstrap_preview_preserves_existing_file_without_rewriting(tmp_path):
    agents_path = tmp_path / "AGENTS.md"
    original = "# Existing Agent Notes\n\nKeep this rule.\n"
    agents_path.write_text(original)
    client = api_client(raise_server_exceptions=False)

    response = client.post("/agent/bootstrap/preview", json={"project_root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["created"] is False
    assert payload["updated"] is True
    assert "Keep this rule." in payload["content"]
    assert "<!-- MORPHEUS:BEGIN -->" in payload["content"]
    assert agents_path.read_text() == original


def test_agent_bootstrap_replaces_managed_section_and_preserves_existing_content(tmp_path):
    agents_path = tmp_path / "AGENTS.md"
    agents_path.write_text(
        "# Existing Agent Notes\n\n"
        "Keep this project-specific rule.\n\n"
        "<!-- MORPHEUS:BEGIN -->\nold managed text\n<!-- MORPHEUS:END -->\n"
    )
    client = api_client(raise_server_exceptions=False)

    response = client.post("/agent/bootstrap", json={"project_root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["created"] is False
    content = agents_path.read_text()
    assert "Keep this project-specific rule." in content
    assert "old managed text" not in content
    assert content.count("<!-- MORPHEUS:BEGIN -->") == 1
    assert payload["content"] == content


def test_agent_bootstrap_rejects_symlinked_agents_file(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("do not overwrite\n")
    try:
        (tmp_path / "AGENTS.md").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/agent/bootstrap", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "AGENTS.md must not be a symlink" in response.json()["detail"]
    assert outside.read_text() == "do not overwrite\n"


def test_init_creates_project_state_for_ui(tmp_path):
    client = api_client(raise_server_exceptions=False)

    response = client.post("/init", json={"project_root": str(tmp_path)})

    assert response.status_code == 200
    assert response.json() == {
        "initialized": True,
        "project_root": str(tmp_path),
        "created": True,
    }
    assert (tmp_path / ".morpheus" / "morpheus.toml").exists()
    assert (tmp_path / ".morpheus" / "keys" / "local.key").exists()
    assert (tmp_path / ".morpheus" / "keys" / "local.pub").exists()


def test_init_reports_existing_project_without_reinitializing(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    key_path = tmp_path / ".morpheus" / "keys" / "local.key"
    original_key = key_path.read_bytes()
    client = api_client(raise_server_exceptions=False)

    response = client.post("/init", json={"project_root": str(tmp_path)})

    assert response.status_code == 200
    assert response.json() == {
        "initialized": True,
        "project_root": str(tmp_path),
        "created": False,
    }
    assert key_path.read_bytes() == original_key


def test_init_rejects_symlinked_project_root_parent(tmp_path):
    outside_parent = tmp_path / "outside-parent"
    outside_project = outside_parent / "project"
    outside_project.mkdir(parents=True)
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(outside_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    client = api_client(raise_server_exceptions=False)

    response = client.post(
        "/init",
        json={"project_root": str(linked_parent / "project")},
    )

    assert response.status_code == 400
    assert "Project root must not contain a symlink" in response.json()["detail"]
    assert not (outside_project / ".morpheus").exists()


def test_status_reports_initialized_project_before_first_compile(tmp_path):
    client = api_client(raise_server_exceptions=False)
    init_response = client.post("/init", json={"project_root": str(tmp_path)})

    response = client.get("/status", params={"project_root": str(tmp_path)})

    assert init_response.status_code == 200
    assert response.status_code == 200
    assert response.json() == {
        "initialized": True,
        "compiled": False,
        "sources": 0,
        "claims": 0,
        "evidence": 0,
        "compiled_at": None,
    }


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


def test_verify_accepts_project_root_json_body(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: verify API body contract\n")
    client = api_client()

    compile_response = client.post("/compile", json={"project_root": str(tmp_path)})
    verify_response = client.post("/verify", json={"project_root": str(tmp_path)})

    assert compile_response.status_code == 200
    assert verify_response.status_code == 200
    assert verify_response.json()["valid"] is True
    assert verify_response.json()["receipt_id"] == compile_response.json()["receipt_id"]


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


def test_verify_returns_bad_request_for_morpheus_symlink(tmp_path):
    outside = tmp_path / "outside-morpheus"
    outside.mkdir()
    (tmp_path / ".morpheus").symlink_to(outside, target_is_directory=True)
    client = api_client(raise_server_exceptions=False)

    response = client.post("/verify", params={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Not initialized" in response.json()["detail"]


def test_verify_returns_invalid_response_for_receipts_path_file(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    receipts_dir = tmp_path / ".morpheus" / "receipts"
    receipts_dir.rmdir()
    receipts_dir.write_text("not a directory")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/verify", params={"project_root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    assert payload["receipt_id"] == "none"
    assert payload["errors"] == ["receipts path is not a directory"]


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


def test_compile_returns_bad_request_for_morpheus_symlink(tmp_path):
    (tmp_path / "README.md").write_text("TODO: compile with symlinked state\n")
    outside = tmp_path / "outside-morpheus"
    outside.mkdir()
    (tmp_path / ".morpheus").symlink_to(outside, target_is_directory=True)
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Not initialized" in response.json()["detail"]
    assert "Signing failed" not in response.json()["detail"]


def test_compile_rejects_symlinked_project_root_without_writing_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    MorpheusConfig(project_root=target).init_default()
    (target / "README.md").write_text("TODO: do not compile through symlinked root\n")
    project_root = tmp_path / "linked-project"
    try:
        project_root.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(project_root)})

    assert response.status_code == 400
    assert "Not initialized" in response.json()["detail"]
    assert not list((target / ".morpheus" / "receipts").glob("receipt_*.json"))


def test_compile_returns_bad_request_for_receipts_path_file(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: compile with invalid receipts path\n")
    receipts_dir = tmp_path / ".morpheus" / "receipts"
    receipts_dir.rmdir()
    receipts_dir.write_text("not a directory")
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Receipt chain invalid" in response.json()["detail"]
    assert "receipts path is not a directory" in response.json()["detail"]
    assert "Signing failed" not in response.json()["detail"]
    assert "Output write failed" not in response.json()["detail"]


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


def test_compile_returns_bad_request_for_symlinked_output_artifact(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: compile API with symlinked output\n")
    external_output = tmp_path / "external-output"
    external_output.write_text("do not modify")
    (tmp_path / ".morpheus" / "WAKE.md").symlink_to(external_output)
    client = api_client(raise_server_exceptions=False)

    response = client.post("/compile", json={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "Output write failed" in response.json()["detail"]
    assert "must not be a symlink" in response.json()["detail"]
    assert external_output.read_text() == "do not modify"
    assert not list((tmp_path / ".morpheus" / "receipts").glob("receipt_*.json"))


def test_get_wake_rejects_project_path_traversal(tmp_path, monkeypatch):
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    (tmp_path / "WAKE.md").write_text("secret parent wake")
    monkeypatch.chdir(safe_dir)
    client = api_client()

    response = client.get("/wake/%2E%2E")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid project name"


def test_get_wake_rejects_symlinked_project_escape(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "WAKE.md").write_text("secret outside wake")
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    client = api_client(raise_server_exceptions=False)

    response = client.get("/wake/linked")

    assert response.status_code == 404
    assert "secret outside wake" not in response.text


def test_get_wake_rejects_symlinked_home_morpheus_escape(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    outside_morpheus = tmp_path / "outside-morpheus"
    project_dir = outside_morpheus / "project"
    project_dir.mkdir(parents=True)
    (project_dir / "WAKE.md").write_text("secret home wake")
    try:
        (home / ".morpheus").symlink_to(outside_morpheus, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    safe_cwd = tmp_path / "safe"
    safe_cwd.mkdir()
    monkeypatch.chdir(safe_cwd)
    monkeypatch.setattr(Path, "home", lambda: home)
    client = api_client(raise_server_exceptions=False)

    response = client.get("/wake/project")

    assert response.status_code == 404
    assert "secret home wake" not in response.text


def test_get_wake_rejects_symlinked_wake_file_escape(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret_wake = outside / "WAKE.md"
    secret_wake.write_text("secret symlink wake")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "WAKE.md").symlink_to(secret_wake)
    monkeypatch.chdir(tmp_path)
    client = api_client(raise_server_exceptions=False)

    response = client.get("/wake/project")

    assert response.status_code == 404
    assert "secret symlink wake" not in response.text


def test_get_wake_by_project_root_returns_compiled_wake(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    (morpheus_dir / "WAKE.md").write_text("# WAKE\n\nCompiled project state\n")
    client = api_client(raise_server_exceptions=False)

    response = client.get("/wake", params={"project_root": str(tmp_path)})

    assert response.status_code == 200
    assert response.json() == {
        "project_root": str(tmp_path),
        "wake_md": "# WAKE\n\nCompiled project state\n",
    }


def test_get_wake_by_project_root_rejects_symlinked_wake_file(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    external_wake = tmp_path / "external-WAKE.md"
    external_wake.write_text("secret wake")
    (morpheus_dir / "WAKE.md").symlink_to(external_wake)
    client = api_client(raise_server_exceptions=False)

    response = client.get("/wake", params={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "WAKE.md unreadable" in response.json()["detail"]
    assert "secret wake" not in response.text


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


def test_status_returns_bad_request_for_symlinked_state_file(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    external_state = tmp_path / "external-state.json"
    external_state.write_text(
        json.dumps(
            {
                "sources": [],
                "claims": [],
                "evidence": [],
                "compiled_at": "2026-05-13T00:00:00Z",
            }
        )
    )
    (morpheus_dir / "state.json").symlink_to(external_state)
    client = api_client(raise_server_exceptions=False)

    response = client.get("/status", params={"project_root": str(tmp_path)})

    assert response.status_code == 400
    assert "State file invalid" in response.json()["detail"]
    assert "must not be a symlink" in response.json()["detail"]


def test_status_treats_morpheus_symlink_as_uninitialized(tmp_path):
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
    (tmp_path / ".morpheus").symlink_to(outside, target_is_directory=True)
    client = api_client(raise_server_exceptions=False)

    response = client.get("/status", params={"project_root": str(tmp_path)})

    assert response.status_code == 200
    assert response.json() == {"initialized": False}


def test_status_treats_symlinked_project_root_as_uninitialized(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    morpheus_dir = target / ".morpheus"
    morpheus_dir.mkdir()
    (morpheus_dir / "state.json").write_text(
        json.dumps(
            {
                "sources": [],
                "claims": [],
                "evidence": [],
                "compiled_at": "2026-05-13T00:00:00Z",
            }
        )
    )
    project_root = tmp_path / "linked-project"
    try:
        project_root.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    client = api_client(raise_server_exceptions=False)

    response = client.get("/status", params={"project_root": str(project_root)})

    assert response.status_code == 200
    assert response.json() == {"initialized": False}


def test_status_treats_symlinked_project_root_parent_as_uninitialized(tmp_path):
    outside_parent = tmp_path / "outside-parent"
    target = outside_parent / "target"
    morpheus_dir = target / ".morpheus"
    morpheus_dir.mkdir(parents=True)
    (morpheus_dir / "state.json").write_text(
        json.dumps(
            {
                "sources": [],
                "claims": [],
                "evidence": [],
                "compiled_at": "2026-05-13T00:00:00Z",
            }
        )
    )
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(outside_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    client = api_client(raise_server_exceptions=False)

    response = client.get("/status", params={"project_root": str(linked_parent / "target")})

    assert response.status_code == 200
    assert response.json() == {"initialized": False}


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
