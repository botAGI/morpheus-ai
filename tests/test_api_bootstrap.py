"""
Focused tests for API bootstrap helpers that do not require FastAPI to be installed.
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


class FakeFastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def add_middleware(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return lambda func: func

    def post(self, *args, **kwargs):
        return lambda func: func


class FakeHTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def import_server_without_fastapi(monkeypatch):
    fake_fastapi = ModuleType("fastapi")
    fake_fastapi.Body = lambda default=None: default
    fake_fastapi.FastAPI = FakeFastAPI
    fake_fastapi.HTTPException = FakeHTTPException
    fake_fastapi.Request = object

    fake_cors = ModuleType("fastapi.middleware.cors")
    fake_cors.CORSMiddleware = object

    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.middleware", ModuleType("fastapi.middleware"))
    monkeypatch.setitem(sys.modules, "fastapi.middleware.cors", fake_cors)

    module_name = "_morpheus_api_server_bootstrap_test"
    server_path = Path(__file__).parents[1] / "morpheus" / "api" / "server.py"
    spec = importlib.util.spec_from_file_location(module_name, server_path)
    server = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, server)
    spec.loader.exec_module(server)

    return server


def test_agent_bootstrap_reports_no_update_when_content_is_current(tmp_path, monkeypatch):
    server = import_server_without_fastapi(monkeypatch)
    request = SimpleNamespace(base_url="http://testserver/")
    first_response = server.write_agent_bootstrap(request, tmp_path)
    agents_path = tmp_path / "AGENTS.md"
    original_content = agents_path.read_text()

    response = server.write_agent_bootstrap(request, tmp_path)

    assert first_response.created is True
    assert first_response.updated is True
    assert response.created is False
    assert response.updated is False
    assert response.content == original_content
    assert agents_path.read_text() == original_content
