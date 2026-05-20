#!/usr/bin/env python3
"""Live smoke test for local Morpheus MCP truth tools."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode())


def tool_call(base_url: str, name: str, arguments: dict, request_id: str) -> dict:
    payload = post_json(
        f"{base_url}/mcp",
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    if "error" in payload:
        raise RuntimeError(f"{name} failed: {payload['error']}")
    content = payload["result"]["content"][0]["text"]
    return json.loads(content)


def wait_for_health(base_url: str) -> dict:
    last_error: Exception | None = None
    for _ in range(50):
        try:
            return get_json(f"{base_url}/health")
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"server did not become healthy: {last_error}")


def run_smoke(project_root: Path, *, port: int) -> dict:
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "morpheus",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        health = wait_for_health(base_url)
        tools_payload = post_json(
            f"{base_url}/mcp",
            {"jsonrpc": "2.0", "id": "tools", "method": "tools/list"},
        )
        tool_names = [tool["name"] for tool in tools_payload["result"]["tools"]]
        required = {
            "morpheus_check_text",
            "morpheus_get_active_state",
            "morpheus_get_evidence_for_claim",
            "morpheus_get_wake",
        }
        missing = sorted(required - set(tool_names))
        if missing:
            raise RuntimeError(f"missing MCP truth tools: {missing}")

        root_arg = str(project_root)
        check = tool_call(
            base_url,
            "morpheus_check_text",
            {
                "project_root": root_arg,
                "text": "Morpheus is mainly a personal AI agent.",
            },
            "check",
        )
        active_state = tool_call(
            base_url,
            "morpheus_get_active_state",
            {"project_root": root_arg},
            "state",
        )
        evidence = tool_call(
            base_url,
            "morpheus_get_evidence_for_claim",
            {
                "project_root": root_arg,
                "claim": "Morpheus checks coding-agent claims against local source-backed state",
            },
            "evidence",
        )
        wake = tool_call(
            base_url,
            "morpheus_get_wake",
            {"project_root": root_arg},
            "wake",
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    check_statuses = [item.get("status") for item in check.get("results", [])]
    if "stale" not in check_statuses:
        raise RuntimeError(f"check tool did not report stale claim: {check_statuses}")
    if int(active_state.get("claims_count") or 0) <= 0:
        raise RuntimeError("active state returned no claims")
    if int(evidence.get("match_count") or 0) <= 0:
        raise RuntimeError("evidence lookup returned no matches")
    if "## Current State" not in str(wake.get("wake_md") or ""):
        raise RuntimeError("WAKE payload missing Current State")

    return {
        "verdict": "MCP_TRUTH_TOOLS_SMOKE_PASS",
        "base_url": base_url,
        "health": health,
        "tool_names": tool_names,
        "check_statuses": check_statuses,
        "active_claims_count": active_state.get("claims_count"),
        "evidence_match_count": evidence.get("match_count"),
        "wake_path": wake.get("path"),
        "wake_has_current_state": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", nargs="?", default=".")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    project_root = Path(args.project).expanduser().resolve()
    result = run_smoke(project_root, port=args.port)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
