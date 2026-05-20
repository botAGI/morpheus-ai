# MCP Truth Tools Smoke - 2026-05-20

## Command

Run:

```bash
.venv/bin/python scripts/mcp_truth_tools_smoke.py . --port 8765
```

The script starts a local API and calls `/mcp` with JSON-RPC `tools/list` and
`tools/call`.

## Result

- Health: `{"status": "ok", "version": "0.2.0b1"}`
- Tools listed:
  - `morpheus_status`
  - `morpheus_diagnostics`
  - `morpheus_integrations`
  - `morpheus_model_smoke`
  - `morpheus_check_text`
  - `morpheus_get_active_state`
  - `morpheus_get_evidence_for_claim`
  - `morpheus_get_wake`
- `morpheus_check_text` classified the known outdated personal-agent claim as
  `stale`.
- `morpheus_get_active_state` returned `496` active claims.
- `morpheus_get_evidence_for_claim` returned one evidence match for a current
  Morpheus truth-layer claim.
- `morpheus_get_wake` returned `.morpheus/WAKE.md` and included
  `## Current State`.

## Verdict

`MCP_TRUTH_TOOLS_SMOKE_PASS`

The smoke was local-only on `127.0.0.1`. No cloud provider was configured or
called.
