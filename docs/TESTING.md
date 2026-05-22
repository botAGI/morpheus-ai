# Testing And Release Quality Gate

This checklist keeps Morpheus safe for public GitHub work and useful for both
humans and agents.

## Fast Local Gate

Run before every commit:

```bash
.venv/bin/ruff check .
.venv/bin/pytest tests/ -q
```

## Morpheus Gate

Run after meaningful source, docs, API, CLI, or UI changes:

```bash
.venv/bin/morpheus stale .
.venv/bin/morpheus compile
.venv/bin/morpheus verify --all
.venv/bin/morpheus diagnostics --json
.venv/bin/morpheus agent-connect --json
```

Expected result:

- stale reports no stale launch-positioning claims,
- compile writes fresh local `.morpheus/` artifacts,
- verify reports a valid receipt chain,
- diagnostics returns a clear `next_action`,
- agent-connect returns a machine-readable handoff manifest.

## One-Command Wake Gate

Run in a disposable copy or when you intentionally want to refresh root
`WAKE.md`:

```bash
.venv/bin/morpheus wake .
```

For private workspaces, use:

```bash
.venv/bin/morpheus wake . --private
```

Expected result:

- public mode writes root `WAKE.md`,
- private mode keeps `WAKE.md` under `.morpheus/`,
- both modes compile and verify before printing an agent handoff prompt.

## Public Repository Hygiene

Run before pushing to a public repository:

```bash
git status --short
git ls-files | rg '(^|/)__pycache__/|\.pyc$|\.pyo$|^AGENT_[0-9]|^SOUL.md$|^IDENTITY.md$|^MEMORY.md$|^memory/'
rg -n '/Users/|password|token|secret|credential|private key' \
  -g '!tests/**' -g '!SECURITY.md' -g '!README*.md' .
```

Expected result:

- no generated Python bytecode is tracked,
- no old local agent task files are tracked,
- no personal memory files are tracked,
- no local runtime `.morpheus/` artifacts are tracked,
- no real credentials or private hostnames are present.

## UI Smoke

Run the UI:

```bash
.venv/bin/morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173
```

Open `http://127.0.0.1:5173/ui/index.html` and check:

- Start screen renders without console errors,
- Project root can be set,
- Diagnostics returns backend, project, WAKE, receipt, and agent bootstrap state,
- Prepare Agent creates a handoff,
- WAKE.md can be loaded and copied,
- MCP tools probe returns tools,
- mobile-width layout has no clipped primary controls.

## Package Gate

Run before tags and releases:

```bash
make verify
make build
```

Expected result:

- lint passes,
- all tests pass,
- source distribution and wheel build,
- `twine check dist/*` passes.

## Published Package Smoke

Run after a beta publish:

```bash
uvx --from 'morpheus-wake==0.2.0b1' morpheus --version
uvx --from 'morpheus-wake==0.2.0b1' morpheus wake . --private
uvx --from 'morpheus-wake==0.2.0b1' morpheus verify --all
```

Expected result:

- version output is `Morpheus AI v0.2.0b1`,
- private wake compiles local project state,
- receipt verification passes.

## ML Core Live Gate

Run before claiming the learning core improved the project adapter:

```bash
.venv/bin/morpheus learn lab . --dogfood --backend mlx --eval-limit 0 --repeat 2
.venv/bin/morpheus learn status
.venv/bin/morpheus learn train . --dry-run
```

Expected result:

- the lab uses strict source-backed dogfood candidates,
- the dataset has at least 20 accepted candidates and 100 examples,
- base and adapter are both evaluated on the full eval set,
- adapter pass rate meets the configured threshold,
- hallucination rate stays under threshold,
- regression count is zero for a production-ready lab verdict,
- dry-run training selects the latest trainable dataset and creates run
  artifacts,
- no adapter is activated automatically.

## MCP Truth Tools Live Smoke

Run before claiming MCP truth-layer readiness:

```bash
.venv/bin/python scripts/mcp_truth_tools_smoke.py . --port 8765
```

The script starts a local API server and calls `/mcp` with JSON-RPC:

- `tools/list`,
- `tools/call` for `morpheus_check_text`,
- `tools/call` for `morpheus_get_active_state`,
- `tools/call` for `morpheus_get_evidence_for_claim`,
- `tools/call` for `morpheus_get_wake`.

Expected result:

- truth tools are listed,
- stale project claims are classified as `stale`,
- active state returns local claims,
- evidence lookup returns source spans,
- WAKE fetch returns `.morpheus/WAKE.md`,
- API binds to `127.0.0.1` unless explicitly testing a trusted LAN setup.
