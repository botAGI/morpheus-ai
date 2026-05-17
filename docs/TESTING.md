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
.venv/bin/morpheus compile
.venv/bin/morpheus verify --all
.venv/bin/morpheus diagnostics --json
.venv/bin/morpheus agent-connect --json
```

Expected result:

- compile writes fresh local `.morpheus/` artifacts,
- verify reports a valid receipt chain,
- diagnostics returns a clear `next_action`,
- agent-connect returns a machine-readable handoff manifest.

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
