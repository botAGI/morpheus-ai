# AGENTS.md

<!-- MORPHEUS:BEGIN -->
## Morpheus Bootstrap

Fetch the Morpheus manifest before making changes:

- Connect manifest: start the API/UI, then fetch `/agent/connect?project_root=<PROJECT_ROOT>`.
- One-command prepare: `morpheus prepare-agent`.
- Local handoff bundle: `morpheus handoff`.
- Local CLI manifest: `morpheus agent-connect --json`.
- Local diagnostics: `morpheus diagnostics --json`.
- Read `WAKE.md` before edits.
- Run compile and verify after meaningful changes.
- If the API/UI are unavailable, start them with `morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173`.
- Use `0.0.0.0` only for explicit user-approved trusted LAN testing or authenticated proxy environments.

Agent sequence:

1. Fetch `/agent/connect` for this project root.
2. Initialize only when `state.initialized` is false.
3. Compile, then read WAKE.md.
4. Make the requested project change.
5. Compile again and run `morpheus verify --all`.
<!-- MORPHEUS:END -->

## Morpheus v0.2 Autopilot Rule

When asked to continue v0.2 work:

- Do one slice only.
- Prefer tests first.
- Treat truth-layer work as the data-quality gate, not as a replacement for weights-as-memory.
- Treat adapter weights as the final memory artifact once dataset quality, eval, activation, and rollback gates exist.
- Use this roadmap unless the user explicitly changes it:
  - v0.2: semantic/review/check plus dataset compiler.
  - v0.3: training backend, eval, adapter registry, activation, rollback.
  - v0.4: nightly learning loop.
- No accepted source span means no training example.
- No eval pass means no adapter activation.
- No rollback means no production activation.
- Keep `morpheus wake .` deterministic by default.
- Keep cloud providers opt-in.
- Never make semantic candidates active without review.
- Never tag, release, publish, or push unless explicitly instructed.
- Always run:
  - `ruff check .`
  - `pytest tests/ -q`
  - `morpheus wake . --private`
  - `morpheus verify --all`
- If touching `check`, also test file input, stdin input, JSON output, and exit codes.
