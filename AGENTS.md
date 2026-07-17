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

## Morpheus Roadmap Autopilot Rule

When asked to continue Morpheus roadmap work:

- Do one slice only.
- Prefer tests first.
- Treat truth-layer work as the data-quality gate, not as a replacement for weights-as-memory.
- Treat adapter weights as the final memory artifact once dataset quality, eval, activation, and rollback gates exist.
- Use this roadmap unless the user explicitly changes it:
  - v0.3: semantic classifier as product core.
  - v0.4: dataset quality dashboard.
  - v0.5: adapter memory benchmark.
  - v0.6: agent memory routing.
  - v0.7: team learning loop.
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

## Source-Backed Research Rule

When researching or changing Morpheus:

- Treat a claim as source-backed only when it is grounded in a repo path/line, command output, test result, receipt, or cited primary external source.
- If a statement is an inference, label it as an inference.
- Do not invent metrics, release status, model quality, benchmark results, or package behavior.
- Before changing product claims, inspect `README.md`, `WAKE.md`, `SPEC.md`, `docs/ROADMAP.md`, and the relevant code/tests.
- Use primary sources for external technical facts and cite them when user-facing.
- Keep cloud providers opt-in and never send project files to cloud by default.
- Never train on raw Markdown, rejected candidates, pending candidates, stale claims, or inferred-only candidates.
