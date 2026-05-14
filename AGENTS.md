# AGENTS.md

<!-- MORPHEUS:BEGIN -->
## Morpheus Bootstrap

Fetch the Morpheus manifest before making changes:

- Connect manifest: `http://127.0.0.1:8000/agent/connect?project_root=%2FUsers%2Ftestbot%2F.openclaw%2Fworkspace%2Fmorpheus-ai`
- One-command prepare: `morpheus prepare-agent`.
- Local handoff bundle: `morpheus handoff`.
- Local CLI manifest: `morpheus agent-connect --json`.
- Local diagnostics: `morpheus diagnostics --json`.
- Read `WAKE.md` before edits.
- Run compile and verify after meaningful changes.
- If the API/UI are unavailable, start them with `morpheus serve --ui --host 0.0.0.0 --port 8000 --ui-port 5173`.

Agent sequence:

1. Fetch `/agent/connect` for this project root.
2. Initialize only when `state.initialized` is false.
3. Compile, then read WAKE.md.
4. Make the requested project change.
5. Compile again and run `morpheus verify --all`.
<!-- MORPHEUS:END -->
