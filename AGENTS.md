# AGENTS.md

<!-- MORPHEUS:BEGIN -->
## Morpheus Bootstrap

Prepare Morpheus before making changes:

- One-command prepare: `morpheus prepare-agent`.
- One-command WAKE refresh: `morpheus wake .`.
- Local handoff bundle: `morpheus handoff`.
- Local CLI manifest: `morpheus agent-connect --json`.
- Local diagnostics: `morpheus diagnostics --json`.
- HTTP manifest: start the API/UI, then fetch `/agent/connect?project_root=<absolute-project-root>`.
- Read root `WAKE.md` before edits.
- Run compile and verify after meaningful changes.
- If the API/UI are unavailable, start them with `morpheus serve --ui --host 0.0.0.0 --port 8000 --ui-port 5173`.

Agent sequence:

1. Fetch `/agent/connect` for this project root.
2. Initialize only when `state.initialized` is false.
3. Compile, then read root WAKE.md.
4. Make the requested project change.
5. Compile again and run `morpheus verify --all`.
<!-- MORPHEUS:END -->
