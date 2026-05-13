# Repo Hygiene And Packaging Gate Design

## Context

Morpheus AI is a Python package with CLI, provenance core, integrations, tests, and a Tauri UI skeleton. The last completed development increment improved the filesystem integration and added tests. The current baseline test command is:

```bash
python3 -m pytest tests/ -q
```

The baseline passes, but `git status` is polluted by Python bytecode files. Several `__pycache__` files are tracked, and new pytest runs create additional untracked cache files. This makes every future change harder to review and increases the risk of committing generated artifacts.

## Goal

Create a small repository hygiene gate so future development starts from a clean, reproducible working tree after running tests.

## Scope

This increment covers:

- Add a project-level `.gitignore`.
- Ignore Python cache, test cache, virtual environments, build outputs, local Morpheus runtime state, training outputs, Tauri/Rust build artifacts, editor files, and local logs.
- Remove already tracked `__pycache__` and Python bytecode files from the git index without deleting source files.
- Verify the test suite still passes.
- Verify a fresh pytest run no longer leaves cache noise in `git status`.

This increment does not change application behavior, CLI behavior, provenance format, integrations, API routes, or UI code.

## Design

The repository gets one root `.gitignore` with sections for Python, packaging, Morpheus-generated local state, Rust/Tauri build artifacts, logs, and editor/system files. The rules intentionally avoid broad source-like patterns. They ignore generated local artifacts only.

Tracked bytecode is removed using `git rm --cached` against files currently tracked by git. This changes the index while preserving working files on disk. Once `.gitignore` is present, future test runs may recreate bytecode locally, but git will ignore it.

## Validation

Validation is command-driven because this is repository configuration, not runtime behavior:

```bash
python3 -m pytest tests/ -q
git status --short
git ls-files | rg '(^|/)__pycache__/|\.pyc$|\.pyo$'
```

Expected results:

- Pytest reports all tests passing.
- `git status --short` shows the intended source-control changes, not regenerated cache files.
- `git ls-files` returns no tracked Python bytecode or `__pycache__` entries.

## Risks

The main risk is ignoring a file that should be committed. To avoid that, ignore rules stay focused on generated artifacts, local state, virtual environments, and build outputs. Project documentation, source files, lockfiles, and tests remain trackable.
