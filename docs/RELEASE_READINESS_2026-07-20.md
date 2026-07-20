# Morpheus v0.2.0b2 Release Design And Readiness

## Release Decision

The next package version is `0.2.0b2` and the Git tag is `v0.2.0b2`.

This is a beta update, not a stable-release claim. The existing beta exit rule
still requires broader real-repository and no-regression evidence before a
stable `v0.2.0` release. Roadmap labels v0.3 through v0.7 describe product
milestones and do not determine the Python package version.

The release contains the 17 commits after `v0.2.0b1`, including the semantic
quality and benchmark work, guarded adapter activation and rollback authority,
canonical memory routing, signed active-state review authority, and the unified
six-source reviewed team-learning input path.

## Scope

The release change updates:

- Python package metadata and the public `morpheus.__version__` value;
- runtime API, manifest, provenance, and generated-WAKE version surfaces;
- CLI/API/release-scaffolding version contracts;
- English and Russian install examples and current-release wording;
- `CHANGELOG.md`, `SPEC.md`, `WAKE.md`, roadmap/testing/beta-exit docs, and a
  dedicated `docs/release-notes/v0.2.0b2.md` file;
- the tag-driven release workflow with a tag-to-package version check and a
  full lint/test gate before distributions can reach the PyPI publish job.

Historical v0.1 and v0.2.0b1 release notes and dated readiness evidence remain
unchanged. Open Dependabot pull requests are not part of this release.

## Version Contract

`morpheus/__init__.py` is the runtime source for `__version__`. Package metadata
in `pyproject.toml` must equal it. Public API/manifest/provenance/WAKE surfaces
must report the same value, and release-scaffolding tests must fail on drift.

The release workflow must reject a pushed tag whose version without the leading
`v` differs from `project.version` in `pyproject.toml`. A manual
`workflow_dispatch` remains build-only and does not publish to PyPI.

## Public Claims

The public release claim is limited to capabilities exercised by repository
tests and existing source-backed reports:

- deterministic compile, signed receipts, local check, MCP truth tools, and
  agent-connect surfaces;
- review-gated datasets and the local experimental learning lab;
- canonical category benchmark and eval/activation/rollback gates;
- audited memory routing and signed reviewed active-state authority;
- an idempotent, receipt-backed team-learning loop for all six documented input
  types without automatic acceptance, training, or activation.

The release does not claim automatic production adapter activation, cloud
training, raw-Markdown training, or stable-release maturity.

## Pre-Tag Gate

The release commit must pass all of these before a tag exists:

```bash
.venv/bin/ruff check .
.venv/bin/pytest tests/ -q
.venv/bin/morpheus stale .
.venv/bin/morpheus compile
.venv/bin/morpheus diagnostics --json
.venv/bin/morpheus agent-connect --json
.venv/bin/morpheus wake . --private
.venv/bin/morpheus verify --all
make verify
make build
```

The built wheel and sdist must pass Twine validation. A clean temporary
installation from the wheel must report `Morpheus AI v0.2.0b2`. GitHub `main`
CI must then pass Python 3.10, 3.11, 3.12, and package-build jobs for the exact
release commit.

## Publish Sequence

1. Commit the version/runtime/test changes separately from public release docs.
2. Push `main` and wait for green CI on the exact commit SHA.
3. Create annotated tag `v0.2.0b2` on that SHA and push only that tag.
4. Wait for the tag-triggered Release workflow to build, validate, and publish
   `morpheus-wake==0.2.0b2` through PyPI Trusted Publishing.
5. Create the GitHub Release from `docs/release-notes/v0.2.0b2.md` only after
   the PyPI job succeeds.
6. Verify the GitHub Release, PyPI JSON, and exact package files.
7. Run pinned post-publish smoke commands in a disposable project:

```bash
uvx --from 'morpheus-wake==0.2.0b2' morpheus --version
uvx --from 'morpheus-wake==0.2.0b2' morpheus wake . --private
uvx --from 'morpheus-wake==0.2.0b2' morpheus verify --all
```

## Failure Protocol

No tag is created while local or `main` CI is red. Public tags are not moved or
reused. If the tag workflow fails before publication, the failure is diagnosed
without publishing from a dirty or unreviewed tree. If PyPI has accepted any
`0.2.0b2` artifact, that version is immutable; a corrected package uses the next
beta version instead of replacing files. A GitHub Release creation failure may
be retried for the same immutable tag after the PyPI state is verified.
