# Morpheus v0.2.0b2 Release Design And Readiness

## Release Decision

The next package version is `0.2.0b2` and the Git tag is `v0.2.0b2`.

This is a beta update, not a stable-release claim. The existing beta exit rule
still requires broader real-repository and no-regression evidence before a
stable `v0.2.0` release. Roadmap labels v0.3 through v0.7 describe product
milestones and do not determine the Python package version.

The release contains the work since `v0.2.0b1`, including the semantic quality
and benchmark work, guarded adapter activation and rollback authority,
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

## Documentation Status

The public release documentation is prepared for `v0.2.0b2`: the README
quickstarts, changelog section, release notes, and current-version references
use the exact package version. This preparation is not publication evidence.
Final command results, artifact names, and the release SHA are recorded only by
the release execution task.

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

## Local Release Gate Evidence

The gated pre-evidence source commit was
`27b45ae643793f71b03b041b18ddfd1a1ec98b40`. The tracked tree and
`git diff --check` were clean before the gate. The following commands completed
successfully against that source tree on 2026-07-20:

- `.venv/bin/ruff check .`;
- `.venv/bin/pytest tests/ -q`: 1391 passed and 1 skipped;
- `.venv/bin/morpheus stale .`: no stale launch-positioning claims;
- `.venv/bin/morpheus compile`: 512 claims from 223 sources, receipt
  `rcpt_20260720T162445Z_30b5dd34`;
- `.venv/bin/morpheus diagnostics --json`: version `0.2.0b2`, initialized and
  compiled state, and all six diagnostic checks reported `ok: true`;
- `.venv/bin/morpheus agent-connect --json`: version `0.2.0b2`, initialized and
  compiled state, and `next_action.id` equal to `handoff`;
- `.venv/bin/morpheus wake . --private`: receipt
  `rcpt_20260720T162500Z_3cb29b5b`;
- `.venv/bin/morpheus verify --all`: the 217-receipt chain was valid and all
  signatures were verified;
- `make verify`: Ruff passed and the nested suite reported 1391 passed and 1
  skipped;
- `make build` and `.venv/bin/python -m twine check dist/*`: both distributions
  built and passed Twine validation.

The exact local artifact filenames were
`morpheus_wake-0.2.0b2-py3-none-any.whl` and
`morpheus_wake-0.2.0b2.tar.gz`. The wheel `METADATA` and sdist `PKG-INFO` both
reported package name `morpheus-wake` and version `0.2.0b2`. Installing the
wheel and its dependencies into a disposable `mktemp`-based virtual environment
and invoking `morpheus --version` produced exactly `Morpheus AI v0.2.0b2`.

A tracked Git commit cannot self-embed its own content-addressed SHA: adding the
SHA changes the commit contents and therefore changes that SHA. For that reason,
this tracked evidence records the gated pre-evidence source commit. The final
release commit is instead identified after this evidence commit is created and
must be verified by exact-SHA local gates, GitHub `main` CI, and the eventual tag
target. This section does not claim that GitHub CI has passed, that a tag exists,
or that any artifact has been published.
