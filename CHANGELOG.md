# Changelog

All notable changes to Morpheus will be documented in this file.

## [Unreleased]

### Added

- Added an idempotent local `morpheus learn team-loop` and
  `POST /learning/team-loop` flow that stores PR comments, rejected agent claims,
  and human corrections as pending review candidates without training or adapter
  activation.
- Accepted corrections can now carry explicit replacement text into
  negative/correction training and eval examples while remaining excluded from
  positive active project state.

### Fixed

- Removed source-less synthetic truth-gate rows from adapter training, kept
  those scenarios eval-only, archived stale-class candidates with incompatible
  kinds, and made dataset and eval-receipt validation bind exact candidate,
  source-span, evidence, and route metadata.
- Made adapter activation fail closed: deterministic fake/dry-run evals are
  diagnostic-only, activation requires exact paired artifacts with consistent
  metrics, and the legacy force flag no longer bypasses a failed gate.
- Preserved reviewed and correction candidates across semantic rescans, made
  repeated check-correction creation idempotent, and hardened correction artifact
  writes against symlink targets.

## [0.2.0b1] - 2026-05-20

### Added

- Review-gated semantic compile alpha for `morpheus compile --semantic --review`
  and `morpheus wake . --semantic --review`.
- Semantic review CLI for listing, accepting, rejecting, diffing, and applying
  source-backed candidates.
- Local `morpheus check` for file/stdin claim verification with
  `verified`, `stale`, `incorrect`, and `unknown` statuses.
- Learning core commands for dataset generation, dry-run training artifacts,
  eval, adapter registry/status, and autonomous lab runs.
- Learning status now reports the effective trainable dataset across standalone
  and lab datasets.
- Live dogfood MLX lab report showing `ML_CORE_PASS` on current main with
  strict source-backed candidates and full base-vs-adapter eval coverage.
- Repeat-2 dogfood MLX stability report showing 69 strict accepted candidates,
  290 examples, 148/148 eval coverage per run, zero critical failures, and zero
  regressions.
- Local, fake, null, and explicit local Ollama semantic providers.

### Fixed

- Hardened semantic review outputs and apply receipts against symlinked paths.
- Capped local Ollama semantic prompt source content before sending it to the
  local model endpoint.
- Fixed `morpheus check` no-input behavior so an interactive terminal does not
  hang waiting for stdin.
- Updated stale-positioning scan so the valid "truth layer before
  weights-as-memory" framing is no longer reported as stale.
- Made `make build` prefer the project `.venv` and `uv pip install --python`
  so macOS externally-managed Python environments do not block release builds.
- Let dry-run training and eval use the latest trainable lab dataset when no
  standalone reviewed dataset exists.
- Deprecated root `morpheus train` execution by default; legacy raw-dataset
  training now requires an explicit unsafe confirmation flag.
- Replaced the old scheduled raw-training script with a safe daily learning-lab gate
  that does not train or activate adapters.
- Strengthened hard-negative learning examples and the lab system prompt so the
  adapter preserves the "truth layer before weights" framing instead of drifting
  back to a LoRA-trainer framing.
- Added root `morpheus --version` support.
- Updated public docs and package URLs so main no longer reads like v0.1.0.

## [0.1.1] - 2026-05-18

### Fixed

- Removed local testbot path from public AGENTS.md.
- Fixed PyPI long-description links and demo image by using absolute GitHub URLs.
- Removed obsolete Typer `all` extra warning.
- Added project URLs to package metadata.

## [0.1.0] - 2026-05-17

### Added

- `WAKE.md` launch framing with root showcase file and `docs/WHY_WAKE.md`.
- `morpheus wake .` one-command project wake flow with public and private modes.
- `morpheus stale .` scan for stale launch-positioning claims.
- Visual terminal demo embedded in the English and Russian README files.
- PyPI distribution renamed to `morpheus-wake` while keeping the `morpheus` CLI.
- Node 24-compatible GitHub Actions majors for checkout, Python setup, and artifacts.
- A2A Agent Card discovery for agent-to-agent connection metadata.
- MCP Streamable HTTP endpoint with read-only Morpheus tools.
- Quickstart launchpad for humans and agents in the HTTP API and UI.
- Slack and Linear cache-backed integration adapters.
- Model smoke command and API for local Ollama checks.
- GitHub Actions CI and tag-based PyPI Trusted Publishing workflow.
- Dockerfile, Makefile, Dependabot config, security policy, and contributor
  release notes.

### Fixed

- Integration cache timestamp normalization across ISO strings and epoch values.
- Stable CLI path output for CI environments with narrow terminal widths.
