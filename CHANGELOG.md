# Changelog

All notable changes to Morpheus will be documented in this file.

## [Unreleased]

## [0.2.0b2] - 2026-07-20

### Added

- Added signed active-state review authority schema
  `morpheus-active-state-review-authority/1`. Semantic review-apply receipts now
  bind complete reviewed candidates to the exact state claims, evidence records,
  and canonical memory-routing policy used by active-state datasets.
- Completed the canonical v0.5 adapter benchmark contract with schema
  `morpheus-benchmark-categories/1` and the exact coverage IDs
  `product_identity`, `commands_and_cli_behavior`, `architecture`,
  `safety_rules`, `team_conventions`, `stale_claim_correction`, and
  `unsupported_claim_refusal`; diagnostic `project_recall` does not satisfy
  coverage, and security/safety plus convention/team-convention remain
  independent requirements.
- Added human CLI and static UI visibility for benchmark schema, paired base and
  adapter eval IDs, readiness and gate reason, category deltas, all/critical
  regression counts, and blockers while preserving the existing JSON/API
  contract.
- Benchmark comparisons expose per-category pass-rate and hallucination-rate
  deltas, all regressions, and the critical subset.
- Completed the v0.7 `morpheus-team-learning/2` contract for
  `morpheus learn team-loop` and `POST /learning/team-loop`: one strict,
  idempotent path now covers PR comments, rejected agent claims, human
  corrections, accepted review candidates, check results, and stale-claim
  corrections. Every input accepted by local ingestion policy receives an
  immutable content-addressed receipt. Direct feedback, explicit stale-claim
  corrections, and stale/incorrect check results create pending candidates;
  accepted references and verified/unknown checks are receipt-only.
- Accepted corrections can now carry explicit replacement text into
  negative/correction training and eval examples while remaining excluded from
  positive active project state.

### Fixed

- Made `make verify` run Ruff and Pytest through the project Python so it no
  longer depends on globally installed tools.
- Made each team-input batch failure-atomic across receipts, evidence artifacts,
  the shared candidate store, and reports with prepared/committed recovery under
  the shared review lock. Cross-source ID squatting, projection drift, symlink
  races, and unsupported descriptor-relative filesystems now fail closed.
- Preserved the legacy `check --create-training-corrections` candidate IDs,
  evidence bytes, provider metadata, and source labels while delegating check
  results through the unified reviewed-input path.
- Made active-state learning fail closed instead of synthesizing accepted,
  source-backed candidates from every compiled claim. Plain compile/wake and
  legacy receipts remain chain-verifiable integrity records but cannot authorize
  learning; unbound compiler claims are excluded even from a review-apply state.
- Held state and review authority through semantic apply, binding construction,
  signing, and receipt publication so the signed review decision cannot change
  mid-transaction.
- Bound the live activation and rollback-to-adapter path to the same exact
  dataset/eval/readiness/category gate. Force cannot bypass it; rollback-to-none
  remains the fail-safe. Legacy or mismatched manifest/eval/category schema
  requires a dataset rebuild and new base/adapter evals.
- Made the trained adapter manifest's exact regular `.safetensors` path, byte
  size, and SHA-256 authoritative for activation and rollback; preview artifacts
  remain ineligible.
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
