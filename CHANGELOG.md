# Changelog

All notable changes to Morpheus will be documented in this file.

## [Unreleased]

### Added

- `WAKE.md` launch framing with root showcase file and `docs/WHY_WAKE.md`.
- `morpheus wake .` one-command project wake flow with public and private modes.
- `morpheus stale .` scan for stale launch-positioning claims.
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
