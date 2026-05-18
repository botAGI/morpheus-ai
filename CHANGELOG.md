# Changelog

All notable changes to Morpheus will be documented in this file.

## [Unreleased]

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
