# Security Policy

## Supported Versions

Morpheus is pre-1.0 software. Security fixes target the current `main` branch
and the latest tagged release.

| Version | Supported |
| ------- | --------- |
| `0.1.x` | Yes |
| Older snapshots | No |

## Reporting a Vulnerability

Report suspected vulnerabilities privately before opening a public issue. Until
a dedicated security advisory channel is configured, email `team@morpheus.ai`
with:

- affected version or commit,
- operating system and Python version,
- reproduction steps,
- impact and whether secrets, filesystem data, MCP/A2A traffic, or receipts are
  involved.

We will acknowledge actionable reports, avoid public disclosure until a fix is
available, and credit reporters when requested.

## Local-first Security Model

Morpheus is designed to run next to the project it compiles. Treat the selected
project root, `.morpheus/`, `WAKE.md`, receipts, integration caches, and model
smoke prompts as local project data.

Recommended defaults:

- bind API and UI to `127.0.0.1` on untrusted networks,
- use `0.0.0.0` only for trusted LAN testing,
- put authentication, a trusted tunnel, or a reverse proxy in front of remote
  access,
- do not commit `.morpheus/`, generated receipts, local integration caches, or
  tokens,
- keep integration token files outside the project checkout,
- run `morpheus verify --all` after meaningful state changes.

## MCP/A2A Exposure

The MCP and A2A endpoints are intended for local agent interop and LAN tests.
They should be treated as automation surfaces: any remote exposure needs network
controls, request logging, and an explicit trust boundary.
