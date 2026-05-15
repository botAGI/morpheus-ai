# Release Guide

Morpheus ships as a Python package plus an optional local container for quick
API/UI testing.

## Local Verification

Run the full local gate before cutting a tag:

```bash
make verify
make build
```

The build command creates both source distribution and wheel artifacts in
`dist/`, then validates package metadata with Twine.

## PyPI Publishing

Publishing is handled by `.github/workflows/release.yml` on tags matching
`v*.*.*`. The workflow builds artifacts in one job and publishes in a separate
job with PyPI Trusted Publishing.

Required PyPI publisher configuration:

- repository owner and name,
- workflow filename: `release.yml`,
- environment: `pypi`.

The workflow intentionally does not use `PYPI_TOKEN`; GitHub OIDC issues a
short-lived publishing identity for the `publish-pypi` job.

## Container

Build the local container:

```bash
make docker-build
```

Run the API and UI:

```bash
make docker-run
```

Then open `http://127.0.0.1:5173/ui/index.html`.

For project compilation inside a container, mount the target workspace and run
Morpheus from that mounted path so generated `.morpheus/`, `WAKE.md`, and
receipts stay with the project.

## Operational Defaults

- Use `127.0.0.1` for private local work.
- Use `0.0.0.0` only on a trusted LAN or behind an authenticated proxy.
- Keep MCP/A2A remote exposure behind a clear network trust boundary.
- Keep generated state and local integration caches out of git.
