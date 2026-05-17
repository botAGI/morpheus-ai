# Contributing

## Development Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Useful shortcuts:

```bash
make lint
make test
make verify
make serve
```

## Quality Gates

Before committing code, run:

```bash
ruff check .
pytest tests/ -q
python -m build
python -m twine check dist/*
```

Feature work should include focused tests. Security-sensitive changes should
cover symlinks, path traversal, malformed input, and failure paths.

## Release Process

Releases are tag-driven:

1. Update `CHANGELOG.md`.
2. Verify locally with `make verify` and `make build`.
3. Create a version tag such as `v0.1.1`.
4. Push the tag.
5. Confirm the GitHub Actions release workflow built `sdist` and `wheel`
   artifacts.

## Trusted Publishing

The release workflow is prepared for PyPI Trusted Publishing. Configure PyPI
with:

- repository owner and repository name,
- PyPI project name: `morpheus-wake`,
- workflow filename: `release.yml`,
- environment: `pypi`.

Do not add `PYPI_TOKEN`, usernames, or passwords to the workflow. The publish
job uses GitHub OIDC with `id-token: write`, scoped only to that job.

Manual `workflow_dispatch` runs build and validate artifacts. Publishing occurs
only from version tags that match `v*.*.*`.
