# Release Guide

Morpheus ships as a Python package plus an optional local container for quick
API/UI testing.

The PyPI distribution is `morpheus-wake`. The installed CLI command is
`morpheus`.

## Canonical Pre-Tag Protocol

This file is the canonical release sequence. Run the complete local gate from a
clean release commit, in this order, before cutting a tag:

```bash
.venv/bin/ruff check .
.venv/bin/pytest tests/ -q
.venv/bin/morpheus stale .
.venv/bin/morpheus compile
sed -n '1,240p' WAKE.md
.venv/bin/morpheus diagnostics --json
.venv/bin/morpheus agent-connect --json
.venv/bin/morpheus wake . --private
.venv/bin/morpheus verify --all
make verify
make build
.venv/bin/python -m twine check dist/*
```

`make build` creates the source distribution and wheel in `dist/`. The
standalone Twine invocation is intentional: it verifies the final files again
after the build target exits.

Inspect the package name and version in both artifacts rather than relying only
on their filenames. The wheel metadata entry matches
`morpheus_wake-*.dist-info/METADATA`:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import tarfile
import zipfile

wheel = next(Path("dist").glob("morpheus_wake-*.whl"))
with zipfile.ZipFile(wheel) as archive:
    metadata_name = next(
        name for name in archive.namelist()
        if name.endswith(".dist-info/METADATA")
    )
    assert metadata_name.startswith("morpheus_wake-")
    metadata = archive.read(metadata_name).decode()

sdist = next(Path("dist").glob("morpheus_wake-*.tar.gz"))
with tarfile.open(sdist) as archive:
    pkg_info_name = next(name for name in archive.getnames() if name.endswith("/PKG-INFO"))
    pkg_info_file = archive.extractfile(pkg_info_name)
    assert pkg_info_file is not None
    pkg_info = pkg_info_file.read().decode()

for label, content in [
    ("wheel METADATA", metadata),
    ("sdist PKG-INFO", pkg_info),
]:
    assert "Name: morpheus-wake\n" in content, label
    assert "Version: 0.2.0b2\n" in content, label
    print(f"{label}: morpheus-wake 0.2.0b2")
PY
```

Finally, install the built wheel into an isolated environment and verify the
runtime version:

```bash
release_smoke_dir="$(mktemp -d)"
.venv/bin/python -m venv "$release_smoke_dir/venv"
"$release_smoke_dir/venv/bin/python" -m pip install dist/morpheus_wake-*.whl
"$release_smoke_dir/venv/bin/morpheus" --version
```

The final command must print exactly `Morpheus AI v0.2.0b2`. Remove the
temporary directory after recording the result.

Push the verified release commit to `main`. For that exact commit SHA, the
GitHub CI jobs named Python 3.10, Python 3.11, Python 3.12, and Package build
must all be green before creating or pushing an annotated tag. A green run for
another commit is not release evidence.

## PyPI Publishing

Publishing is handled by `.github/workflows/release.yml` on tags matching
`v*.*.*`. The workflow builds artifacts in one job and publishes in a separate
job with PyPI Trusted Publishing.

Required PyPI publisher configuration:

- repository owner and name,
- PyPI project name: `morpheus-wake`,
- workflow filename: `release.yml`,
- environment: `pypi`.

The workflow intentionally does not use `PYPI_TOKEN`; GitHub OIDC issues a
short-lived publishing identity for the `publish-pypi` job.

## Immutable Publish Sequence

The required order is:

```text
tag -> tag workflow -> verify PyPI wheel and sdist -> GitHub Release -> pinned post-publish smoke
```

1. Create the annotated version tag on the exact CI-green release commit and
   push only that tag.
2. Wait for the tag-triggered Release workflow to build, validate, and publish
   through PyPI Trusted Publishing.
3. Verify that PyPI serves both the expected wheel and sdist for the exact
   version, including their package metadata.
4. Create the GitHub Release only after the PyPI artifacts are verified.
5. Run the pinned commands in the Published Package Smoke section of
   [TESTING.md](TESTING.md) from a disposable project.

Never reuse a public tag. Never replace an artifact accepted by PyPI. If a
published artifact needs correction, increment the package version and create a
new tag; accepted PyPI files and their public tag remain immutable.

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
