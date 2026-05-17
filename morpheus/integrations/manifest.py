"""
Machine-readable integration manifest for CLI, API, UI, and agents.
"""
from pathlib import Path

from morpheus.core.safe_io import reject_symlink_components


INTEGRATION_DEFINITIONS = [
    {
        "id": "github",
        "label": "GitHub",
        "auth": "cache + PAT",
        "source": "cached or live issues, pull requests, commits",
        "token_file": "github_token.txt",
        "cache_file": "github_cache.json",
        "setup_command": "morpheus integrate github",
    },
    {
        "id": "gmail",
        "label": "Gmail",
        "auth": "cache + OAuth2",
        "source": "email cache/OAuth placeholder",
        "token_file": None,
        "cache_file": "gmail_cache.json",
        "setup_command": "morpheus integrate gmail",
    },
    {
        "id": "calendar",
        "label": "Calendar",
        "auth": "cache + OAuth2",
        "source": "calendar cache/OAuth placeholder",
        "token_file": None,
        "cache_file": "calendar_cache.json",
        "setup_command": "morpheus integrate calendar",
    },
    {
        "id": "slack",
        "label": "Slack",
        "auth": "cache + token",
        "source": "exported messages",
        "token_file": "slack_token.txt",
        "cache_file": "slack_cache.json",
        "setup_command": "morpheus integrate slack",
    },
    {
        "id": "linear",
        "label": "Linear",
        "auth": "cache + token",
        "source": "exported issues",
        "token_file": "linear_token.txt",
        "cache_file": "linear_cache.json",
        "setup_command": "morpheus integrate linear",
    },
]


def integration_token_path_error(token_path: Path, service_label: str) -> str | None:
    token_dir = token_path.parent
    if token_dir.is_symlink():
        return f"{service_label} token directory must not be a symlink: {token_dir}"
    if token_dir.exists() and not token_dir.is_dir():
        return f"{service_label} token directory is not a directory: {token_dir}"
    if token_path.is_symlink():
        return f"{service_label} token path must not be a symlink: {token_path}"
    if token_path.exists() and not token_path.is_file():
        return f"{service_label} token path is not a file: {token_path}"
    try:
        reject_symlink_components(token_path, f"{service_label} token path")
    except ValueError as exc:
        return str(exc)
    return None


def integration_cache_path_error(cache_path: Path, service_label: str) -> str | None:
    cache_dir = cache_path.parent
    if cache_dir.is_symlink():
        return f"{service_label} cache directory must not be a symlink: {cache_dir}"
    if cache_dir.exists() and not cache_dir.is_dir():
        return f"{service_label} cache directory is not a directory: {cache_dir}"
    if cache_path.is_symlink():
        return f"{service_label} cache path must not be a symlink: {cache_path}"
    if cache_path.exists() and not cache_path.is_file():
        return f"{service_label} cache path is not a file: {cache_path}"
    try:
        reject_symlink_components(cache_path, f"{service_label} cache path")
    except ValueError as exc:
        return str(exc)
    return None


def integration_manifest(home: Path | None = None) -> dict:
    morpheus_home = (home or Path.home()) / ".morpheus"
    services = [
        integration_service_entry(definition, morpheus_home)
        for definition in INTEGRATION_DEFINITIONS
    ]
    return {
        "service": "morpheus",
        "version": "0.1.0",
        "home": str(morpheus_home),
        "services": services,
    }


def integration_service_entry(definition: dict, morpheus_home: Path) -> dict:
    service_id = definition["id"]
    label = definition["label"]
    token_path = (
        morpheus_home / definition["token_file"]
        if definition.get("token_file")
        else None
    )
    cache_path = (
        morpheus_home / definition["cache_file"]
        if definition.get("cache_file")
        else None
    )

    errors = []
    if token_path:
        token_error = integration_token_path_error(token_path, label)
        if token_error:
            errors.append(token_error)
    if cache_path:
        cache_error = integration_cache_path_error(cache_path, label)
        if cache_error:
            errors.append(cache_error)

    if errors:
        status = "invalid"
        detail = "; ".join(errors)
    elif token_path and token_path.is_file():
        status = "configured"
        detail = f"{label} token configured"
    elif cache_path and cache_path.is_file():
        status = "cache_ready"
        detail = f"{label} cache ready"
    else:
        status = "not_configured"
        detail = f"Run `{definition['setup_command']}`"

    entry = {
        "id": service_id,
        "label": label,
        "auth": definition["auth"],
        "source": definition["source"],
        "status": status,
        "detail": detail,
        "setup_command": definition["setup_command"],
    }
    if token_path:
        entry["token_path"] = str(token_path)
    if cache_path:
        entry["cache_path"] = str(cache_path)
    return entry
