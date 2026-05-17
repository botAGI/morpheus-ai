"""Shared helpers for local integration cache files."""
import gzip
import json
from pathlib import Path

from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths


def load_cache_payload(cache_path: Path) -> object | None:
    """Load a JSON or JSONL cache file payload."""
    try:
        reject_symlink_paths([cache_path], "Integration cache path")
        reject_symlink_components(cache_path, "Integration cache path")
        contents = cache_path.read_bytes()
        if cache_path.suffix.casefold() == ".gz" or contents.startswith(b"\x1f\x8b"):
            text = gzip.decompress(contents).decode()
        else:
            text = contents.decode()
        return parse_cache_payload(text)
    except (EOFError, OSError, UnicodeError, ValueError):
        return None


def parse_cache_payload(text: str) -> object | None:
    """Parse a cache payload from JSON or newline-delimited JSON text."""
    text = text.removeprefix("\ufeff")
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _parse_json_lines(text)


def _parse_json_lines(text: str) -> list | None:
    rows = []
    for line in text.splitlines():
        line = line.strip().removeprefix("\ufeff")
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows or None


def cache_rows(payload: object, *collection_keys: str) -> list:
    """Return cache rows from either a raw JSON array or a wrapped export object."""
    found, rows = _find_cache_rows(
        payload,
        collection_keys,
        seen=set(),
        accept_raw_list=True,
        allow_graphql_connection=True,
    )
    return rows if found else []


def _find_cache_rows(
    payload: object,
    collection_keys: tuple[str, ...],
    *,
    seen: set[int],
    accept_raw_list: bool,
    allow_graphql_connection: bool,
) -> tuple[bool, list]:
    if isinstance(payload, list):
        payload_id = id(payload)
        if payload_id in seen:
            return False, []
        seen.add(payload_id)

        if accept_raw_list:
            found_nested, nested_rows = _find_cache_rows_in_list(
                payload,
                collection_keys,
                seen=seen,
                require_all_items=True,
            )
            if found_nested:
                return True, nested_rows
            return True, payload
        return _find_cache_rows_in_list(
            payload,
            collection_keys,
            seen=seen,
            require_all_items=False,
        )
    if not isinstance(payload, dict):
        return False, []

    payload_id = id(payload)
    if payload_id in seen:
        return False, []
    seen.add(payload_id)

    for key in collection_keys:
        found_key, value = _collection_value(payload, key)
        if not found_key:
            continue
        if isinstance(value, list):
            found_nested, nested_rows = _find_cache_rows_in_list(
                value,
                collection_keys,
                seen=seen,
                require_all_items=True,
            )
            if found_nested:
                return True, nested_rows
            return True, value

    for key in collection_keys:
        found_key, value = _collection_value(payload, key)
        if not found_key:
            continue
        if isinstance(value, dict):
            found, rows = _find_cache_rows(
                value,
                collection_keys,
                seen=seen,
                accept_raw_list=False,
                allow_graphql_connection=True,
            )
            if found:
                return True, rows

    if allow_graphql_connection:
        found_connection, connection_rows = _graphql_connection_rows(
            payload,
            collection_keys,
            seen=seen,
        )
        if found_connection:
            return True, connection_rows

    for value in payload.values():
        if not isinstance(value, (dict, list)):
            continue
        found, rows = _find_cache_rows(
            value,
            collection_keys,
            seen=seen,
            accept_raw_list=False,
            allow_graphql_connection=False,
        )
        if found:
            return True, rows
    return False, []


def _collection_value(payload: dict, key: str) -> tuple[bool, object]:
    if key in payload:
        return True, payload[key]

    normalized_key = _collection_key_fingerprint(key)
    if not normalized_key:
        return False, None
    for candidate_key, value in payload.items():
        if not isinstance(candidate_key, str):
            continue
        if _collection_key_fingerprint(candidate_key) == normalized_key:
            return True, value
    return False, None


def _collection_key_fingerprint(key: str) -> str:
    """Normalize common JSON export key styles such as snake_case and camelCase."""
    return "".join(character for character in key.casefold() if character.isalnum())


def _find_cache_rows_in_list(
    payload: list,
    collection_keys: tuple[str, ...],
    *,
    seen: set[int],
    require_all_items: bool,
) -> tuple[bool, list]:
    rows = []
    found_any = False
    for item in payload:
        if not isinstance(item, (dict, list)):
            if require_all_items:
                rows.append(item)
                continue
            continue
        row_index = len(rows)
        if require_all_items:
            rows.append(item)
        found, item_rows = _find_cache_rows(
            item,
            collection_keys,
            seen=seen,
            accept_raw_list=False,
            allow_graphql_connection=False,
        )
        if not found:
            if require_all_items:
                continue
            continue
        found_any = True
        if require_all_items:
            # Named export lists can mix raw rows with wrapped per-service chunks.
            rows[row_index:row_index + 1] = item_rows
            continue
        rows.extend(item_rows)
    return (True, rows) if found_any else (False, [])


def _graphql_connection_rows(
    payload: dict,
    collection_keys: tuple[str, ...],
    *,
    seen: set[int],
) -> tuple[bool, list]:
    found_nodes, nodes = _collection_value(payload, "nodes")
    if found_nodes and isinstance(nodes, list):
        return _find_cache_rows(
            nodes,
            collection_keys,
            seen=seen,
            accept_raw_list=True,
            allow_graphql_connection=True,
        )

    found_edges, edges = _collection_value(payload, "edges")
    if not found_edges or not isinstance(edges, list):
        return False, []

    rows = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        found_node, node = _collection_value(edge, "node")
        if not found_node or node is None:
            continue
        if isinstance(node, list):
            rows.extend(node)
        else:
            rows.append(node)

    if not rows:
        return False, []
    return _find_cache_rows(
        rows,
        collection_keys,
        seen=seen,
        accept_raw_list=True,
        allow_graphql_connection=True,
    )
