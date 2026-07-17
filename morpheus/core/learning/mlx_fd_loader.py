"""MLX training entry point that reads Morpheus datasets only from pinned FDs."""

from hashlib import sha256
import json
import os
import stat
import sys
from typing import Callable

from morpheus.core.learning.training_runtime import PINNED_DATASET_FDS_ENV


_REQUIRED_SPLITS = ("train.jsonl", "valid.jsonl", "test.jsonl")


def read_pinned_jsonl_splits(
    raw_mapping: str | None = None,
) -> dict[str, list[dict]]:
    """Read and verify the three MLX splits through inherited descriptors."""
    raw_mapping = raw_mapping or os.environ.get(PINNED_DATASET_FDS_ENV)
    try:
        mapping = json.loads(raw_mapping or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pinned MLX dataset descriptor map invalid: {exc}") from exc
    if not isinstance(mapping, dict):
        raise ValueError("Pinned MLX dataset descriptor map must be an object")
    splits = {}
    for path in _REQUIRED_SPLITS:
        metadata = mapping.get(path)
        if not isinstance(metadata, dict):
            raise ValueError(f"Pinned MLX dataset split missing: {path}")
        descriptor = metadata.get("descriptor")
        size_bytes = metadata.get("size_bytes")
        expected_sha256 = metadata.get("sha256")
        if (
            isinstance(descriptor, bool)
            or not isinstance(descriptor, int)
            or descriptor < 0
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not _valid_sha256(expected_sha256)
        ):
            raise ValueError(f"Pinned MLX dataset split metadata invalid: {path}")
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != size_bytes:
            raise ValueError(f"Pinned MLX dataset split identity invalid: {path}")
        data = os.pread(descriptor, size_bytes + 1, 0)
        if len(data) != size_bytes or sha256(data).hexdigest() != expected_sha256:
            raise ValueError(f"Pinned MLX dataset split hash mismatch: {path}")
        try:
            rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Pinned MLX dataset split invalid: {path}: {exc}") from exc
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"Pinned MLX dataset split rows invalid: {path}")
        splits[path] = rows
    return splits


def pinned_mlx_load_dataset(
    args,
    tokenizer,
    *,
    create_dataset: Callable | None = None,
):
    """MLX-compatible loader backed exclusively by verified descriptor bytes."""
    if getattr(args, "hf_dataset", False):
        raise ValueError("Pinned MLX loader does not allow remote datasets")
    if str(getattr(args, "data", "")) != ".":
        raise ValueError("Pinned MLX loader requires the guarded dataset marker")
    if create_dataset is None:
        from mlx_lm.tuner.datasets import create_dataset as mlx_create_dataset

        create_dataset = mlx_create_dataset
    rows = read_pinned_jsonl_splits()
    train, valid, test = [
        create_dataset(rows[path], tokenizer, args)
        for path in _REQUIRED_SPLITS
    ]
    if getattr(args, "train", False) and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if getattr(args, "test", False) and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test


def main() -> int:
    """Patch MLX's dataset hook before any model training can consume data."""
    try:
        read_pinned_jsonl_splits()
        import mlx_lm.lora as lora

        lora.load_dataset = pinned_mlx_load_dataset
        lora.main()
    except (ImportError, OSError, ValueError) as exc:
        print(f"Pinned MLX training failed: {exc}", file=sys.stderr)
        return 2
    return 0


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


if __name__ == "__main__":
    raise SystemExit(main())
