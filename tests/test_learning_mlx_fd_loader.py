from hashlib import sha256
import json
import os
from types import SimpleNamespace

import pytest

from morpheus.core.learning.mlx_fd_loader import pinned_mlx_load_dataset
from morpheus.core.learning.mlx_fd_loader import read_pinned_jsonl_splits
from morpheus.core.learning.training_runtime import PINNED_DATASET_FDS_ENV


def _pinned_split_environment(tmp_path):
    descriptors = []
    mapping = {}
    expected = {}
    for split in ("train", "valid", "test"):
        rows = [{"split": split, "trusted": True}]
        data = (json.dumps(rows[0], sort_keys=True) + "\n").encode()
        path = tmp_path / f"{split}.jsonl"
        path.write_bytes(data)
        descriptor = os.open(path, os.O_RDONLY)
        descriptors.append(descriptor)
        mapping[path.name] = {
            "descriptor": descriptor,
            "size_bytes": len(data),
            "sha256": sha256(data).hexdigest(),
        }
        expected[path.name] = rows
    return json.dumps(mapping), descriptors, expected


def test_pinned_mlx_loader_reads_verified_descriptors_not_dataset_paths(
    tmp_path,
    monkeypatch,
):
    raw_mapping, descriptors, expected = _pinned_split_environment(tmp_path)
    untrusted_dir = tmp_path / "untrusted"
    untrusted_dir.mkdir()
    for split in ("train", "valid", "test"):
        (untrusted_dir / f"{split}.jsonl").write_text('{"trusted": false}\n')
    monkeypatch.chdir(untrusted_dir)
    monkeypatch.setenv(PINNED_DATASET_FDS_ENV, raw_mapping)

    try:
        actual = read_pinned_jsonl_splits()
        train, valid, test = pinned_mlx_load_dataset(
            SimpleNamespace(data=".", train=True, test=True, hf_dataset=False),
            object(),
            create_dataset=lambda rows, _tokenizer, _args: rows,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)

    assert actual == expected
    assert [train, valid, test] == [
        expected["train.jsonl"],
        expected["valid.jsonl"],
        expected["test.jsonl"],
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("descriptor", True, "metadata invalid"),
        ("size_bytes", True, "metadata invalid"),
        ("size_bytes", 999, "identity invalid"),
        ("sha256", "0" * 64, "hash mismatch"),
    ],
)
def test_pinned_mlx_loader_rejects_invalid_descriptor_contract(
    tmp_path,
    field,
    value,
    message,
):
    raw_mapping, descriptors, _expected = _pinned_split_environment(tmp_path)
    mapping = json.loads(raw_mapping)
    mapping["train.jsonl"][field] = value

    try:
        with pytest.raises(ValueError, match=message):
            read_pinned_jsonl_splits(json.dumps(mapping))
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def test_pinned_mlx_loader_rejects_remote_or_unmarked_dataset(tmp_path, monkeypatch):
    raw_mapping, descriptors, _expected = _pinned_split_environment(tmp_path)
    monkeypatch.setenv(PINNED_DATASET_FDS_ENV, raw_mapping)

    try:
        with pytest.raises(ValueError, match="remote datasets"):
            pinned_mlx_load_dataset(
                SimpleNamespace(data=".", train=True, test=False, hf_dataset=True),
                object(),
                create_dataset=lambda rows, _tokenizer, _args: rows,
            )
        with pytest.raises(ValueError, match="guarded dataset marker"):
            pinned_mlx_load_dataset(
                SimpleNamespace(
                    data=str(tmp_path),
                    train=True,
                    test=False,
                    hf_dataset=False,
                ),
                object(),
                create_dataset=lambda rows, _tokenizer, _args: rows,
            )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
