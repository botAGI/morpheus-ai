"""
Tests for morpheus.core.config.
"""

import pytest

from morpheus.core.config import MorpheusConfig


def test_init_default_rejects_morpheus_symlink_without_writing_target(tmp_path):
    outside = tmp_path / "outside-morpheus"
    outside.mkdir()
    (tmp_path / ".morpheus").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=".morpheus path must not be a symlink"):
        MorpheusConfig(project_root=tmp_path).init_default()

    assert not (outside / "morpheus.toml").exists()
    assert not (outside / "keys").exists()


def test_init_default_rejects_keys_symlink_without_writing_target(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    outside_keys = tmp_path / "outside-keys"
    outside_keys.mkdir()
    (morpheus_dir / "keys").symlink_to(outside_keys, target_is_directory=True)

    with pytest.raises(ValueError, match="Keys path must not be a symlink"):
        MorpheusConfig(project_root=tmp_path).init_default()

    assert not (outside_keys / "local.key").exists()
    assert not (outside_keys / "local.pub").exists()


def test_init_default_rejects_receipts_symlink(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    outside_receipts = tmp_path / "outside-receipts"
    outside_receipts.mkdir()
    (morpheus_dir / "receipts").symlink_to(outside_receipts, target_is_directory=True)

    with pytest.raises(ValueError, match="Receipts path must not be a symlink"):
        MorpheusConfig(project_root=tmp_path).init_default()
