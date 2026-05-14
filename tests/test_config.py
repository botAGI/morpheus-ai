"""
Tests for morpheus.core.config.
"""

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from morpheus.core.config import MorpheusConfig


def test_init_default_rejects_morpheus_symlink_without_writing_target(tmp_path):
    outside = tmp_path / "outside-morpheus"
    outside.mkdir()
    (tmp_path / ".morpheus").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=".morpheus path must not be a symlink"):
        MorpheusConfig(project_root=tmp_path).init_default()

    assert not (outside / "morpheus.toml").exists()
    assert not (outside / "keys").exists()


def test_init_default_rejects_symlinked_project_root_without_writing_target(tmp_path):
    outside = tmp_path / "outside-project"
    outside.mkdir()
    project_root = tmp_path / "linked-project"
    try:
        project_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="Project root must not be a symlink"):
        MorpheusConfig(project_root=project_root).init_default()

    assert not (outside / ".morpheus").exists()


def test_init_default_rejects_morpheus_state_file(tmp_path):
    (tmp_path / ".morpheus").write_text("not a directory")

    with pytest.raises(ValueError, match=".morpheus path is not a directory"):
        MorpheusConfig(project_root=tmp_path).init_default()


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


def test_init_default_rejects_config_symlink(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    outside_config = tmp_path / "outside.toml"
    outside_config.write_text("watch_dirs = ['.']\n")
    (morpheus_dir / "morpheus.toml").symlink_to(outside_config)

    with pytest.raises(ValueError, match="Config path must not be a symlink"):
        MorpheusConfig(project_root=tmp_path).init_default()


def test_init_default_rejects_private_key_symlink_without_writing_public_key(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    keys_dir = morpheus_dir / "keys"
    keys_dir.mkdir(parents=True)
    private_key = ed25519.Ed25519PrivateKey.generate()
    outside_key = tmp_path / "outside.key"
    outside_key.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    (keys_dir / "local.key").symlink_to(outside_key)

    with pytest.raises(ValueError, match="Private key path must not be a symlink"):
        MorpheusConfig(project_root=tmp_path).init_default()

    assert not (keys_dir / "local.pub").exists()


def test_init_default_rejects_public_key_symlink(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    keys_dir = morpheus_dir / "keys"
    keys_dir.mkdir(parents=True)
    private_key = ed25519.Ed25519PrivateKey.generate()
    (keys_dir / "local.key").write_bytes(
        private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    outside_public_key = tmp_path / "outside.pub"
    outside_public_key.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    )
    (keys_dir / "local.pub").symlink_to(outside_public_key)

    with pytest.raises(ValueError, match="Public key path must not be a symlink"):
        MorpheusConfig(project_root=tmp_path).init_default()


def test_load_rejects_morpheus_symlink(tmp_path):
    outside = tmp_path / "outside-morpheus"
    outside.mkdir()
    (outside / "morpheus.toml").write_text("watch_dirs = ['outside']\n")
    (tmp_path / ".morpheus").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=".morpheus path must not be a symlink"):
        MorpheusConfig(project_root=tmp_path).load()


def test_load_rejects_symlinked_project_root_without_reading_target(tmp_path):
    outside = tmp_path / "outside-project"
    outside_morpheus = outside / ".morpheus"
    outside_morpheus.mkdir(parents=True)
    (outside_morpheus / "morpheus.toml").write_text("watch_dirs = ['outside']\n")
    project_root = tmp_path / "linked-project"
    try:
        project_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="Project root must not be a symlink"):
        MorpheusConfig(project_root=project_root).load()


def test_load_rejects_morpheus_state_file(tmp_path):
    (tmp_path / ".morpheus").write_text("not a directory")

    with pytest.raises(ValueError, match=".morpheus path is not a directory"):
        MorpheusConfig(project_root=tmp_path).load()


def test_load_rejects_config_symlink(tmp_path):
    morpheus_dir = tmp_path / ".morpheus"
    morpheus_dir.mkdir()
    outside_config = tmp_path / "outside.toml"
    outside_config.write_text("watch_dirs = ['outside']\n")
    (morpheus_dir / "morpheus.toml").symlink_to(outside_config)

    with pytest.raises(ValueError, match="Config path must not be a symlink"):
        MorpheusConfig(project_root=tmp_path).load()
