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
