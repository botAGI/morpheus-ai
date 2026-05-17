from pathlib import Path

import pytest

from morpheus.core.safe_io import reject_symlink_components


def test_reject_symlink_components_rejects_broken_symlink_path(tmp_path):
    broken_link = tmp_path / "broken-link"
    try:
        broken_link.symlink_to(tmp_path / "missing-target")
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="Input path must not contain a symlink"):
        reject_symlink_components(broken_link, "Input path")


def test_reject_symlink_components_rejects_broken_symlink_parent(tmp_path):
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(tmp_path / "missing-directory", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    with pytest.raises(ValueError, match="Input path must not contain a symlink"):
        reject_symlink_components(linked_parent / "child.txt", "Input path")


def test_reject_symlink_components_rejects_relative_symlink_path(tmp_path, monkeypatch):
    target = tmp_path / "target.txt"
    link = tmp_path / "linked-file"
    target.write_text("safe target")
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Input path must not contain a symlink"):
        reject_symlink_components(Path("linked-file"), "Input path")


def test_reject_symlink_components_rejects_relative_symlink_parent(tmp_path, monkeypatch):
    target = tmp_path / "target-directory"
    linked_parent = tmp_path / "linked-parent"
    target.mkdir()
    try:
        linked_parent.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Input path must not contain a symlink"):
        reject_symlink_components(Path("linked-parent") / "child.txt", "Input path")
