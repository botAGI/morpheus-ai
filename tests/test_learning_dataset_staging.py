from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import stat
from pathlib import Path

import pytest

from morpheus.core.learning import dataset as dataset_module
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.registry import datasets_root, latest_dataset_dir
from morpheus.core.semantic.models import SemanticCandidate
from morpheus.core.semantic.review import ReviewStore


FIXED_DATASET_ID = "20260718T000000000000Z"


def _learning_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    source = project_root / "README.md"
    evidence = "Morpheus publishes reviewed datasets atomically."
    source.write_text(evidence + "\n")
    timestamp = datetime.now(timezone.utc)
    candidate = SemanticCandidate(
        id="candidate_staging",
        run_id="semrun_staging",
        kind="current_state",
        claim=evidence,
        source_path="README.md",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_mtime=timestamp,
        source_revision="git:test",
        line_start=1,
        line_end=1,
        evidence_excerpt=evidence,
        evidence_sha256=hashlib.sha256(evidence.encode()).hexdigest(),
        confidence=0.99,
        label="source_backed",
        status="accepted",
        created_at=timestamp,
        provider={"name": "local", "model": "fixture"},
        prompt_sha256="a" * 64,
        reviewed_by="tester",
        reviewed_at=timestamp,
    )
    ReviewStore(project_root).save_candidates([candidate])
    return project_root


def _hidden_staging_dirs(project_root: Path) -> list[Path]:
    root = datasets_root(project_root)
    if not root.is_dir():
        return []
    return sorted(
        (entry for entry in root.iterdir() if entry.name.startswith(".") and entry.is_dir()),
        key=lambda entry: entry.name,
    )


def _private_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)


def test_review_mutation_leaves_private_hidden_orphan_without_current_dataset(
    tmp_path,
    monkeypatch,
):
    project_root = _learning_project(tmp_path)
    original_check = dataset_module._source_hashes_are_current

    def mutate_review(*args, **kwargs):
        store = ReviewStore(project_root)
        candidates = store.load_candidates()
        store.save_candidates([
            candidates[0].model_copy(update={"status": "pending"}),
        ])
        return original_check(*args, **kwargs)

    monkeypatch.setattr(dataset_module, "_source_hashes_are_current", mutate_review)

    with pytest.raises(ValueError, match="Review state changed"):
        build_learning_dataset(project_root)

    staging_dirs = _hidden_staging_dirs(project_root)
    assert len(staging_dirs) == 1
    assert _private_mode(staging_dirs[0]) == 0o700
    assert all(_private_mode(path) == 0o600 for path in staging_dirs[0].iterdir())
    assert latest_dataset_dir(project_root) is None
    assert not [
        entry
        for entry in datasets_root(project_root).iterdir()
        if not entry.name.startswith(".")
    ]


def test_staging_names_are_random_even_when_dataset_identity_is_fixed(tmp_path, monkeypatch):
    project_root = _learning_project(tmp_path)
    monkeypatch.setattr(dataset_module, "_dataset_id", lambda: FIXED_DATASET_ID)
    monkeypatch.setattr(dataset_module, "_source_hashes_are_current", lambda *args: False)

    for _ in range(2):
        with pytest.raises(ValueError, match="Learning sources changed"):
            build_learning_dataset(project_root)

    staging_dirs = _hidden_staging_dirs(project_root)
    assert len(staging_dirs) == 2
    assert len({path.name for path in staging_dirs}) == 2
    assert all(path.name.startswith(f".{FIXED_DATASET_ID}.") for path in staging_dirs)
    assert all(_private_mode(path) == 0o700 for path in staging_dirs)


def test_replaced_staging_identity_refuses_publish_and_preserves_sentinel(
    tmp_path,
    monkeypatch,
):
    project_root = _learning_project(tmp_path)
    monkeypatch.setattr(dataset_module, "_dataset_id", lambda: FIXED_DATASET_ID)
    real_lock = dataset_module.portable_file_lock
    captured: dict[str, Path] = {}

    @contextmanager
    def replacing_lock(lock_path):
        with real_lock(lock_path):
            staging = _hidden_staging_dirs(project_root)[0]
            detached = staging.with_name(staging.name + ".detached")
            staging.rename(detached)
            staging.mkdir(mode=0o700)
            sentinel = staging / "sentinel.txt"
            sentinel.write_text("external sentinel\n")
            sentinel.chmod(0o600)
            captured.update(staging=staging, detached=detached, sentinel=sentinel)
            yield

    monkeypatch.setattr(dataset_module, "portable_file_lock", replacing_lock)

    with pytest.raises(ValueError, match="staging identity changed"):
        build_learning_dataset(project_root)

    assert captured["sentinel"].read_text() == "external sentinel\n"
    assert captured["detached"].is_dir()
    assert not (datasets_root(project_root) / FIXED_DATASET_ID).exists()
    assert latest_dataset_dir(project_root) is None


def test_renamed_staging_refuses_publish_and_preserves_detached_tree(
    tmp_path,
    monkeypatch,
):
    project_root = _learning_project(tmp_path)
    monkeypatch.setattr(dataset_module, "_dataset_id", lambda: FIXED_DATASET_ID)
    real_lock = dataset_module.portable_file_lock
    captured: dict[str, Path] = {}

    @contextmanager
    def renaming_lock(lock_path):
        with real_lock(lock_path):
            staging = _hidden_staging_dirs(project_root)[0]
            detached = staging.with_name(staging.name + ".detached")
            staging.rename(detached)
            captured.update(staging=staging, detached=detached)
            yield

    monkeypatch.setattr(dataset_module, "portable_file_lock", renaming_lock)

    with pytest.raises(ValueError, match="staging identity changed"):
        build_learning_dataset(project_root)

    assert not captured["staging"].exists()
    assert captured["detached"].is_dir()
    assert (captured["detached"] / "manifest.json").is_file()
    assert not (datasets_root(project_root) / FIXED_DATASET_ID).exists()


def test_unknown_staging_child_refuses_publish_without_deleting_it(tmp_path, monkeypatch):
    project_root = _learning_project(tmp_path)
    monkeypatch.setattr(dataset_module, "_dataset_id", lambda: FIXED_DATASET_ID)
    real_lock = dataset_module.portable_file_lock
    captured: dict[str, Path] = {}

    @contextmanager
    def injecting_lock(lock_path):
        with real_lock(lock_path):
            staging = _hidden_staging_dirs(project_root)[0]
            unknown = staging / "unapproved.bin"
            unknown.write_bytes(b"do not delete")
            unknown.chmod(0o600)
            captured.update(staging=staging, unknown=unknown)
            yield

    monkeypatch.setattr(dataset_module, "portable_file_lock", injecting_lock)

    with pytest.raises(ValueError, match="unexpected entries"):
        build_learning_dataset(project_root)

    assert captured["unknown"].read_bytes() == b"do not delete"
    assert captured["staging"].is_dir()
    assert not (datasets_root(project_root) / FIXED_DATASET_ID).exists()
    assert latest_dataset_dir(project_root) is None


def test_allowlisted_symlink_refuses_publish_without_touching_target(tmp_path, monkeypatch):
    project_root = _learning_project(tmp_path)
    monkeypatch.setattr(dataset_module, "_dataset_id", lambda: FIXED_DATASET_ID)
    real_lock = dataset_module.portable_file_lock
    outside = tmp_path / "outside.jsonl"
    outside.write_text("sentinel outside staging\n")
    captured: dict[str, Path] = {}

    @contextmanager
    def symlinking_lock(lock_path):
        with real_lock(lock_path):
            staging = _hidden_staging_dirs(project_root)[0]
            artifact = staging / "dataset.instruction.jsonl"
            artifact.unlink()
            artifact.symlink_to(outside)
            captured.update(staging=staging, artifact=artifact)
            yield

    monkeypatch.setattr(dataset_module, "portable_file_lock", symlinking_lock)

    with pytest.raises(ValueError, match="regular file"):
        build_learning_dataset(project_root)

    assert outside.read_text() == "sentinel outside staging\n"
    assert captured["artifact"].is_symlink()
    assert captured["staging"].is_dir()
    assert not (datasets_root(project_root) / FIXED_DATASET_ID).exists()


def test_source_change_after_registry_lock_refuses_publish(tmp_path, monkeypatch):
    project_root = _learning_project(tmp_path)
    monkeypatch.setattr(dataset_module, "_dataset_id", lambda: FIXED_DATASET_ID)
    real_lock = dataset_module.portable_file_lock

    @contextmanager
    def mutating_lock(lock_path):
        with real_lock(lock_path):
            (project_root / "README.md").write_text("Changed after staging.\n")
            yield

    monkeypatch.setattr(dataset_module, "portable_file_lock", mutating_lock)

    with pytest.raises(ValueError, match="Learning sources changed"):
        build_learning_dataset(project_root)

    assert len(_hidden_staging_dirs(project_root)) == 1
    assert not (datasets_root(project_root) / FIXED_DATASET_ID).exists()
    assert latest_dataset_dir(project_root) is None


@pytest.mark.parametrize("target", ["dataset.instruction.jsonl", "manifest.json"])
def test_tampered_staged_content_refuses_publish(tmp_path, monkeypatch, target):
    project_root = _learning_project(tmp_path)
    monkeypatch.setattr(dataset_module, "_dataset_id", lambda: FIXED_DATASET_ID)
    real_lock = dataset_module.portable_file_lock
    captured: dict[str, Path] = {}

    @contextmanager
    def tampering_lock(lock_path):
        with real_lock(lock_path):
            staging = _hidden_staging_dirs(project_root)[0]
            target_path = staging / target
            if target == "manifest.json":
                manifest = json.loads(target_path.read_text())
                manifest["dataset_sha256"] = "0" * 64
                target_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
                target_path.chmod(0o600)
            else:
                target_path.write_text(target_path.read_text() + "{}\n")
            captured.update(staging=staging, target=target_path)
            yield

    monkeypatch.setattr(dataset_module, "portable_file_lock", tampering_lock)

    with pytest.raises(ValueError, match="[Ss]taged dataset .* changed"):
        build_learning_dataset(project_root)

    assert captured["staging"].is_dir()
    assert captured["target"].is_file()
    assert not (datasets_root(project_root) / FIXED_DATASET_ID).exists()


def test_successful_publish_keeps_staging_inode_and_hidden_orphans_are_ignored(
    tmp_path,
    monkeypatch,
):
    project_root = _learning_project(tmp_path)
    monkeypatch.setattr(dataset_module, "_dataset_id", lambda: FIXED_DATASET_ID)
    original_create = dataset_module._create_private_staging_dir
    captured: dict[str, object] = {}

    def capture_staging(*args, **kwargs):
        staging, identity = original_create(*args, **kwargs)
        captured.update(path=staging, identity=identity)
        return staging, identity

    monkeypatch.setattr(dataset_module, "_create_private_staging_dir", capture_staging)

    result = build_learning_dataset(project_root)

    published = Path(result["dataset_dir"])
    published_stat = published.stat(follow_symlinks=False)
    assert (published_stat.st_dev, published_stat.st_ino) == captured["identity"]
    assert _private_mode(published) == 0o700
    assert all(_private_mode(path) == 0o600 for path in published.iterdir())
    assert not Path(captured["path"]).exists()
    hidden_orphan = datasets_root(project_root) / ".99991231T235959999999Z.orphan.staging"
    hidden_orphan.mkdir(mode=0o700)
    (hidden_orphan / "manifest.json").write_text("{}\n")
    assert latest_dataset_dir(project_root) == published
