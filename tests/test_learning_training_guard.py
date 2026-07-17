from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.lab import run_autonomous_lab
from morpheus.core.learning.mlx_fd_loader import read_pinned_jsonl_splits
from morpheus.core.learning.train import plan_training_run
from morpheus.core.learning import training_runtime
from morpheus.core.learning.training_guard import validate_training_run_guard
from morpheus.core.learning.training_guard import main as training_guard_main
import morpheus.core.semantic.review as review_module
from morpheus.core.semantic.review import ReviewStore
from tests.test_learning_dataset import copy_learning_project
from tests.test_learning_lab import copy_autonomous_repo


def _write_minimal_snapshot(tmp_path):
    snapshot_dir = tmp_path / "dataset"
    snapshot_dir.mkdir()
    selected_file = Path("train.jsonl")
    split_bytes = {
        split: json.dumps({
            "messages": [{"role": "user", "content": f"trusted-{split}"}],
        }).encode() + b"\n"
        for split in ("train", "valid", "test")
    }
    artifacts = {}
    for split, content in split_bytes.items():
        name = f"{split}.jsonl"
        (snapshot_dir / name).write_bytes(content)
        artifacts[name] = {
            "sha256": sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    expected_bytes = split_bytes["train"]
    manifest = {
        "selected_dataset_file": selected_file.as_posix(),
        "dataset_sha256": artifacts[selected_file.as_posix()]["sha256"],
        "artifacts": artifacts,
    }
    manifest["dataset_binding_sha256"] = training_runtime.dataset_binding_sha256(
        manifest
    )
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest))
    return snapshot_dir, manifest, selected_file, expected_bytes


def _trusted_mlx_command() -> str:
    return " ".join([
        shlex.quote(sys.executable),
        "-m",
        "morpheus.core.learning.mlx_fd_loader",
        "--model",
        "local-model",
        "--train",
        "--data",
        '"${MORPHEUS_DATASET_DIR}"',
        "--adapter-path",
        '"${MORPHEUS_OUTPUT_DIR}"',
        "--iters",
        "1",
        "--batch-size",
        "1",
        "--num-layers",
        "4",
        "--learning-rate",
        "1e-5",
        "--mask-prompt",
    ])


def _new_output_dir(path: Path) -> tuple[Path, tuple[int, int]]:
    path.mkdir(mode=0o700)
    opened = path.lstat()
    return path, (opened.st_dev, opened.st_ino)


def test_training_run_uses_guarded_dataset_snapshot(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)

    training = plan_training_run(
        project_root,
        backend="peft",
        method="lora",
        dry_run=True,
    )

    run_dir = Path(training["run_dir"])
    snapshot_dir = run_dir / "dataset"
    source_manifest = json.loads(
        (Path(dataset["dataset_dir"]) / "manifest.json").read_text()
    )
    snapshot_manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    train_config = (run_dir / "train_config.yaml").read_text()
    command = (run_dir / "command.sh").read_text()
    guard = validate_training_run_guard(
        project_root,
        Path(dataset["dataset_dir"]),
        snapshot_dir,
        source_manifest["dataset_binding_sha256"],
    )

    assert guard["valid"] is True
    assert snapshot_manifest == source_manifest
    assert all((snapshot_dir / name).is_file() for name in source_manifest["artifacts"])
    assert f"dataset_manifest_path: {snapshot_dir / 'manifest.json'}" in train_config
    assert f"dataset_path: {snapshot_dir / source_manifest['selected_dataset_file']}" in train_config
    assert "-m morpheus.core.learning.training_guard" in command
    assert command.index("morpheus.core.learning.training_guard") < command.index(
        "morpheus.learning_peft_train"
    )
    assert '--dataset "${MORPHEUS_DATASET_PATH}"' in command
    assert not (
        (snapshot_dir / source_manifest["selected_dataset_file"]).stat().st_mode
        & 0o222
    )
    assert not (Path(training["command_path"]).stat().st_mode & 0o111)


def test_generated_training_command_refuses_revoked_review_authority(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    training = plan_training_run(
        project_root,
        backend="peft",
        method="lora",
        dry_run=True,
    )
    ReviewStore(project_root).reject(
        "c_current",
        reason="revoked after training command generation",
    )

    result = subprocess.run(
        ["/bin/bash", training["command_path"]],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "Training preview only" in result.stderr
    assert "morpheus.learning_peft_train" not in result.stderr


def test_generated_training_command_refuses_tampered_snapshot(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    training = plan_training_run(
        project_root,
        backend="peft",
        method="lora",
        dry_run=True,
    )
    run_dir = Path(training["run_dir"])
    snapshot_manifest = json.loads((run_dir / "dataset/manifest.json").read_text())
    selected_path = run_dir / "dataset" / snapshot_manifest["selected_dataset_file"]
    selected_path.chmod(0o644)
    selected_path.write_text("{}\n")
    selected_path.chmod(0o444)

    result = subprocess.run(
        ["/bin/bash", training["command_path"]],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "Training preview only" in result.stderr
    assert "morpheus.learning_peft_train" not in result.stderr


def test_generated_training_command_refuses_symlinked_snapshot(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    training = plan_training_run(
        project_root,
        backend="peft",
        method="lora",
        dry_run=True,
    )
    run_dir = Path(training["run_dir"])
    snapshot_dir = run_dir / "dataset"
    moved_snapshot_dir = run_dir / "moved-dataset"
    snapshot_dir.chmod(0o755)
    snapshot_dir.rename(moved_snapshot_dir)
    snapshot_dir.symlink_to(moved_snapshot_dir, target_is_directory=True)

    result = subprocess.run(
        ["/bin/bash", training["command_path"]],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "Training preview only" in result.stderr


def test_guard_rejects_canonical_target_of_symlinked_runs_registry(tmp_path):
    project_root = copy_learning_project(tmp_path)
    dataset = build_learning_dataset(project_root)
    training = plan_training_run(
        project_root,
        backend="peft",
        method="lora",
        dry_run=True,
    )
    runs_root = project_root / ".morpheus/training/runs"
    external_runs = tmp_path / "external-runs"
    runs_root.rename(external_runs)
    runs_root.symlink_to(external_runs, target_is_directory=True)
    canonical_snapshot = (
        external_runs / Path(training["run_dir"]).name / "dataset"
    )
    source_manifest = json.loads(
        (Path(dataset["dataset_dir"]) / "manifest.json").read_text()
    )

    with pytest.raises(ValueError, match="Training runs registry must not be a symlink"):
        validate_training_run_guard(
            project_root,
            Path(dataset["dataset_dir"]),
            canonical_snapshot,
            source_manifest["dataset_binding_sha256"],
        )


def test_guard_supervisor_returns_backend_exit_code_and_cleans_fd_view(
    tmp_path,
    monkeypatch,
):
    project_root = copy_autonomous_repo(tmp_path)
    lab = run_autonomous_lab(
        project_root,
        backend="fake",
        no_train=True,
        fixture_only=True,
    )
    lab_dir = Path(lab["lab_dir"])
    dataset_dir = lab_dir / "dataset"
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    output_dir, output_identity = _new_output_dir(
        lab_dir / "training" / "adapter"
    )
    observed = {}

    def fake_run(argv, **kwargs):
        observed.update(argv=argv, cwd=Path.cwd(), kwargs=kwargs)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(training_runtime.subprocess, "run", fake_run)

    exit_code = training_guard_main([
        "--project-root",
        str(project_root),
        "--source-dataset-dir",
        str(dataset_dir),
        "--snapshot-dir",
        str(dataset_dir),
        "--expected-binding",
        manifest["dataset_binding_sha256"],
        "--backend-command",
        _trusted_mlx_command(),
        "--trusted-loader",
        training_runtime.MLX_PINNED_LOADER_CONTRACT,
        "--output-dir",
        str(output_dir),
        "--expected-output-device",
        str(output_identity[0]),
        "--expected-output-inode",
        str(output_identity[1]),
    ])

    assert exit_code == 7
    assert observed["argv"][:3] == [
        sys.executable,
        "-m",
        "morpheus.core.learning.mlx_fd_loader",
    ]
    assert not observed["cwd"].exists()
    assert list(output_dir.iterdir()) == []


def test_trusted_mlx_loader_reads_pinned_bytes_after_view_symlink_replacement(
    tmp_path,
    monkeypatch,
):
    snapshot_dir, manifest, _selected_file, expected_bytes = (
        _write_minimal_snapshot(tmp_path)
    )
    output_dir, output_identity = _new_output_dir(tmp_path / "adapter")
    expected_train_rows = [json.loads(expected_bytes)]
    attacker_rows = [{"messages": [{"role": "user", "content": "attacker"}]}]
    observed = {}
    held_view_validated = False
    real_validate_view = training_runtime._validate_held_fd_view

    def validate_held_view(snapshot):
        nonlocal held_view_validated
        real_validate_view(snapshot)
        held_view_validated = True

    def fake_run(_argv, **kwargs):
        assert held_view_validated is True
        train_path = Path("train.jsonl")
        pinned_target = os.readlink(train_path)
        train_path.unlink()
        train_path.write_text(
            "".join(json.dumps(row) + "\n" for row in attacker_rows)
        )
        try:
            observed["splits"] = read_pinned_jsonl_splits(
                kwargs["env"][training_runtime.PINNED_DATASET_FDS_ENV]
            )
            observed["path_bytes"] = train_path.read_bytes()
        finally:
            train_path.unlink()
            os.symlink(pinned_target, train_path)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        training_runtime,
        "_validate_held_fd_view",
        validate_held_view,
    )
    monkeypatch.setattr(training_runtime.subprocess, "run", fake_run)

    with training_runtime.pin_training_output_directory(
        output_dir,
        output_identity,
    ) as output_descriptor:
        with training_runtime.pin_dataset_snapshot(
            snapshot_dir,
            manifest["dataset_binding_sha256"],
            view_parent_descriptor=output_descriptor,
            view_parent_path=output_dir,
            expected_view_parent_identity=output_identity,
        ) as pinned:
            returncode = training_runtime.supervise_training_backend(
                _trusted_mlx_command(),
                pinned,
                trusted_loader=training_runtime.MLX_PINNED_LOADER_CONTRACT,
                output_descriptor=output_descriptor,
            )

    assert returncode == 0
    assert observed["path_bytes"] != expected_bytes
    assert observed["splits"]["train.jsonl"] == expected_train_rows
    assert observed["splits"]["train.jsonl"] != attacker_rows


def test_backend_cwd_pins_fd_view_after_view_root_replacement(
    tmp_path,
    monkeypatch,
):
    snapshot_dir, manifest, selected_file, expected_bytes = _write_minimal_snapshot(
        tmp_path
    )
    output_dir, output_identity = _new_output_dir(tmp_path / "adapter")
    observed = {}

    def fake_run(_argv, **kwargs):
        observed["cwd"] = Path.cwd()
        observed["splits"] = read_pinned_jsonl_splits(
            kwargs["env"][training_runtime.PINNED_DATASET_FDS_ENV]
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(training_runtime.subprocess, "run", fake_run)

    with training_runtime.pin_training_output_directory(
        output_dir,
        output_identity,
    ) as output_descriptor:
        with training_runtime.pin_dataset_snapshot(
            snapshot_dir,
            manifest["dataset_binding_sha256"],
            view_parent_descriptor=output_descriptor,
            view_parent_path=output_dir,
            expected_view_parent_identity=output_identity,
        ) as pinned:
            view_dir = pinned.view_dir
            moved_view = view_dir.with_name(view_dir.name + "-moved")
            view_dir.rename(moved_view)
            view_dir.mkdir(mode=0o700)
            replacement_file = view_dir / selected_file
            replacement_file.parent.mkdir(parents=True, exist_ok=True)
            replacement_file.write_text('{"attacker": true}\n')
            try:
                returncode = training_runtime.supervise_training_backend(
                    _trusted_mlx_command(),
                    pinned,
                    trusted_loader=training_runtime.MLX_PINNED_LOADER_CONTRACT,
                    output_descriptor=output_descriptor,
                )
            finally:
                shutil.rmtree(view_dir)
                moved_view.rename(view_dir)

    assert returncode == 0
    assert observed["cwd"] == moved_view
    assert observed["splits"]["train.jsonl"] == [json.loads(expected_bytes)]


def test_backend_refuses_changed_held_fd_view_layout(tmp_path, monkeypatch):
    snapshot_dir, manifest, _selected_file, _expected_bytes = (
        _write_minimal_snapshot(tmp_path)
    )
    backend_record = tmp_path / "backend-ran.txt"
    output_dir, output_identity = _new_output_dir(tmp_path / "adapter")

    def fake_run(*_args, **_kwargs):
        backend_record.write_text("backend ran\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(training_runtime.subprocess, "run", fake_run)

    with training_runtime.pin_training_output_directory(
        output_dir,
        output_identity,
    ) as output_descriptor:
        with training_runtime.pin_dataset_snapshot(
            snapshot_dir,
            manifest["dataset_binding_sha256"],
            view_parent_descriptor=output_descriptor,
            view_parent_path=output_dir,
            expected_view_parent_identity=output_identity,
        ) as pinned:
            injected = pinned.view_dir / "unexpected.txt"
            injected.write_text("untrusted\n")
            try:
                with pytest.raises(ValueError, match="FD view layout changed"):
                    training_runtime.supervise_training_backend(
                        _trusted_mlx_command(),
                        pinned,
                        trusted_loader=training_runtime.MLX_PINNED_LOADER_CONTRACT,
                        output_descriptor=output_descriptor,
                    )
            finally:
                injected.unlink()

    assert not backend_record.exists()


def test_output_bound_view_reads_and_writes_through_held_topology(
    tmp_path,
    monkeypatch,
):
    snapshot_dir, manifest, _selected_file, expected_bytes = _write_minimal_snapshot(
        tmp_path
    )
    output_dir, output_identity = _new_output_dir(tmp_path / "adapter")
    moved_output = tmp_path / "moved-adapter"
    external_output = tmp_path / "external-adapter"
    external_output.mkdir()

    def fake_run(_argv, **kwargs):
        splits = read_pinned_jsonl_splits(
            kwargs["env"][training_runtime.PINNED_DATASET_FDS_ENV]
        )
        output_path = Path(kwargs["env"][training_runtime.RUNTIME_OUTPUT_DIR_ENV])
        (output_path / "artifact.txt").write_text(json.dumps(splits["train.jsonl"]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(training_runtime.subprocess, "run", fake_run)

    with training_runtime.pin_training_output_directory(
        output_dir,
        output_identity,
    ) as output_descriptor:
        try:
            snapshot_context = training_runtime.pin_dataset_snapshot(
                snapshot_dir,
                manifest["dataset_binding_sha256"],
                view_parent_descriptor=output_descriptor,
                view_parent_path=output_dir,
                expected_view_parent_identity=output_identity,
            )
        except TypeError as exc:
            pytest.fail(f"output-bound FD view is unsupported: {exc}")
        with snapshot_context as pinned:
            assert pinned.view_dir.parent == output_dir
            output_dir.rename(moved_output)
            output_dir.symlink_to(external_output, target_is_directory=True)
            try:
                returncode = training_runtime.supervise_training_backend(
                    _trusted_mlx_command(),
                    pinned,
                    trusted_loader=training_runtime.MLX_PINNED_LOADER_CONTRACT,
                    output_descriptor=output_descriptor,
                )
                written_rows = json.loads(
                    (moved_output / "artifact.txt").read_text()
                )
                assert written_rows == [json.loads(expected_bytes)]
                assert not (external_output / "artifact.txt").exists()
            finally:
                output_dir.unlink(missing_ok=True)
                moved_output.rename(output_dir)

    assert returncode == 0
    assert json.loads((output_dir / "artifact.txt").read_text()) == [
        json.loads(expected_bytes)
    ]


def test_backend_refuses_view_reparented_away_from_pinned_output(tmp_path):
    snapshot_dir, manifest, _selected_file, _expected_bytes = (
        _write_minimal_snapshot(tmp_path)
    )
    output_dir, output_identity = _new_output_dir(tmp_path / "adapter")
    foreign_parent = tmp_path / "foreign-parent"
    foreign_parent.mkdir()
    backend_record = tmp_path / "backend-ran.txt"

    with training_runtime.pin_training_output_directory(
        output_dir,
        output_identity,
    ) as output_descriptor:
        with training_runtime.pin_dataset_snapshot(
            snapshot_dir,
            manifest["dataset_binding_sha256"],
            view_parent_descriptor=output_descriptor,
            view_parent_path=output_dir,
            expected_view_parent_identity=output_identity,
        ) as pinned:
            reparented_view = foreign_parent / pinned.view_dir.name
            pinned.view_dir.rename(reparented_view)
            try:
                with pytest.raises(ValueError, match="FD view parent identity changed"):
                    training_runtime.supervise_training_backend(
                        _trusted_mlx_command(),
                        pinned,
                        trusted_loader=training_runtime.MLX_PINNED_LOADER_CONTRACT,
                        output_descriptor=output_descriptor,
                    )
            finally:
                reparented_view.rename(pinned.view_dir)

    assert not backend_record.exists()


@pytest.mark.parametrize(
    "backend_command",
    [
        "touch backend-marker.txt",
        " ".join([
            shlex.quote(sys.executable),
            "-m",
            "forged.mlx_fd_loader",
            "--train",
            "--data",
            '"${MORPHEUS_DATASET_DIR}"',
            "--adapter-path",
            '"${MORPHEUS_OUTPUT_DIR}"',
        ]),
        _trusted_mlx_command() + " --config=/tmp/untrusted.yaml",
    ],
    ids=["generic-shell", "forged-loader", "forbidden-option"],
)
def test_supervisor_rejects_unauthenticated_backend_command(
    tmp_path,
    monkeypatch,
    backend_command,
):
    snapshot_dir, manifest, _selected_file, _expected_bytes = (
        _write_minimal_snapshot(tmp_path)
    )
    output_dir, output_identity = _new_output_dir(tmp_path / "adapter")
    backend_marker = tmp_path / "backend-ran.txt"

    def fake_run(*_args, **_kwargs):
        backend_marker.write_text("backend ran\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(training_runtime.subprocess, "run", fake_run)

    with training_runtime.pin_training_output_directory(
        output_dir,
        output_identity,
    ) as output_descriptor:
        with training_runtime.pin_dataset_snapshot(
            snapshot_dir,
            manifest["dataset_binding_sha256"],
            view_parent_descriptor=output_descriptor,
            view_parent_path=output_dir,
            expected_view_parent_identity=output_identity,
        ) as pinned:
            with pytest.raises(
                ValueError,
                match="loader identity mismatch|option is not allowed",
            ):
                training_runtime.supervise_training_backend(
                    backend_command,
                    pinned,
                    trusted_loader=training_runtime.MLX_PINNED_LOADER_CONTRACT,
                    output_descriptor=output_descriptor,
                )

    assert not backend_marker.exists()


def test_pinned_dataset_cleanup_refuses_replaced_view(tmp_path, monkeypatch):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    training = plan_training_run(project_root, dry_run=True)
    snapshot_dir = Path(training["dataset_snapshot_dir"])
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    view_dir = tmp_path / "fd-view"

    def make_private_view(*, prefix):
        assert prefix == "morpheus-fd-dataset-"
        view_dir.mkdir()
        return str(view_dir)

    monkeypatch.setattr(training_runtime.tempfile, "mkdtemp", make_private_view)

    pinned_descriptors = ()
    moved_links = []
    with pytest.raises(ValueError, match="replaced training FD view"):
        with training_runtime.pin_dataset_snapshot(
            snapshot_dir,
            manifest["dataset_binding_sha256"],
        ) as pinned:
            pinned_descriptors = pinned.file_descriptors
            moved_view = tmp_path / "moved-fd-view"
            pinned.view_dir.rename(moved_view)
            moved_links = [
                path for path in moved_view.rglob("*") if path.is_symlink()
            ]
            pinned.view_dir.write_text("replacement must survive cleanup\n")

    assert view_dir.read_text() == "replacement must survive cleanup\n"
    for descriptor in pinned_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert moved_links
    assert all(path.is_symlink() and not path.exists() for path in moved_links)


def test_pinned_dataset_cleanup_refuses_unknown_populated_directory(tmp_path):
    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    training = plan_training_run(project_root, dry_run=True)
    snapshot_dir = Path(training["dataset_snapshot_dir"])
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    external_dir = tmp_path / "external-content"
    external_dir.mkdir()
    (external_dir / "keep.txt").write_text("must survive cleanup\n")
    injected_dir = None
    pinned_descriptors = ()

    with pytest.raises(ValueError, match="layout changed"):
        with training_runtime.pin_dataset_snapshot(
            snapshot_dir,
            manifest["dataset_binding_sha256"],
        ) as pinned:
            pinned_descriptors = pinned.file_descriptors
            injected_dir = pinned.view_dir / "unexpected-directory"
            external_dir.rename(injected_dir)

    assert injected_dir is not None
    assert (injected_dir / "keep.txt").read_text() == "must survive cleanup\n"
    for descriptor in pinned_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_supervisor_reports_unsupported_training_runtime(tmp_path, monkeypatch):
    snapshot_dir, manifest, _selected_file, _expected_bytes = (
        _write_minimal_snapshot(tmp_path)
    )
    output_dir, output_identity = _new_output_dir(tmp_path / "adapter")

    with training_runtime.pin_training_output_directory(
        output_dir,
        output_identity,
    ) as output_descriptor:
        with training_runtime.pin_dataset_snapshot(
            snapshot_dir,
            manifest["dataset_binding_sha256"],
            view_parent_descriptor=output_descriptor,
            view_parent_path=output_dir,
            expected_view_parent_identity=output_identity,
        ) as pinned:
            monkeypatch.setattr(training_runtime, "_is_posix_runtime", lambda: False)
            with pytest.raises(
                ValueError,
                match="Training runtime unsupported: POSIX is required",
            ):
                training_runtime.supervise_training_backend(
                    _trusted_mlx_command(),
                    pinned,
                    trusted_loader=training_runtime.MLX_PINNED_LOADER_CONTRACT,
                    output_descriptor=output_descriptor,
                )


def test_review_rejection_waits_for_complete_training_authority_lease(
    tmp_path,
    monkeypatch,
):
    project_root = copy_autonomous_repo(tmp_path)
    lab = run_autonomous_lab(
        project_root,
        backend="fake",
        no_train=True,
        fixture_only=True,
    )
    lab_dir = Path(lab["lab_dir"])
    dataset_dir = lab_dir / "dataset"
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    output_dir, output_identity = _new_output_dir(
        lab_dir / "training" / "adapter"
    )
    store = ReviewStore(lab_dir / "workspace")
    candidate_id = next(
        candidate.id
        for candidate in store.load_candidates()
        if candidate.status == "accepted"
    )
    backend_started = threading.Event()
    allow_backend_finish = threading.Event()
    backend_completed = threading.Event()
    rejection_lock_attempted = threading.Event()
    rejection_lock_acquired = threading.Event()
    rejection_completed = threading.Event()
    real_review_lock = review_module.portable_file_lock
    failures = []
    training_result = {}
    rejection_thread = None

    @contextmanager
    def observed_review_lock(path):
        is_rejection = threading.current_thread() is rejection_thread
        if is_rejection:
            rejection_lock_attempted.set()
        with real_review_lock(path):
            if is_rejection:
                rejection_lock_acquired.set()
                assert backend_completed.is_set()
            yield

    def fake_run(_argv, **_kwargs):
        backend_started.set()
        if not allow_backend_finish.wait(timeout=5):
            raise AssertionError("backend completion was never released")
        backend_completed.set()
        return SimpleNamespace(returncode=0)

    guard_arguments = [
        "--project-root",
        str(project_root),
        "--source-dataset-dir",
        str(dataset_dir),
        "--snapshot-dir",
        str(dataset_dir),
        "--expected-binding",
        manifest["dataset_binding_sha256"],
        "--backend-command",
        _trusted_mlx_command(),
        "--trusted-loader",
        training_runtime.MLX_PINNED_LOADER_CONTRACT,
        "--output-dir",
        str(output_dir),
        "--expected-output-device",
        str(output_identity[0]),
        "--expected-output-inode",
        str(output_identity[1]),
    ]

    def run_training():
        try:
            training_result["exit_code"] = training_guard_main(guard_arguments)
        except BaseException as exc:  # pragma: no cover - asserted below.
            failures.append(exc)

    def reject_candidate():
        try:
            store.reject(candidate_id, reason="rejected during guarded training")
            rejection_completed.set()
        except BaseException as exc:  # pragma: no cover - asserted below.
            failures.append(exc)

    monkeypatch.setattr(review_module, "portable_file_lock", observed_review_lock)
    monkeypatch.setattr(training_runtime.subprocess, "run", fake_run)
    training_thread = threading.Thread(target=run_training, name="guarded-training")
    training_thread.start()
    assert backend_started.wait(timeout=5)
    rejection_thread = threading.Thread(
        target=reject_candidate,
        name="concurrent-review-rejection",
    )
    rejection_thread.start()
    try:
        assert rejection_lock_attempted.wait(timeout=5)
        assert not rejection_lock_acquired.is_set()
        assert not rejection_completed.is_set()
    finally:
        allow_backend_finish.set()
    training_thread.join(timeout=5)
    rejection_thread.join(timeout=5)

    assert not training_thread.is_alive()
    assert not rejection_thread.is_alive()
    assert failures == []
    assert training_result["exit_code"] == 0
    assert rejection_lock_acquired.is_set()
    assert rejection_completed.is_set()
    rejected = next(
        candidate for candidate in store.load_candidates() if candidate.id == candidate_id
    )
    assert rejected.status == "rejected"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux memfd sealing contract",
)
def test_linux_pinned_memfd_seals_block_proc_reopen_and_truncate(tmp_path):
    import fcntl

    project_root = copy_learning_project(tmp_path)
    build_learning_dataset(project_root)
    training = plan_training_run(project_root, dry_run=True)
    snapshot_dir = Path(training["dataset_snapshot_dir"])
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    expected_bytes = (
        snapshot_dir / manifest["selected_dataset_file"]
    ).read_bytes()

    with training_runtime.pin_dataset_snapshot(
        snapshot_dir,
        manifest["dataset_binding_sha256"],
    ) as pinned:
        selected_descriptor = int(pinned.selected_path.name)
        required_seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_SEAL
        )
        assert (
            fcntl.fcntl(selected_descriptor, fcntl.F_GET_SEALS)
            & required_seals
        ) == required_seals
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os,sys; p=sys.argv[1]; "
                    "\ntry:\n fd=os.open(p, os.O_WRONLY | os.O_TRUNC); "
                    "os.write(fd, b'tampered'); os.close(fd)"
                    "\nexcept OSError:\n raise SystemExit(0)"
                    "\nraise SystemExit(91)"
                ),
                str(pinned.selected_path),
            ],
            check=False,
            pass_fds=(selected_descriptor,),
        )
        assert completed.returncode == 0
        assert os.pread(selected_descriptor, len(expected_bytes), 0) == expected_bytes


@pytest.mark.parametrize("authority_loss", ["review_revocation", "artifact_tamper"])
def test_lab_training_command_rechecks_preserved_authority(
    tmp_path,
    authority_loss,
):
    project_root = copy_autonomous_repo(tmp_path)
    lab = run_autonomous_lab(
        project_root,
        backend="fake",
        no_train=True,
        fixture_only=True,
    )
    lab_dir = Path(lab["lab_dir"])
    if authority_loss == "review_revocation":
        store = ReviewStore(lab_dir / "workspace")
        candidates = store.load_candidates()
        candidates[0] = candidates[0].model_copy(update={"status": "pending"})
        store.save_candidates(candidates)
    else:
        manifest = json.loads((lab_dir / "dataset/manifest.json").read_text())
        selected_path = lab_dir / "dataset" / manifest["selected_dataset_file"]
        selected_path.chmod(0o644)
        selected_path.write_bytes(selected_path.read_bytes() + b"{}\n")
        selected_path.chmod(0o444)

    result = subprocess.run(
        [lab_dir / "training/train_command.sh"],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "Training guard failed" in result.stderr
    expected = (
        "review_snapshot_changed"
        if authority_loss == "review_revocation"
        else "dataset_artifact_hash_mismatch"
    )
    assert expected in result.stderr
