"""Shared helpers for sealed, guarded learning-training execution."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import BinaryIO, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - rejected by the runtime capability gate
    fcntl = None

from morpheus.core.learning.dataset_validation import dataset_binding_sha256
from morpheus.core.safe_io import reject_symlink_components, reject_symlink_paths


RUNTIME_DATASET_DIR_ENV = "MORPHEUS_DATASET_DIR"
RUNTIME_DATASET_PATH_ENV = "MORPHEUS_DATASET_PATH"
RUNTIME_OUTPUT_DIR_ENV = "MORPHEUS_OUTPUT_DIR"
PINNED_DATASET_FDS_ENV = "MORPHEUS_PINNED_DATASET_FDS"
MLX_PINNED_LOADER_CONTRACT = "mlx-pinned-fd-v1"
_COPY_CHUNK_SIZE = 1024 * 1024
_BASH_PATH = Path("/bin/bash")
INDEPENDENT_OPEN_CONTRACT = "independent-open"
SINGLE_OPEN_CONTRACT = "single-open"


@dataclass(frozen=True)
class RuntimeDatasetArgument:
    """Typed marker for one Morpheus-owned runtime environment expansion."""

    environment_name: str


RUNTIME_DATASET_DIR_PLACEHOLDER = RuntimeDatasetArgument(RUNTIME_DATASET_DIR_ENV)
RUNTIME_DATASET_PATH_PLACEHOLDER = RuntimeDatasetArgument(RUNTIME_DATASET_PATH_ENV)
RUNTIME_OUTPUT_DIR_PLACEHOLDER = RuntimeDatasetArgument(RUNTIME_OUTPUT_DIR_ENV)


@dataclass(frozen=True)
class PinnedDatasetSnapshot:
    """Anonymous validated dataset files exposed through one private FD view."""

    view_dir: Path
    selected_path: Path
    view_descriptor: int
    file_descriptors: tuple[int, ...]
    artifact_descriptors: tuple[tuple[str, int, int, str], ...]
    descriptor_open_contract: str
    _view_identity: tuple[int, int] = field(repr=False)
    _view_layout: "_FDViewLayout" = field(repr=False)
    _expected_parent_identity: tuple[int, int] | None = field(repr=False)
    _output_bound: bool = field(repr=False)
    _files: tuple[BinaryIO, ...] = field(repr=False)

    def environment(self) -> dict[str, str]:
        return {
            RUNTIME_DATASET_DIR_ENV: ".",
            RUNTIME_DATASET_PATH_ENV: str(self.selected_path),
            PINNED_DATASET_FDS_ENV: json.dumps(
                {
                    path: {
                        "descriptor": descriptor,
                        "size_bytes": size_bytes,
                        "sha256": artifact_sha256,
                    }
                    for path, descriptor, size_bytes, artifact_sha256
                    in self.artifact_descriptors
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }


@dataclass(frozen=True)
class _FDViewLayout:
    directory_identities: tuple[tuple[str, int, int], ...]
    symlink_targets: tuple[tuple[str, str], ...]


def render_guarded_training_command(
    backend_command: str,
    *,
    project_root: Path,
    source_dataset_dir: Path,
    snapshot_dir: Path,
    expected_binding_sha256: str,
    trusted_loader: str | None = None,
    output_dir: Path | None = None,
    expected_output_identity: tuple[int, int] | None = None,
) -> str:
    """Render one script whose guard supervises an FD-backed backend process."""
    guard_args = [
        sys.executable,
        "-m",
        "morpheus.core.learning.training_guard",
        "--project-root",
        str(project_root),
        "--source-dataset-dir",
        str(source_dataset_dir),
        "--snapshot-dir",
        str(snapshot_dir),
        "--expected-binding",
        expected_binding_sha256,
        "--backend-command",
        backend_command,
    ]
    if trusted_loader is not None:
        guard_args.extend(["--trusted-loader", trusted_loader])
    if output_dir is not None or expected_output_identity is not None:
        if output_dir is None or expected_output_identity is None:
            raise ValueError("Guarded training output identity is incomplete")
        guard_args.extend([
            "--output-dir",
            str(output_dir),
            "--expected-output-device",
            str(expected_output_identity[0]),
            "--expected-output-inode",
            str(expected_output_identity[1]),
        ])
    guard_line = " ".join(shlex.quote(argument) for argument in guard_args)
    return "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"exec {guard_line}",
        "",
    ])


def shell_quote_training_argument(argument: object) -> str:
    """Quote one backend argument, expanding only Morpheus-owned FD variables."""
    if argument is RUNTIME_DATASET_DIR_PLACEHOLDER:
        return f'"${{{RUNTIME_DATASET_DIR_ENV}}}"'
    if argument is RUNTIME_DATASET_PATH_PLACEHOLDER:
        return f'"${{{RUNTIME_DATASET_PATH_ENV}}}"'
    if argument is RUNTIME_OUTPUT_DIR_PLACEHOLDER:
        return f'"${{{RUNTIME_OUTPUT_DIR_ENV}}}"'
    return shlex.quote(str(argument))


@contextmanager
def pin_dataset_snapshot(
    snapshot_dir: Path,
    expected_binding_sha256: str,
    *,
    view_parent_descriptor: int | None = None,
    view_parent_path: Path | None = None,
    expected_view_parent_identity: tuple[int, int] | None = None,
) -> Iterator[PinnedDatasetSnapshot]:
    """Copy manifest-bound bytes into anonymous files and expose an FD view."""
    parent_arguments = (
        view_parent_descriptor,
        view_parent_path,
        expected_view_parent_identity,
    )
    if any(argument is not None for argument in parent_arguments) and any(
        argument is None for argument in parent_arguments
    ):
        raise ValueError("Training FD view parent identity is incomplete")
    output_bound = view_parent_descriptor is not None
    descriptor_root, open_contract = require_training_runtime_support()
    snapshot_dir = _safe_directory(snapshot_dir, "Training dataset snapshot")
    root_descriptor = _open_directory(snapshot_dir)
    pinned_files: list[BinaryIO] = []
    view_dir: Path | None = None
    view_descriptor: int | None = None
    view_identity: tuple[int, int] | None = None
    view_directories: dict[Path, tuple[int, int]] = {}
    view_symlinks: dict[Path, str] = {}
    try:
        manifest_file, manifest_sha, manifest_bytes = _pin_relative_file(
            root_descriptor,
            Path("manifest.json"),
            collect_bytes=True,
        )
        pinned_files.append(manifest_file)
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Training dataset snapshot manifest invalid: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ValueError("Training dataset snapshot manifest must be a JSON object")
        if (
            manifest.get("dataset_binding_sha256") != expected_binding_sha256
            or dataset_binding_sha256(manifest) != expected_binding_sha256
        ):
            raise ValueError("Training dataset snapshot binding mismatch")

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("Training dataset snapshot artifacts invalid")
        selected_file = manifest.get("selected_dataset_file")
        if not isinstance(selected_file, str) or selected_file not in artifacts:
            raise ValueError("Training dataset snapshot selected artifact invalid")

        pinned_by_path: dict[Path, BinaryIO] = {
            Path("manifest.json"): manifest_file,
        }
        artifact_hashes = {"manifest.json": manifest_sha}
        artifact_descriptors = []
        for raw_path, metadata in sorted(artifacts.items()):
            relative = _safe_relative_path(raw_path)
            if relative == Path("manifest.json"):
                raise ValueError("Training dataset snapshot artifacts invalid")
            if not isinstance(metadata, dict) or not _valid_sha256(
                metadata.get("sha256")
            ):
                raise ValueError(f"Training dataset artifact metadata invalid: {raw_path}")
            pinned_file, actual_sha, _ = _pin_relative_file(
                root_descriptor,
                relative,
                collect_bytes=False,
            )
            pinned_files.append(pinned_file)
            pinned_by_path[relative] = pinned_file
            artifact_hashes[relative.as_posix()] = actual_sha
            if actual_sha != metadata["sha256"]:
                raise ValueError(
                    "Training dataset snapshot validation failed: "
                    f"dataset_artifact_hash_mismatch: {raw_path}"
                )
            opened_artifact = os.fstat(pinned_file.fileno())
            expected_size = metadata.get("size_bytes")
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size < 0
                or opened_artifact.st_size != expected_size
            ):
                raise ValueError(
                    "Training dataset snapshot validation failed: "
                    f"dataset_artifact_size_mismatch: {raw_path}"
                )
            artifact_descriptors.append((
                relative.as_posix(),
                pinned_file.fileno(),
                expected_size,
                actual_sha,
            ))

        if output_bound:
            parent_opened = os.fstat(view_parent_descriptor)
            if (
                not stat.S_ISDIR(parent_opened.st_mode)
                or (parent_opened.st_dev, parent_opened.st_ino)
                != expected_view_parent_identity
            ):
                raise ValueError("Training FD view parent identity mismatch")
            view_name, view_descriptor, view_identity = _create_private_fd_view(
                view_parent_descriptor
            )
            view_dir = view_parent_path / view_name
        else:
            view_dir = Path(tempfile.mkdtemp(prefix="morpheus-fd-dataset-"))
            view_dir.chmod(0o700)
            view_stat = view_dir.lstat()
            view_identity = (view_stat.st_dev, view_stat.st_ino)
            view_descriptor = _open_directory(view_dir)
        for relative, pinned_file in pinned_by_path.items():
            destination_parent = _ensure_fd_view_parent(
                view_descriptor,
                relative.parent,
                view_directories,
            )
            try:
                descriptor_path = _descriptor_path(
                    pinned_file.fileno(),
                    descriptor_root,
                )
                os.symlink(
                    descriptor_path,
                    relative.name,
                    dir_fd=destination_parent,
                )
                view_symlinks[relative] = str(descriptor_path)
            finally:
                os.close(destination_parent)
        selected_pinned_file = pinned_by_path[_safe_relative_path(selected_file)]
        selected_path = _descriptor_path(
            selected_pinned_file.fileno(),
            descriptor_root,
        )
        if artifact_hashes.get(selected_file) != manifest.get("dataset_sha256"):
            raise ValueError("Training dataset snapshot selected artifact hash mismatch")

        view_layout = _fd_view_layout(view_directories, view_symlinks)
        opened_view = os.fstat(view_descriptor)
        if (opened_view.st_dev, opened_view.st_ino) != view_identity:
            raise ValueError("Training FD view descriptor identity changed")
        _validate_fd_view_descriptor(view_descriptor, Path("."), view_layout)
        parent_stat = os.stat("..", dir_fd=view_descriptor, follow_symlinks=False)
        parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        if output_bound and parent_identity != expected_view_parent_identity:
            raise ValueError("Training FD view parent identity mismatch")
        snapshot = PinnedDatasetSnapshot(
            view_dir=view_dir,
            selected_path=selected_path,
            view_descriptor=view_descriptor,
            file_descriptors=(
                *(item.fileno() for item in pinned_files),
                view_descriptor,
            ),
            artifact_descriptors=tuple(artifact_descriptors),
            descriptor_open_contract=open_contract,
            _view_identity=view_identity,
            _view_layout=view_layout,
            _expected_parent_identity=parent_identity,
            _output_bound=output_bound,
            _files=tuple(pinned_files),
        )
        yield snapshot
    finally:
        try:
            os.close(root_descriptor)
        finally:
            try:
                try:
                    if view_dir is not None and view_identity is not None:
                        _remove_fd_view(
                            view_dir,
                            view_identity,
                            _fd_view_layout(view_directories, view_symlinks),
                        )
                finally:
                    if view_descriptor is not None:
                        os.close(view_descriptor)
            finally:
                _close_pinned_files(pinned_files)


def supervise_training_backend(
    backend_command: str,
    snapshot: PinnedDatasetSnapshot,
    *,
    trusted_loader: str | None = None,
    output_descriptor: int | None = None,
) -> int:
    """Run the authenticated MLX loader while retaining anonymous FDs."""
    require_training_runtime_support()
    if trusted_loader != MLX_PINNED_LOADER_CONTRACT:
        raise ValueError(
            "Training runtime requires the trusted pinned-FD MLX loader"
        )
    environment = os.environ.copy()
    for inherited_name in (
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
    ):
        environment.pop(inherited_name, None)
    environment.pop(RUNTIME_OUTPUT_DIR_ENV, None)
    environment.update(snapshot.environment())
    if output_descriptor is not None:
        if not snapshot._output_bound:
            raise ValueError("Training output is not the pinned FD view parent")
        opened_output = os.fstat(output_descriptor)
        if (
            snapshot._expected_parent_identity is None
            or (opened_output.st_dev, opened_output.st_ino)
            != snapshot._expected_parent_identity
        ):
            raise ValueError("Training output descriptor identity mismatch")
        environment[RUNTIME_OUTPUT_DIR_ENV] = ".."
    elif snapshot._output_bound:
        raise ValueError("Pinned training output descriptor is required")
    backend_argv = _trusted_mlx_backend_argv(
        backend_command,
        output_bound=output_descriptor is not None,
    )
    _validate_held_fd_view(snapshot)
    original_directory = _open_directory(Path("."))
    try:
        os.fchdir(snapshot.view_descriptor)
        completed = subprocess.run(
            backend_argv,
            check=False,
            env=environment,
            pass_fds=snapshot.file_descriptors,
        )
    finally:
        try:
            os.fchdir(original_directory)
        finally:
            os.close(original_directory)
    if completed.returncode < 0:
        return 128 - completed.returncode
    return completed.returncode


def _trusted_mlx_backend_argv(
    backend_command: str,
    *,
    output_bound: bool,
) -> list[str]:
    """Parse and authenticate the only executable training command."""
    try:
        arguments = shlex.split(backend_command)
    except ValueError as exc:
        raise ValueError(f"Trusted MLX training command invalid: {exc}") from exc
    expected_prefix = [
        sys.executable,
        "-m",
        "morpheus.core.learning.mlx_fd_loader",
    ]
    if arguments[:3] != expected_prefix:
        raise ValueError("Trusted MLX training loader identity mismatch")
    arguments = [
        "."
        if argument == f"${{{RUNTIME_DATASET_DIR_ENV}}}"
        else ".."
        if argument == f"${{{RUNTIME_OUTPUT_DIR_ENV}}}"
        else argument
        for argument in arguments
    ]
    _validate_exact_mlx_options(arguments[3:])
    _require_exact_option(arguments, "--data", ".")
    _require_exact_option(
        arguments,
        "--adapter-path",
        ".." if output_bound else None,
    )
    return arguments


def _validate_exact_mlx_options(arguments: list[str]) -> None:
    value_options = {
        "--adapter-path",
        "--batch-size",
        "--data",
        "--iters",
        "--learning-rate",
        "--model",
        "--num-layers",
    }
    flag_options = {"--mask-prompt", "--train"}
    seen = set()
    index = 0
    while index < len(arguments):
        option = arguments[index]
        if option in flag_options:
            if option in seen:
                raise ValueError(f"Trusted MLX training option repeated: {option}")
            seen.add(option)
            index += 1
            continue
        if option not in value_options or index + 1 >= len(arguments):
            raise ValueError(f"Trusted MLX training option is not allowed: {option}")
        if option in seen:
            raise ValueError(f"Trusted MLX training option repeated: {option}")
        value = arguments[index + 1]
        if not value or value.startswith("--") or "\x00" in value:
            raise ValueError(f"Trusted MLX training option value invalid: {option}")
        seen.add(option)
        index += 2
    expected = value_options | flag_options
    if seen != expected:
        missing = ", ".join(sorted(expected - seen))
        raise ValueError(f"Trusted MLX training options incomplete: {missing}")


def _require_exact_option(
    arguments: list[str],
    option: str,
    expected_value: str | None,
) -> None:
    indices = [index for index, argument in enumerate(arguments) if argument == option]
    if len(indices) != 1 or indices[0] + 1 >= len(arguments):
        raise ValueError(f"Trusted MLX training command requires {option}")
    if expected_value is None or arguments[indices[0] + 1] != expected_value:
        raise ValueError(f"Trusted MLX training command has invalid {option}")


@contextmanager
def pin_training_output_directory(
    output_dir: Path,
    expected_identity: tuple[int, int],
) -> Iterator[int]:
    """Hold one empty output inode and reject pathname replacement."""
    raw_output_dir = output_dir.expanduser()
    if raw_output_dir.is_symlink():
        raise ValueError(f"Training output directory must not be a symlink: {output_dir}")
    reject_symlink_components(raw_output_dir, "Training output directory")
    parent = _safe_directory(raw_output_dir.parent, "Training output parent")
    parent_descriptor = _open_directory(parent)
    output_descriptor: int | None = None
    try:
        try:
            output_descriptor = _open_directory_at(
                parent_descriptor,
                raw_output_dir.name,
            )
        except OSError as exc:
            raise ValueError(
                "Training output directory changed before execution"
            ) from exc
        opened = os.fstat(output_descriptor)
        if (opened.st_dev, opened.st_ino) != expected_identity:
            raise ValueError("Training output directory identity mismatch")
        if os.listdir(output_descriptor):
            raise ValueError("Training output directory must be empty")
        yield output_descriptor
        current = os.stat(
            raw_output_dir.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != expected_identity
        ):
            raise ValueError("Training output directory changed during execution")
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
        os.close(parent_descriptor)


def seal_dataset_snapshot(snapshot_dir: Path) -> None:
    """Make a validated private snapshot read-only for ordinary processes."""
    snapshot_dir = _safe_directory(snapshot_dir, "Training dataset snapshot")
    paths = sorted(
        snapshot_dir.rglob("*"),
        key=lambda path: (len(path.parts), path.as_posix()),
        reverse=True,
    )
    for path in paths:
        reject_symlink_paths([path], "Training dataset snapshot")
        reject_symlink_components(path, "Training dataset snapshot")
        path.chmod(0o555 if path.is_dir() else 0o444)
    snapshot_dir.chmod(0o555)


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


def _pin_relative_file(
    root_descriptor: int,
    relative: Path,
    *,
    collect_bytes: bool,
) -> tuple[BinaryIO, str, bytes]:
    source_descriptor = _open_relative_file(root_descriptor, relative)
    use_sealed_memfd = _uses_sealed_memfd()
    if use_sealed_memfd:
        memfd_flags = os.MFD_ALLOW_SEALING
        if hasattr(os, "MFD_CLOEXEC"):
            memfd_flags |= os.MFD_CLOEXEC
        writer = os.fdopen(
            os.memfd_create("morpheus-pinned-artifact", memfd_flags),
            "w+b",
        )
        writer_path = None
    else:
        writer = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix="morpheus-pinned-artifact-",
            delete=False,
        )
        writer_path = Path(writer.name)
    writer_stat = os.fstat(writer.fileno())
    readonly_descriptor: int | None = None
    writer_unlinked = False
    digest = sha256()
    collected = bytearray()
    try:
        while True:
            chunk = os.read(source_descriptor, _COPY_CHUNK_SIZE)
            if not chunk:
                break
            writer.write(chunk)
            digest.update(chunk)
            if collect_bytes:
                collected.extend(chunk)
        writer.flush()
        os.fsync(writer.fileno())
        if use_sealed_memfd:
            _seal_memfd(writer.fileno())
            readonly_descriptor = os.open(
                Path("/proc/self/fd") / str(writer.fileno()),
                os.O_RDONLY,
            )
        else:
            readonly_flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                readonly_flags |= os.O_NOFOLLOW
            readonly_descriptor = os.open(writer_path, readonly_flags)
        readonly_stat = os.fstat(readonly_descriptor)
        if (
            not stat.S_ISREG(readonly_stat.st_mode)
            or (readonly_stat.st_dev, readonly_stat.st_ino)
            != (writer_stat.st_dev, writer_stat.st_ino)
        ):
            raise ValueError("Pinned training dataset artifact identity changed")
        if writer_path is not None:
            os.unlink(writer_path)
            writer_unlinked = True
        writer.close()
        pinned_file = os.fdopen(readonly_descriptor, "rb")
        readonly_descriptor = None
        os.set_inheritable(pinned_file.fileno(), True)
        return pinned_file, digest.hexdigest(), bytes(collected)
    except Exception:
        if readonly_descriptor is not None:
            os.close(readonly_descriptor)
        if not writer.closed:
            writer.close()
        if writer_path is not None and not writer_unlinked:
            _unlink_owned_temporary_file(writer_path, writer_stat)
        raise
    finally:
        os.close(source_descriptor)


def _unlink_owned_temporary_file(path: Path, expected: os.stat_result) -> None:
    try:
        current = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise ValueError("Refusing to remove replaced pinned dataset artifact")
    path.unlink()


def _close_pinned_files(pinned_files: list[BinaryIO]) -> None:
    first_error = None
    for pinned_file in reversed(pinned_files):
        try:
            pinned_file.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _uses_sealed_memfd() -> bool:
    return sys.platform.startswith("linux")


def _seal_memfd(descriptor: int) -> None:
    if fcntl is None:
        raise ValueError("Training runtime unsupported: memfd sealing is unavailable")
    seal_names = (
        "F_SEAL_WRITE",
        "F_SEAL_GROW",
        "F_SEAL_SHRINK",
        "F_SEAL_SEAL",
    )
    if not all(hasattr(fcntl, name) for name in seal_names):
        raise ValueError("Training runtime unsupported: memfd sealing is unavailable")
    seals = sum(getattr(fcntl, name) for name in seal_names)
    fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
    applied = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
    if applied & seals != seals:
        raise ValueError("Training runtime unsupported: memfd sealing failed")


def _open_relative_file(root_descriptor: int, relative: Path) -> int:
    relative = _safe_relative_path(relative.as_posix())
    directory_descriptor = os.dup(root_descriptor)
    try:
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        for part in relative.parts[:-1]:
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        descriptor = os.open(
            relative.parts[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError(f"Training dataset artifact is not a file: {relative}")
        return descriptor
    finally:
        os.close(directory_descriptor)


def _safe_relative_path(raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Training dataset artifact path invalid: {raw_path!r}")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Training dataset artifact path invalid: {raw_path}")
    return relative


def require_training_runtime_support() -> tuple[Path, str]:
    """Return the usable descriptor root and its open-offset contract."""
    if not _is_posix_runtime():
        raise ValueError("Training runtime unsupported: POSIX is required")
    if not _BASH_PATH.is_file() or not os.access(_BASH_PATH, os.X_OK):
        raise ValueError(
            "Training runtime unsupported: executable /bin/bash is required"
        )
    root = _descriptor_root()
    if sys.platform.startswith("linux") and (
        root != Path("/proc/self/fd") or not _linux_memfd_sealing_available()
    ):
        raise ValueError(
            "Training runtime unsupported: Linux memfd sealing is required"
        )
    contract = (
        INDEPENDENT_OPEN_CONTRACT
        if root == Path("/proc/self/fd")
        else SINGLE_OPEN_CONTRACT
    )
    return root, contract


def _is_posix_runtime() -> bool:
    return os.name == "posix"


def _descriptor_root() -> Path:
    for root in (Path("/proc/self/fd"), Path("/dev/fd")):
        if root.is_dir() and os.access(root, os.R_OK | os.X_OK):
            return root
    raise ValueError(
        "Training runtime unsupported: /proc/self/fd or /dev/fd is required"
    )


def _linux_memfd_sealing_available() -> bool:
    required_fcntl = (
        "F_ADD_SEALS",
        "F_GET_SEALS",
        "F_SEAL_WRITE",
        "F_SEAL_GROW",
        "F_SEAL_SHRINK",
        "F_SEAL_SEAL",
    )
    return bool(
        hasattr(os, "memfd_create")
        and hasattr(os, "MFD_ALLOW_SEALING")
        and fcntl is not None
        and all(hasattr(fcntl, name) for name in required_fcntl)
    )


def _descriptor_path(descriptor: int, root: Path) -> Path:
    descriptor_path = root / str(descriptor)
    if not descriptor_path.exists():
        raise ValueError(
            "Training runtime unsupported: descriptor root cannot expose "
            "inherited dataset files"
        )
    return descriptor_path


def _create_private_fd_view(
    parent_descriptor: int,
) -> tuple[str, int, tuple[int, int]]:
    for _attempt in range(16):
        name = f".morpheus-fd-dataset-{secrets.token_hex(16)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        created = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        identity = (created.st_dev, created.st_ino)
        descriptor: int | None = None
        try:
            descriptor = _open_directory_at(parent_descriptor, name)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != identity
            ):
                raise ValueError("Training FD view changed during creation")
            os.fchmod(descriptor, 0o700)
            return name, descriptor, identity
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            current = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == identity
            ):
                os.rmdir(name, dir_fd=parent_descriptor)
            raise
    raise ValueError("Unable to allocate private training FD view")


def _ensure_fd_view_parent(
    view_descriptor: int,
    relative_parent: Path,
    identities: dict[Path, tuple[int, int]],
) -> int:
    descriptor = os.dup(view_descriptor)
    if relative_parent == Path("."):
        return descriptor
    current = Path()
    try:
        for part in relative_parent.parts:
            current /= part
            expected_identity = identities.get(current)
            if expected_identity is None:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
            next_descriptor = _open_directory_at(descriptor, part)
            try:
                opened = os.fstat(next_descriptor)
                identity = (opened.st_dev, opened.st_ino)
                if expected_identity is not None and identity != expected_identity:
                    raise ValueError(
                        "Training FD view directory changed during creation"
                    )
                identities[current] = identity
            except Exception:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _fd_view_layout(
    directories: dict[Path, tuple[int, int]],
    symlinks: dict[Path, str],
) -> _FDViewLayout:
    return _FDViewLayout(
        directory_identities=tuple(
            (path.as_posix(), identity[0], identity[1])
            for path, identity in sorted(
                directories.items(),
                key=lambda item: item[0].as_posix(),
            )
        ),
        symlink_targets=tuple(
            (path.as_posix(), target)
            for path, target in sorted(
                symlinks.items(),
                key=lambda item: item[0].as_posix(),
            )
        ),
    )


def _validate_held_fd_view(snapshot: PinnedDatasetSnapshot) -> None:
    opened = os.fstat(snapshot.view_descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != snapshot._view_identity
    ):
        raise ValueError("Training FD view descriptor identity changed")
    parent = os.stat(
        "..",
        dir_fd=snapshot.view_descriptor,
        follow_symlinks=False,
    )
    if (
        snapshot._expected_parent_identity is not None
        and (parent.st_dev, parent.st_ino) != snapshot._expected_parent_identity
    ):
        raise ValueError("Training FD view parent identity changed")
    _validate_fd_view_descriptor(
        snapshot.view_descriptor,
        Path("."),
        snapshot._view_layout,
    )


def _remove_fd_view(
    view_dir: Path,
    expected_identity: tuple[int, int],
    layout: _FDViewLayout,
) -> None:
    parent_descriptor = _open_directory(view_dir.parent)
    try:
        view_descriptor = _open_expected_view(
            parent_descriptor,
            view_dir.name,
            expected_identity,
        )
        try:
            _validate_fd_view_descriptor(view_descriptor, Path("."), layout)
            _remove_expected_view_entries(view_descriptor, Path("."), layout)
        finally:
            os.close(view_descriptor)
        try:
            current = os.stat(
                view_dir.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != expected_identity
        ):
            raise ValueError("Refusing to remove replaced training FD view")
        os.rmdir(view_dir.name, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _open_expected_view(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
) -> int:
    try:
        descriptor = _open_directory_at(parent_descriptor, name)
    except OSError as exc:
        raise ValueError("Refusing to remove replaced training FD view") from exc
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != expected_identity:
        os.close(descriptor)
        raise ValueError("Refusing to remove replaced training FD view")
    return descriptor


def _open_directory_at(parent_descriptor: int, name: str) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(name, flags, dir_fd=parent_descriptor)


def _validate_fd_view_descriptor(
    directory_descriptor: int,
    relative: Path,
    layout: _FDViewLayout,
) -> None:
    directory_identities = {
        Path(path): (device, inode)
        for path, device, inode in layout.directory_identities
    }
    symlink_targets = {
        Path(path): target for path, target in layout.symlink_targets
    }
    expected_children = _expected_view_children(
        relative,
        directory_identities,
        symlink_targets,
    )
    if _directory_entries(directory_descriptor) != expected_children:
        raise ValueError("Training FD view layout changed; refusing cleanup")
    for name in sorted(expected_children):
        child = Path(name) if relative == Path(".") else relative / name
        if child in symlink_targets:
            _require_expected_symlink(
                directory_descriptor,
                name,
                symlink_targets[child],
            )
            continue
        expected_identity = directory_identities.get(child)
        if expected_identity is None:
            raise ValueError("Training FD view layout invalid")
        child_descriptor = _open_expected_directory_at(
            directory_descriptor,
            name,
            expected_identity,
        )
        try:
            _validate_fd_view_descriptor(child_descriptor, child, layout)
        finally:
            os.close(child_descriptor)


def _remove_expected_view_entries(
    directory_descriptor: int,
    relative: Path,
    layout: _FDViewLayout,
) -> None:
    directory_identities = {
        Path(path): (device, inode)
        for path, device, inode in layout.directory_identities
    }
    symlink_targets = {
        Path(path): target for path, target in layout.symlink_targets
    }
    children = _expected_view_children(
        relative,
        directory_identities,
        symlink_targets,
    )
    for name in sorted(children):
        child = Path(name) if relative == Path(".") else relative / name
        if child in symlink_targets:
            _require_expected_symlink(
                directory_descriptor,
                name,
                symlink_targets[child],
            )
            os.unlink(name, dir_fd=directory_descriptor)
    for name in sorted(children):
        child = Path(name) if relative == Path(".") else relative / name
        expected_identity = directory_identities.get(child)
        if expected_identity is None:
            continue
        child_descriptor = _open_expected_directory_at(
            directory_descriptor,
            name,
            expected_identity,
        )
        try:
            _remove_expected_view_entries(child_descriptor, child, layout)
        finally:
            os.close(child_descriptor)
        current = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != expected_identity
        ):
            raise ValueError("Training FD view changed during cleanup")
        try:
            os.rmdir(name, dir_fd=directory_descriptor)
        except OSError as exc:
            raise ValueError("Training FD view changed during cleanup") from exc


def _directory_entries(directory_descriptor: int) -> set[str]:
    fresh_descriptor = _open_directory_at(directory_descriptor, ".")
    try:
        return set(os.listdir(fresh_descriptor))
    finally:
        os.close(fresh_descriptor)


def _expected_view_children(
    relative: Path,
    directory_identities: dict[Path, tuple[int, int]],
    symlink_targets: dict[Path, str],
) -> set[str]:
    children = set()
    for path in [*directory_identities, *symlink_targets]:
        if path.parent == relative:
            children.add(path.name)
    return children


def _open_expected_directory_at(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
) -> int:
    try:
        descriptor = _open_directory_at(parent_descriptor, name)
    except OSError as exc:
        raise ValueError("Training FD view changed during cleanup") from exc
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != expected_identity:
        os.close(descriptor)
        raise ValueError("Training FD view changed during cleanup")
    return descriptor


def _require_expected_symlink(
    directory_descriptor: int,
    name: str,
    expected_target: str,
) -> None:
    try:
        opened = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        target = os.readlink(name, dir_fd=directory_descriptor)
    except OSError as exc:
        raise ValueError("Training FD view changed during cleanup") from exc
    if not stat.S_ISLNK(opened.st_mode) or target != expected_target:
        raise ValueError("Training FD view changed during cleanup")


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _safe_directory(path: Path, label: str) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    reject_symlink_components(path, label)
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    return path
