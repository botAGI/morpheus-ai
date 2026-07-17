"""Portable reentrant file locks for local cross-process transactions."""

from collections.abc import Iterator
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import threading
import time

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - exercised on Windows.
    fcntl = None

try:
    import msvcrt
except ModuleNotFoundError:  # pragma: no cover - exercised on POSIX.
    msvcrt = None


class _ProcessFileLock:
    """One reentrant in-process owner for a lock-file path."""

    def __init__(self) -> None:
        self.thread_lock = threading.RLock()
        self.descriptor: int | None = None
        self.depth = 0


_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, _ProcessFileLock] = {}
_PROCESS_IDENTITY = (os.getpid(), object())


def _path_lock(path: Path) -> _ProcessFileLock:
    key = os.path.normcase(os.path.abspath(os.fspath(path)))
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = _ProcessFileLock()
            _PROCESS_LOCKS[key] = lock
        return lock


def _reset_process_locks_after_fork() -> None:
    """Drop inherited descriptors and thread ownership in the fork child."""
    global _PROCESS_IDENTITY, _PROCESS_LOCKS_GUARD, _PROCESS_LOCKS
    inherited_locks = _PROCESS_LOCKS
    _PROCESS_IDENTITY = (os.getpid(), object())
    _PROCESS_LOCKS_GUARD = threading.Lock()
    _PROCESS_LOCKS = {}
    for lock in inherited_locks.values():
        if lock.descriptor is not None:
            try:
                os.close(lock.descriptor)
            except OSError:
                pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_process_locks_after_fork)


def _open_lock(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, 0o600)


def _acquire_lock(descriptor: int) -> None:
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return
    if msvcrt is None:  # pragma: no cover - supported platforms provide one.
        raise RuntimeError("No cross-process file-lock primitive is available")
    if os.fstat(descriptor).st_size == 0:  # pragma: no cover - Windows only.
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, b"\0")
        os.fsync(descriptor)
    while True:  # pragma: no cover - Windows only.
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            time.sleep(0.05)


def _release_lock(descriptor: int) -> None:
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return
    if msvcrt is None:  # pragma: no cover - paired acquire already failed.
        return
    os.lseek(descriptor, 0, os.SEEK_SET)  # pragma: no cover - Windows only.
    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)  # pragma: no cover


@contextmanager
def portable_file_lock(path: Path) -> Iterator[None]:
    """Hold one lock file across threads and local processes.

    The lock is reentrant for one thread. Callers remain responsible for
    validating the lock path before entering this context.
    """
    process_identity = _PROCESS_IDENTITY
    process_lock = _path_lock(path)
    with process_lock.thread_lock:
        if process_lock.depth:
            process_lock.depth += 1
            try:
                yield
            finally:
                if process_identity is _PROCESS_IDENTITY:
                    process_lock.depth -= 1
            return

        descriptor = _open_lock(path)
        acquired = False
        try:
            _acquire_lock(descriptor)
            acquired = True
            process_lock.descriptor = descriptor
            process_lock.depth = 1
            try:
                yield
            finally:
                if process_identity is _PROCESS_IDENTITY:
                    process_lock.depth = 0
                    process_lock.descriptor = None
        finally:
            if process_identity is _PROCESS_IDENTITY:
                try:
                    if acquired:
                        _release_lock(descriptor)
                finally:
                    os.close(descriptor)
