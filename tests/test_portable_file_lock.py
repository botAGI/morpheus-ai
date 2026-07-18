import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from textwrap import dedent

import pytest


_WINDOWS_LOCK_PREAMBLE = """
import subprocess
import sys
import types

calls = []
fake_msvcrt = types.ModuleType("msvcrt")
fake_msvcrt.LK_NBLCK = 1
fake_msvcrt.LK_UNLCK = 2

def locking(descriptor, mode, size):
    calls.append((mode, size))

fake_msvcrt.locking = locking
sys.modules["fcntl"] = None
sys.modules["msvcrt"] = fake_msvcrt
"""


def _run_windows_lock_probe(project_root: Path, body: str) -> subprocess.CompletedProcess[str]:
    script = dedent(_WINDOWS_LOCK_PREAMBLE + "\n" + body)
    return subprocess.run(
        [sys.executable, "-c", script, str(project_root)],
        check=False,
        capture_output=True,
        text=True,
    )


def _fork_transaction_child(transaction) -> int:
    child_pid = os.fork()
    if child_pid == 0:
        signal.alarm(2)
        try:
            with transaction():
                pass
        except BaseException:
            os._exit(1)
        signal.alarm(0)
        os._exit(0)
    return child_pid


def _assert_child_succeeded(child_pid: int) -> None:
    _, status = os.waitpid(child_pid, 0)
    assert os.WIFEXITED(status), f"child terminated by signal {os.WTERMSIG(status)}"
    assert os.WEXITSTATUS(status) == 0


def _fork_while_other_thread_holds(transaction) -> int:
    entered = threading.Event()
    release = threading.Event()

    def hold_transaction() -> None:
        with transaction():
            entered.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold_transaction)
    holder.start()
    assert entered.wait(timeout=2)
    try:
        return _fork_transaction_child(transaction)
    finally:
        release.set()
        holder.join(timeout=2)
        assert not holder.is_alive()


def test_review_store_uses_reentrant_windows_byte_lock(tmp_path):
    result = _run_windows_lock_probe(
        tmp_path,
        """
from pathlib import Path
from morpheus.core.semantic.review import ReviewStore

project_root = Path(sys.argv[1])
store = ReviewStore(project_root)
with store.transaction():
    with store.transaction():
        assert calls == [(fake_msvcrt.LK_NBLCK, 1)], calls

assert calls == [
    (fake_msvcrt.LK_NBLCK, 1),
    (fake_msvcrt.LK_UNLCK, 1),
], calls
assert (project_root / ".morpheus/review/.store.lock").read_bytes() == b"\\0"
""",
    )

    assert result.returncode == 0, result.stderr


def test_activation_recovery_runs_under_reentrant_windows_byte_lock(tmp_path):
    result = _run_windows_lock_probe(
        tmp_path,
        """
from pathlib import Path
import morpheus.core.learning.adapters as adapters_module

project_root = Path(sys.argv[1])
project_root.mkdir(parents=True, exist_ok=True)
recoveries = []

def checked_recovery(root):
    assert calls == [(fake_msvcrt.LK_NBLCK, 1)], calls
    recoveries.append(root)

adapters_module._recover_activation_transaction = checked_recovery
with adapters_module._activation_state_transaction(project_root):
    with adapters_module._activation_state_transaction(project_root):
        assert calls == [(fake_msvcrt.LK_NBLCK, 1)], calls

assert recoveries == [project_root, project_root], recoveries
assert calls == [
    (fake_msvcrt.LK_NBLCK, 1),
    (fake_msvcrt.LK_UNLCK, 1),
], calls
assert (project_root / ".morpheus/training/.activation.lock").read_bytes() == b"\\0"
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_review_store_transaction_reinitializes_thread_lock_after_fork(tmp_path):
    from morpheus.core.semantic.review import ReviewStore

    child_pid = _fork_while_other_thread_holds(
        lambda: ReviewStore(tmp_path).transaction()
    )

    _assert_child_succeeded(child_pid)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_activation_transaction_reinitializes_thread_lock_after_fork(tmp_path):
    import morpheus.core.learning.adapters as adapters_module

    child_pid = _fork_while_other_thread_holds(
        lambda: adapters_module._activation_state_transaction(tmp_path)
    )

    _assert_child_succeeded(child_pid)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.parametrize("reentrant", [False, True])
def test_inherited_lock_context_never_closes_reused_descriptor(tmp_path, reentrant):
    import morpheus.core.portable_lock as portable_lock_module

    lock_path = tmp_path / "transaction.lock"
    reused_path = tmp_path / "reused.txt"
    in_child = False
    reused_descriptor = None

    try:
        with portable_lock_module.portable_file_lock(lock_path):
            inherited_descriptor = portable_lock_module._path_lock(
                lock_path
            ).descriptor
            assert inherited_descriptor is not None

            if reentrant:
                with portable_lock_module.portable_file_lock(lock_path):
                    child_pid = os.fork()
                    in_child = child_pid == 0
                    if in_child:
                        source_descriptor = os.open(
                            reused_path,
                            os.O_RDWR | os.O_CREAT,
                            0o600,
                        )
                        if source_descriptor != inherited_descriptor:
                            os.dup2(source_descriptor, inherited_descriptor)
                            os.close(source_descriptor)
                        reused_descriptor = inherited_descriptor
            else:
                child_pid = os.fork()
                in_child = child_pid == 0
                if in_child:
                    source_descriptor = os.open(
                        reused_path,
                        os.O_RDWR | os.O_CREAT,
                        0o600,
                    )
                    if source_descriptor != inherited_descriptor:
                        os.dup2(source_descriptor, inherited_descriptor)
                        os.close(source_descriptor)
                    reused_descriptor = inherited_descriptor
    except BaseException:
        if in_child:
            os._exit(1)
        raise

    if in_child:
        assert reused_descriptor is not None
        try:
            os.fstat(reused_descriptor)
        except OSError:
            os._exit(2)
        os.close(reused_descriptor)
        os._exit(0)

    _assert_child_succeeded(child_pid)
