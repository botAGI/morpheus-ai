from contextlib import contextmanager
import subprocess
import sys
import time
from pathlib import Path

from typer.testing import CliRunner

import morpheus.api.server as api_module
import morpheus.cli as cli_module
import morpheus.core.learning.adapters as adapters_module
import morpheus.core.learning.dataset_validation as dataset_validation_module
import morpheus.core.semantic.review as review_module
from morpheus.core.config import MorpheusConfig
from morpheus.core.learning.adapters import activate_adapter, rollback_adapter
from morpheus.core.learning.dataset import build_learning_dataset
from morpheus.core.learning.dataset_validation import capture_active_state_authority
from morpheus.core.learning.eval import run_learning_eval
from morpheus.core.learning.train import plan_training_run
from morpheus.core.semantic.review import ReviewStore, apply_accepted_candidates
from morpheus.core.state_authority import state_authority_transaction
from tests.test_learning_adapters import (
    make_benchmark_ready_review_fixture,
    mark_all_evals_activation_eligible,
    planned_adapter,
)
from tests.test_learning_dataset import copy_learning_project


def _lock_probe():
    state = {"depth": 0}

    @contextmanager
    def transaction(_project_root):
        state["depth"] += 1
        try:
            yield
        finally:
            state["depth"] -= 1

    return state, transaction


def test_state_authority_transaction_is_reentrant(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()

    with state_authority_transaction(tmp_path):
        with state_authority_transaction(tmp_path):
            assert (tmp_path / ".morpheus/.state-authority.lock").is_file()


def test_state_authority_transaction_serializes_processes(tmp_path):
    MorpheusConfig(project_root=tmp_path).init_default()
    started_path = tmp_path / "child-started"
    acquired_path = tmp_path / "child-acquired"
    script = "\n".join(
        [
            "import sys",
            "from pathlib import Path",
            "from morpheus.core.state_authority import state_authority_transaction",
            "root = Path(sys.argv[1])",
            "(root / 'child-started').write_text('started')",
            "with state_authority_transaction(root):",
            "    (root / 'child-acquired').write_text('acquired')",
        ]
    )
    process = None
    try:
        with state_authority_transaction(tmp_path):
            process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)])
            deadline = time.monotonic() + 5
            while not started_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert started_path.exists()
            time.sleep(0.2)
            assert not acquired_path.exists()
        process.wait(timeout=5)
        assert process.returncode == 0
        assert acquired_path.read_text() == "acquired"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_semantic_apply_holds_state_lock_before_review_and_through_receipt(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "README.md").write_text(
        "Morpheus generates WAKE.md for AI agents.\n"
    )
    MorpheusConfig(project_root=tmp_path).init_default()

    state, transaction = _lock_probe()
    review_depth = 0
    original_review_transaction = ReviewStore.transaction
    original_write_receipt = review_module._write_state_receipt

    @contextmanager
    def checked_review_transaction(self):
        nonlocal review_depth
        assert state["depth"] == 1
        with original_review_transaction(self):
            review_depth += 1
            try:
                yield
            finally:
                review_depth -= 1

    def checked_write_receipt(*args, **kwargs):
        assert state["depth"] == 1
        assert review_depth == 1
        return original_write_receipt(*args, **kwargs)

    monkeypatch.setattr(review_module, "state_authority_transaction", transaction)
    monkeypatch.setattr(ReviewStore, "transaction", checked_review_transaction)
    monkeypatch.setattr(review_module, "_write_state_receipt", checked_write_receipt)

    result = apply_accepted_candidates(tmp_path)

    assert result["receipt_id"].startswith("rcpt_")
    assert state["depth"] == 0
    assert review_depth == 0


def test_cli_compile_holds_state_lock_through_signing(tmp_path, monkeypatch):
    runner = CliRunner()
    state, transaction = _lock_probe()
    original_compile = cli_module.compile_project
    original_build_receipt = cli_module.build_receipt

    def checked_compile(*args, **kwargs):
        assert state["depth"] == 1
        return original_compile(*args, **kwargs)

    def checked_build_receipt(*args, **kwargs):
        assert state["depth"] == 1
        return original_build_receipt(*args, **kwargs)

    monkeypatch.setattr(cli_module, "state_authority_transaction", transaction)
    monkeypatch.setattr(cli_module, "compile_project", checked_compile)
    monkeypatch.setattr(cli_module, "build_receipt", checked_build_receipt)

    monkeypatch.chdir(tmp_path)
    Path("README.md").write_text("TODO: serialize CLI compile\n")
    assert runner.invoke(cli_module.app, ["init"]).exit_code == 0

    result = runner.invoke(cli_module.app, ["compile"])

    assert result.exit_code == 0, result.output
    assert state["depth"] == 0


def test_api_compile_holds_state_lock_through_signing(tmp_path, monkeypatch):
    MorpheusConfig(project_root=tmp_path).init_default()
    (tmp_path / "README.md").write_text("TODO: serialize API compile\n")
    state, transaction = _lock_probe()
    original_compile = api_module.compile_project
    original_build_receipt = api_module.build_receipt

    def checked_compile(*args, **kwargs):
        assert state["depth"] == 1
        return original_compile(*args, **kwargs)

    def checked_build_receipt(*args, **kwargs):
        assert state["depth"] == 1
        return original_build_receipt(*args, **kwargs)

    monkeypatch.setattr(api_module, "state_authority_transaction", transaction)
    monkeypatch.setattr(api_module, "compile_project", checked_compile)
    monkeypatch.setattr(api_module, "build_receipt", checked_build_receipt)

    response = api_module.compile(api_module.CompileRequest(project_root=str(tmp_path)))

    assert response.receipt_id.startswith("rcpt_")
    assert state["depth"] == 0


def test_capture_active_state_authority_holds_state_lock(tmp_path, monkeypatch):
    project_root = copy_learning_project(tmp_path)
    MorpheusConfig(project_root=project_root).init_default()
    apply_accepted_candidates(project_root)
    state, transaction = _lock_probe()
    original_read_context = dataset_validation_module._read_active_state_context

    def checked_read_context(*args, **kwargs):
        assert state["depth"] == 1
        return original_read_context(*args, **kwargs)

    monkeypatch.setattr(
        dataset_validation_module,
        "state_authority_transaction",
        transaction,
    )
    monkeypatch.setattr(
        dataset_validation_module,
        "_read_active_state_context",
        checked_read_context,
    )

    authority = capture_active_state_authority(project_root)

    assert authority["receipt_id"].startswith("rcpt_")
    assert state["depth"] == 0


def test_activation_reenters_state_lock_for_active_state_dataset(tmp_path):
    project_root = copy_learning_project(tmp_path)
    make_benchmark_ready_review_fixture(project_root)
    MorpheusConfig(project_root=project_root).init_default()
    apply_accepted_candidates(project_root)
    build_learning_dataset(project_root, source="active-state")
    training = plan_training_run(project_root, dry_run=True)
    run_learning_eval(project_root, base_only=True, dry_run=True)
    run_learning_eval(
        project_root,
        adapter_id=training["adapter_id"],
        dry_run=True,
    )
    mark_all_evals_activation_eligible(project_root)
    script = "\n".join(
        [
            "import sys",
            "from pathlib import Path",
            "from morpheus.core.learning.adapters import activate_adapter",
            "activate_adapter(Path(sys.argv[1]), sys.argv[2])",
        ]
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(project_root),
            training["adapter_id"],
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr


def test_activation_and_rollback_hold_state_before_review_and_pointer_commit(
    tmp_path,
    monkeypatch,
):
    project_root = copy_learning_project(tmp_path)
    first = planned_adapter(project_root, activation_eligible=True)
    activate_adapter(project_root, first["adapter_id"])
    second = plan_training_run(project_root, dry_run=True)
    run_learning_eval(
        project_root,
        adapter_id=second["adapter_id"],
        dry_run=True,
    )
    mark_all_evals_activation_eligible(project_root)
    state, transaction = _lock_probe()
    review_depth = 0
    original_identity = adapters_module._activation_authority_identity
    original_commit = adapters_module._commit_activation_transaction

    @contextmanager
    def checked_review(_project_root):
        nonlocal review_depth
        assert state["depth"] == 1
        review_depth += 1
        try:
            yield
        finally:
            review_depth -= 1

    def checked_identity(*args, **kwargs):
        assert state["depth"] == 1
        assert review_depth == 1
        return original_identity(*args, **kwargs)

    def checked_commit(*args, **kwargs):
        assert state["depth"] == 1
        assert review_depth == 1
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(adapters_module, "state_authority_transaction", transaction)
    monkeypatch.setattr(adapters_module, "_review_authority_transaction", checked_review)
    monkeypatch.setattr(adapters_module, "_activation_authority_identity", checked_identity)
    monkeypatch.setattr(adapters_module, "_commit_activation_transaction", checked_commit)

    activate_adapter(project_root, second["adapter_id"])
    rollback_adapter(project_root)

    assert state["depth"] == 0
    assert review_depth == 0
