from pathlib import Path


def test_ui_exposes_diagnostics_next_action_controls():
    html = Path("ui/index.html").read_text()

    required_snippets = [
        'id="diagnostics-next-action-label"',
        'id="diagnostics-next-action-detail"',
        'id="diagnostics-next-action-command"',
        'id="run-next-action-btn"',
        "function updateNextAction",
        "async function runNextAction",
        "payload.next_action",
    ]
    for snippet in required_snippets:
        assert snippet in html


def test_prepare_agent_updates_receipt_step_from_verify_result():
    html = Path("ui/index.html").read_text()

    assert "payload.verified && payload.verified.valid" in html
    assert "setStep('verify', 'ready', 'Verified', `Receipt: ${receiptId}`)" in html


def test_next_action_button_only_enables_runnable_actions():
    html = Path("ui/index.html").read_text()

    assert "const runnableActions = ['prepare_agent', 'handoff'];" in html
    assert "runNextActionBtn.disabled = !runnableActions.includes(action.id);" in html


def test_ui_exposes_context_sources_and_faq():
    html = Path("ui/index.html").read_text()

    required_snippets = [
        'id="context-watch-dirs-input"',
        'id="save-context-paths-btn"',
        'id="load-context-paths-btn"',
        'id="context-paths-summary"',
        "async function loadProjectConfig",
        "async function saveProjectConfig",
        "apiRequest('/config'",
        "FAQ",
        "How do I connect Morpheus to a project?",
        "How do I include several folders or projects?",
    ]
    for snippet in required_snippets:
        assert snippet in html
