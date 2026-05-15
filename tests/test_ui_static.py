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
