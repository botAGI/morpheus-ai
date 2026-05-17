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

    assert "const runnableActions = ['set_project_root', 'prepare_agent', 'handoff'];" in html
    assert "currentNextAction.id === 'set_project_root'" in html
    assert "focusProjectRootInput()" in html
    assert "runNextActionBtn.disabled = !runnableActions.includes(action.id);" in html


def test_ui_exposes_direct_status_wake_and_chat_actions():
    html = Path("ui/index.html").read_text()

    required_snippets = [
        'id="refresh-status-btn"',
        'id="status-diagnostics-btn"',
        'id="refresh-wake-btn"',
        'id="copy-wake-btn"',
        'id="chat-prepare-btn"',
        'id="chat-diagnostics-btn"',
        'id="chat-compile-btn"',
        'id="chat-verify-btn"',
        'id="chat-wake-btn"',
        "function runChatAction",
        "refreshStatusBtn.addEventListener('click'",
        "copyWakeBtn.addEventListener('click'",
        "chatPrepareBtn.addEventListener('click'",
    ]
    for snippet in required_snippets:
        assert snippet in html


def test_sidebar_exposes_full_quick_action_set():
    html = Path("ui/index.html").read_text()

    required_snippets = [
        'id="quick-init-btn"',
        'id="quick-prepare-btn"',
        'id="quick-diagnostics-btn"',
        'id="quick-handoff-btn"',
        "quickInitBtn.addEventListener('click'",
        "quickPrepareBtn.addEventListener('click'",
        "quickDiagnosticsBtn.addEventListener('click'",
        "quickHandoffBtn.addEventListener('click'",
        "openPanel('start')",
        "buildAgentHandoff({ silent: false })",
    ]
    for snippet in required_snippets:
        assert snippet in html


def test_settings_integrations_panel_uses_real_manifest_controls():
    html = Path("ui/index.html").read_text()

    required_snippets = [
        'id="settings-integrations-summary"',
        'id="settings-refresh-integrations-btn"',
        "settingsIntegrationsSummary",
        "settingsRefreshIntegrationsBtn.addEventListener('click'",
        "refreshIntegrations({ silent: false })",
    ]
    for snippet in required_snippets:
        assert snippet in html

    assert "coming soon" not in html.lower()


def test_settings_integrations_panel_has_single_button_row_wrapper():
    html = Path("ui/index.html").read_text()
    settings_integrations = html.split("<h4>Integrations</h4>", 1)[1].split("<h4>Voice</h4>", 1)[0]

    assert settings_integrations.count('class="button-row"') == 1


def test_ui_bootstraps_backend_api_from_query_parameter():
    html = Path("ui/index.html").read_text()

    required_snippets = [
        "function apiBaseFromQuery",
        "new URLSearchParams(window.location.search)",
        "params.get('api')",
        "localStorage.setItem('morpheus.apiBase', queryApiBase)",
        "apiBaseInput.value = queryApiBase ||",
    ]
    for snippet in required_snippets:
        assert snippet in html


def test_ui_exposes_agent_interop_endpoint_actions():
    html = Path("ui/index.html").read_text()

    required_snippets = [
        'id="copy-native-manifest-btn"',
        'id="copy-a2a-card-btn"',
        'id="copy-mcp-endpoint-btn"',
        'id="test-mcp-tools-btn"',
        'id="mcp-tools-output"',
        "async function runMcpToolsProbe",
        "\"method\": \"tools/list\"",
        "copyNativeManifestBtn.addEventListener('click'",
        "copyA2aCardBtn.addEventListener('click'",
        "copyMcpEndpointBtn.addEventListener('click'",
        "testMcpToolsBtn.addEventListener('click'",
    ]
    for snippet in required_snippets:
        assert snippet in html


def test_voice_button_uses_browser_speech_recognition_when_available():
    html = Path("ui/index.html").read_text()

    required_snippets = [
        "window.SpeechRecognition || window.webkitSpeechRecognition",
        "recognition.onresult",
        "inputEl.value = transcript",
        "sendMessage()",
        "Voice input unavailable",
    ]
    for snippet in required_snippets:
        assert snippet in html


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
        "Should WAKE.md be committed?",
        "Is Morpheus a model trainer?",
    ]
    for snippet in required_snippets:
        assert snippet in html


def test_ui_exposes_integration_paths_for_agents():
    html = Path("ui/index.html").read_text()

    required_snippets = [
        'id="integrations-summary"',
        'id="integrations-list"',
        'id="refresh-integrations-btn"',
        "function renderIntegrations",
        "async function refreshIntegrations",
        "apiRequest('/integrations'",
        "Integration Paths",
    ]
    for snippet in required_snippets:
        assert snippet in html


def test_ui_exposes_model_smoke_controls():
    html = Path("ui/index.html").read_text()

    required_snippets = [
        'id="model-smoke-base-model"',
        'id="model-smoke-prompt"',
        'id="run-model-smoke-btn"',
        'id="model-smoke-output"',
        "async function runModelSmoke",
        "apiRequest('/models/smoke'",
        "Model Smoke",
    ]
    for snippet in required_snippets:
        assert snippet in html


def test_ui_exposes_launchpad_for_humans_and_agents():
    html = Path("ui/index.html").read_text()

    required_snippets = [
        "Morpheus Launchpad",
        'id="quickstart-summary"',
        'id="quickstart-commands"',
        'id="connect-pack-output"',
        'id="refresh-quickstart-btn"',
        'id="copy-connect-pack-btn"',
        "async function refreshQuickstart",
        "function renderQuickstart",
        "function buildConnectPack",
        "apiRequest('/quickstart'",
        "Human start",
        "Agent start",
        "A2A Agent Card",
        "MCP endpoint",
    ]
    for snippet in required_snippets:
        assert snippet in html
