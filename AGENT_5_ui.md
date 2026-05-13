# Agent 5 — Tauri Frontend Skeleton

## Your Task

Implement Tauri desktop app skeleton in `/Users/testbot/.openclaw/workspace/morpheus-ai/ui/`.

### Files to create:

### 1. `Cargo.toml`
```toml
[package]
name = "morpheus-ui"
version = "0.1.0"
description = "Morpheus AI Desktop App"
authors = ["Morpheus Team"]
license = "MIT"
repository = "https://github.com/morpheus-ai/morpheus"
edition = "2021"

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = ["devtools"] }
tauri-plugin-shell = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
console_error_panic_hook = "0.1"

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]

[profile.release]
panic = "abort"
codegen-units = 1
lto = true
opt-level = "s"
strip = true
```

### 2. `tauri.conf.json`
```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "Morpheus AI",
  "version": "0.1.0",
  "identifier": "ai.morpheus.app",
  "build": {
    "beforeDevCommand": "",
    "devUrl": "http://localhost:5173",
    "beforeBuildCommand": "",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "Morpheus AI",
        "width": 900,
        "height": 700,
        "minWidth": 600,
        "minHeight": 400,
        "resizable": true,
        "fullscreen": false,
        "center": true
      }
    ],
    "security": {
      "csp": null
    }
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ]
  }
}
```

### 3. `src/main.rs`
```rust
// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    console_error_panic_hook::set_once();
    
    morpheus_ui::run();
}
```

### 4. `src/lib.rs`
```rust
use tauri::Manager;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! I'm Morpheus.", name)
}

#[tauri::command]
fn compile_project() -> Result<String, String> {
    // Call morpheus CLI
    Ok("Compiling...".to_string())
}

#[tauri::command]
fn get_status() -> Result<String, String> {
    Ok("{}".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            compile_project,
            get_status
        ])
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();
            window.set_title("Morpheus AI").unwrap();
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 5. `build.rs`
```rust
fn main() {
    tauri_build::build()
}
```

### 6. `index.html`
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Morpheus AI</title>
    <style>
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }
      
      :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a25;
        --text-primary: #e4e4e7;
        --text-secondary: #a1a1aa;
        --accent: #6366f1;
        --accent-hover: #818cf8;
        --border: #27272a;
        --success: #22c55e;
        --warning: #f59e0b;
        --error: #ef4444;
      }
      
      body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
        height: 100vh;
        overflow: hidden;
      }
      
      #app {
        display: flex;
        flex-direction: column;
        height: 100vh;
      }
      
      header {
        background: var(--bg-secondary);
        border-bottom: 1px solid var(--border);
        padding: 16px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      
      header h1 {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
      }
      
      header h1 span {
        color: var(--accent);
      }
      
      .status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: var(--text-secondary);
      }
      
      .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--success);
      }
      
      main {
        flex: 1;
        display: flex;
        overflow: hidden;
      }
      
      .sidebar {
        width: 240px;
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
        padding: 16px;
        overflow-y: auto;
      }
      
      .sidebar-section {
        margin-bottom: 24px;
      }
      
      .sidebar-section h3 {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 12px;
      }
      
      .nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 6px;
        color: var(--text-secondary);
        text-decoration: none;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.15s;
      }
      
      .nav-item:hover {
        background: var(--bg-tertiary);
        color: var(--text-primary);
      }
      
      .nav-item.active {
        background: var(--accent);
        color: white;
      }
      
      .chat-area {
        flex: 1;
        display: flex;
        flex-direction: column;
      }
      
      .messages {
        flex: 1;
        overflow-y: auto;
        padding: 24px;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }
      
      .message {
        max-width: 70%;
        padding: 12px 16px;
        border-radius: 12px;
        font-size: 14px;
        line-height: 1.5;
      }
      
      .message.user {
        align-self: flex-end;
        background: var(--accent);
        color: white;
        border-bottom-right-radius: 4px;
      }
      
      .message.morpheus {
        align-self: flex-start;
        background: var(--bg-tertiary);
        color: var(--text-primary);
        border-bottom-left-radius: 4px;
      }
      
      .message.error {
        background: var(--error);
        color: white;
      }
      
      .input-area {
        padding: 16px 24px;
        background: var(--bg-secondary);
        border-top: 1px solid var(--border);
      }
      
      .input-wrapper {
        display: flex;
        gap: 12px;
      }
      
      input[type="text"] {
        flex: 1;
        background: var(--bg-tertiary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 16px;
        color: var(--text-primary);
        font-size: 14px;
        outline: none;
      }
      
      input[type="text"]:focus {
        border-color: var(--accent);
      }
      
      button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.15s;
      }
      
      button:hover {
        background: var(--accent-hover);
      }
      
      button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      
      .panel {
        display: none;
        flex: 1;
        padding: 24px;
        overflow-y: auto;
      }
      
      .panel.active {
        display: block;
      }
      
      .panel h2 {
        font-size: 20px;
        margin-bottom: 16px;
      }
      
      .card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
      }
      
      .card h4 {
        font-size: 14px;
        margin-bottom: 8px;
      }
      
      .card p {
        font-size: 13px;
        color: var(--text-secondary);
        line-height: 1.5;
      }
      
      .stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 24px;
      }
      
      .stat {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
      }
      
      .stat-value {
        font-size: 24px;
        font-weight: 600;
        color: var(--accent);
      }
      
      .stat-label {
        font-size: 12px;
        color: var(--text-secondary);
        margin-top: 4px;
      }
    </style>
  </head>
  <body>
    <div id="app">
      <header>
        <h1>Morpheus <span>AI</span></h1>
        <div class="status">
          <div class="status-dot"></div>
          <span>Connected</span>
        </div>
      </header>
      
      <main>
        <aside class="sidebar">
          <div class="sidebar-section">
            <h3>Navigation</h3>
            <nav>
              <div class="nav-item active" data-panel="chat">💬 Chat</div>
              <div class="nav-item" data-panel="status">📊 Status</div>
              <div class="nav-item" data-panel="wake">📄 WAKE.md</div>
              <div class="nav-item" data-panel="settings">⚙️ Settings</div>
            </nav>
          </div>
          
          <div class="sidebar-section">
            <h3>Quick Actions</h3>
            <nav>
              <div class="nav-item" id="compile-btn">🔄 Compile</div>
              <div class="nav-item" id="verify-btn">✅ Verify</div>
            </nav>
          </div>
        </aside>
        
        <div class="chat-area">
          <div class="messages" id="messages">
            <div class="message morpheus">
              Hello! I'm Morpheus. Ready to help you compile and verify your project state.
            </div>
          </div>
          
          <div class="input-area">
            <div class="input-wrapper">
              <input type="text" id="message-input" placeholder="Type your message..." />
              <button id="send-btn">Send</button>
            </div>
          </div>
        </div>
        
        <div class="panel" id="panel-status">
          <h2>Project Status</h2>
          <div class="stats">
            <div class="stat">
              <div class="stat-value" id="stat-sources">0</div>
              <div class="stat-label">Sources</div>
            </div>
            <div class="stat">
              <div class="stat-value" id="stat-claims">0</div>
              <div class="stat-label">Claims</div>
            </div>
            <div class="stat">
              <div class="stat-value" id="stat-evidence">0</div>
              <div class="stat-label">Evidence</div>
            </div>
          </div>
          <div class="card">
            <h4>Last Compilation</h4>
            <p id="last-compile">Not compiled yet</p>
          </div>
        </div>
        
        <div class="panel" id="panel-wake">
          <h2>WAKE.md</h2>
          <pre id="wake-content" style="white-space: pre-wrap; font-size: 13px; line-height: 1.6;"></pre>
        </div>
        
        <div class="panel" id="panel-settings">
          <h2>Settings</h2>
          <div class="card">
            <h4>Project Root</h4>
            <p id="project-root">Not set</p>
          </div>
          <div class="card">
            <h4>Integrations</h4>
            <p>Configure Gmail, Calendar, GitHub...</p>
          </div>
        </div>
      </main>
    </div>
    
    <script>
      // Simple chat functionality
      const messagesEl = document.getElementById('messages');
      const inputEl = document.getElementById('message-input');
      const sendBtn = document.getElementById('send-btn');
      const navItems = document.querySelectorAll('.nav-item[data-panel]');
      const panels = document.querySelectorAll('.panel');
      
      let currentPanel = 'chat';
      
      navItems.forEach(item => {
        item.addEventListener('click', () => {
          const panel = item.dataset.panel;
          navItems.forEach(n => n.classList.remove('active'));
          item.classList.add('active');
          panels.forEach(p => p.classList.remove('active'));
          document.getElementById(`panel-${panel}`).classList.add('active');
          currentPanel = panel;
        });
      });
      
      function addMessage(text, type = 'morpheus') {
        const msg = document.createElement('div');
        msg.className = `message ${type}`;
        msg.textContent = text;
        messagesEl.appendChild(msg);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
      
      sendBtn.addEventListener('click', () => {
        const text = inputEl.value.trim();
        if (!text) return;
        
        addMessage(text, 'user');
        inputEl.value = '';
        
        // Simple echo for now - real implementation calls Tauri command
        setTimeout(() => {
          addMessage(`You said: ${text}`, 'morpheus');
        }, 500);
      });
      
      inputEl.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendBtn.click();
      });
      
      document.getElementById('compile-btn').addEventListener('click', () => {
        addMessage('Running morpheus compile...', 'morpheus');
      });
      
      document.getElementById('verify-btn').addEventListener('click', () => {
        addMessage('Running morpheus verify...', 'morpheus');
      });
    </script>
  </body>
</html>
```

### 7. `icons/` directory
Create placeholder icons directory with note:

```
# Icons Directory

Place the following icon files here:
- 32x32.png
- 128x128.png
- 128x128@2x.png
- icon.icns (macOS)
- icon.ico (Windows)

You can generate these from a 1024x1024 PNG using:
- https://favicon.io/
- https://icon.kitchen/
```

## Instructions

1. Create all files in `/Users/testbot/.openclaw/workspace/morpheus-ai/ui/`
2. Create `icons/` directory with placeholder README
3. The HTML is a complete working skeleton with:
   - Dark theme
   - Sidebar navigation
   - Chat interface
   - Status panel
   - WAKE.md viewer
   - Settings panel
4. JavaScript is vanilla - no frameworks
5. CSS is inline - no external dependencies
