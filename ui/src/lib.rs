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
