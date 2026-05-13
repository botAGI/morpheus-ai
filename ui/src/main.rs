// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    console_error_panic_hook::set_once();
    
    morpheus_ui::run();
}
