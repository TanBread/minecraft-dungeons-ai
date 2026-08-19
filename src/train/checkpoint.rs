use std::path::Path;
use crate::model::MinecraftDungeonsPolicy;

pub fn save_checkpoint(policy: &MinecraftDungeonsPolicy, path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Err(e) = policy.save(path) {
        eprintln!("[Checkpoint] Failed to save: {}", e);
    } else {
        println!("[Checkpoint] Saved: {}", path);
    }
}

pub fn load_checkpoint(policy: &mut MinecraftDungeonsPolicy, path: &str) -> bool {
    if Path::new(path).exists() {
        match policy.load(path) {
            Ok(_) => {
                println!("[Checkpoint] Loaded: {}", path);
                true
            }
            Err(e) => {
                eprintln!("[Checkpoint] Failed to load: {}", e);
                false
            }
        }
    } else {
        false
    }
}
