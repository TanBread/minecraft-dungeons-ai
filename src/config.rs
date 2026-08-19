use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_game")]
    pub game: GameConfig,
    #[serde(default = "default_capture")]
    pub capture: CaptureConfig,
    #[serde(default = "default_env")]
    pub env: EnvConfig,
    #[serde(default = "default_training")]
    pub training: TrainingConfig,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct GameConfig {
    #[serde(default = "default_process_name")]
    pub process_name: String,
    #[serde(default = "default_window_title")]
    pub window_title: String,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct CaptureConfig {
    #[serde(default = "default_target_fps")]
    pub target_fps: u32,
    #[serde(default = "default_resolution")]
    pub resolution: [u32; 2],
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct EnvConfig {
    #[serde(default = "default_max_episode_steps")]
    pub max_episode_steps: u32,
    #[serde(default = "default_frame_stack")]
    pub frame_stack: usize,
    #[serde(default = "default_num_items")]
    pub num_items: usize,
    #[serde(default = "default_view_radius")]
    pub view_radius: i32,
    #[serde(default = "default_num_enemies_phase1")]
    pub num_enemies_phase1: usize,
    #[serde(default = "default_num_enemies_phase2")]
    pub num_enemies_phase2: usize,
    #[serde(default = "default_phase1_clears")]
    pub phase1_clears: u32,
    #[serde(default = "default_action_repeat")]
    pub action_repeat: usize,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TrainingConfig {
    #[serde(default = "default_total_timesteps")]
    pub total_timesteps: u64,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_n_steps")]
    pub n_steps: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_n_epochs")]
    pub n_epochs: usize,
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    #[serde(default = "default_gae_lambda")]
    pub gae_lambda: f64,
    #[serde(default = "default_clip_range")]
    pub clip_range: f64,
    #[serde(default = "default_ent_coef")]
    pub ent_coef: f64,
    #[serde(default = "default_vf_coef")]
    pub vf_coef: f64,
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,
    #[serde(default)]
    pub device: String,
    #[serde(default = "default_save_freq")]
    pub save_freq: u64,
    #[serde(default)]
    pub log_dir: String,
    #[serde(default)]
    pub model_dir: String,
    #[serde(default = "default_num_envs")]
    pub num_envs: usize,
    #[serde(default = "default_fps_cap")]
    pub fps_cap: u32,
    #[serde(default)]
    pub cpu_cores: Option<Vec<usize>>,
}

fn default_game() -> GameConfig { GameConfig::default() }
fn default_capture() -> CaptureConfig { CaptureConfig::default() }
fn default_env() -> EnvConfig { EnvConfig::default() }
fn default_training() -> TrainingConfig { TrainingConfig::default() }
fn default_process_name() -> String { "Dungeons-Win64-Shipping.exe".into() }
fn default_window_title() -> String { "Minecraft Dungeons".into() }
fn default_target_fps() -> u32 { 30 }
fn default_resolution() -> [u32; 2] { [160, 90] }
fn default_max_episode_steps() -> u32 { 10000 }
fn default_frame_stack() -> usize { 2 }
fn default_num_items() -> usize { 5 }
fn default_view_radius() -> i32 { 6 }
fn default_num_enemies_phase1() -> usize { 0 }
fn default_num_enemies_phase2() -> usize { 36 }
fn default_phase1_clears() -> u32 { 5 }
fn default_action_repeat() -> usize { 4 }
fn default_total_timesteps() -> u64 { 2_000_000 }
fn default_learning_rate() -> f64 { 3e-4 }
fn default_n_steps() -> usize { 1024 }
fn default_batch_size() -> usize { 256 }
fn default_n_epochs() -> usize { 4 }
fn default_gamma() -> f64 { 0.99 }
fn default_gae_lambda() -> f64 { 0.95 }
fn default_clip_range() -> f64 { 0.2 }
fn default_ent_coef() -> f64 { 0.1 }
fn default_vf_coef() -> f64 { 0.5 }
fn default_max_grad_norm() -> f64 { 0.5 }
fn default_save_freq() -> u64 { 100_000 }
fn default_num_envs() -> usize { 2 }
fn default_fps_cap() -> u32 { 60 }

impl Default for Config {
    fn default() -> Self {
        Self {
            game: GameConfig {
                process_name: default_process_name(),
                window_title: default_window_title(),
            },
            capture: CaptureConfig {
                target_fps: default_target_fps(),
                resolution: default_resolution(),
            },
            env: EnvConfig {
                max_episode_steps: default_max_episode_steps(),
                frame_stack: default_frame_stack(),
                num_items: default_num_items(),
                view_radius: default_view_radius(),
                num_enemies_phase1: default_num_enemies_phase1(),
                num_enemies_phase2: default_num_enemies_phase2(),
                phase1_clears: default_phase1_clears(),
                action_repeat: default_action_repeat(),
            },
            training: TrainingConfig {
                total_timesteps: default_total_timesteps(),
                learning_rate: default_learning_rate(),
                n_steps: default_n_steps(),
                batch_size: default_batch_size(),
                n_epochs: default_n_epochs(),
                gamma: default_gamma(),
                gae_lambda: default_gae_lambda(),
                clip_range: default_clip_range(),
                ent_coef: default_ent_coef(),
                vf_coef: default_vf_coef(),
                max_grad_norm: default_max_grad_norm(),
                device: "cuda".into(),
                save_freq: default_save_freq(),
                log_dir: "logs/".into(),
                model_dir: "models/".into(),
                num_envs: default_num_envs(),
                fps_cap: default_fps_cap(),
                cpu_cores: None,
            },
        }
    }
}

impl Config {
    pub fn load(path: &str) -> Self {
        let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Failed to read config '{}': {}", path, e);
            std::process::exit(1);
        });
        serde_yaml::from_str(&data).unwrap_or_else(|e| {
            eprintln!("Failed to parse config: {}", e);
            std::process::exit(1);
        })
    }
}
