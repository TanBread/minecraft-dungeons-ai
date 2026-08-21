mod config;
mod sim;
mod env;
mod model;
mod train;
mod viewer;

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};
use viewer::SharedViewerState;

#[derive(Parser)]
#[command(name = "minecraft-dungeons-ai")]
#[command(about = "Minecraft Dungeons AI — Rust Edition")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Sim {
        #[arg(long, default_value = "config.yaml")]
        config: String,
        #[arg(long)]
        resume: Option<String>,
        #[arg(long)]
        viewer: bool,
        #[arg(long)]
        real_maps: bool,
    },
    Test {
        #[arg(long, default_value = "config.yaml")]
        config: String,
        #[arg(long, default_value_t = 100)]
        steps: u32,
        #[arg(long)]
        real_maps: bool,
    },
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Sim { config, resume, viewer, real_maps } => {
            run_sim(&config, resume.as_deref(), viewer, real_maps);
        }
        Commands::Test { config, steps, real_maps } => {
            run_test(&config, steps, real_maps);
        }
    }
}

fn run_sim(config_path: &str, resume: Option<&str>, show_viewer: bool, use_real_maps: bool) {
    let config = config::Config::load(config_path);
    let tc = &config.training;
    let ec = &config.env;

    println!("[Sim] Starting training on {}", tc.device);
    println!("[Sim] Resolution: {}x{}", config.capture.resolution[0], config.capture.resolution[1]);
    println!("[Sim] Num envs: {} | FPS cap: {}", tc.num_envs, tc.fps_cap);
    println!("[Sim] Total timesteps: {:?}", tc.total_timesteps);

    let device = if tc.device == "cuda" {
        candle_core::Device::new_cuda(0).unwrap_or_else(|_| {
            println!("[CUDA] Failed, falling back to CPU");
            candle_core::Device::Cpu
        })
    } else {
        candle_core::Device::Cpu
    };

    let resolution = (config.capture.resolution[0] as usize, config.capture.resolution[1] as usize);

    let mut envs: Vec<env::DungeonEnv> = (0..tc.num_envs)
        .map(|_| {
            env::DungeonEnv::new(
                resolution,
                ec.frame_stack,
                ec.max_episode_steps,
                ec.num_enemies_phase1,
                ec.num_enemies_phase2,
                ec.num_items,
                ec.view_radius,
                ec.action_repeat,
                ec.phase1_clears,
                use_real_maps,
            )
        })
        .collect();
    println!("[Env] Created {} envs", envs.len());

    let screen_channels = ec.frame_stack * 3;
    let screen_shape = (resolution.1, resolution.0);
    let mut policy = model::MinecraftDungeonsPolicy::new(
        screen_channels,
        screen_shape,
        3,
        1024,
        device.clone(),
    ).expect("Failed to create policy");

    if let Some(path) = resume {
        policy.load(path).ok();
    }

    let mut trainer = train::PPOTrainer::new(
        policy,
        tc.learning_rate,
        tc.gamma,
        tc.gae_lambda,
        tc.clip_range,
        tc.ent_coef,
        tc.vf_coef,
        tc.max_grad_norm,
        tc.n_epochs,
        tc.n_steps,
        tc.batch_size,
        device,
        screen_channels,
        resolution.1,
        resolution.0,
    );

    let total_timesteps = tc.total_timesteps;
    let num_envs = tc.num_envs;
    let n_steps = tc.n_steps;
    let shared = SharedViewerState::new();

    // Spawn training in background thread
    let shared_train = if show_viewer { Some(shared.clone()) } else { None };
    let train_running = Arc::new(AtomicBool::new(true));
    let train_running_clone = train_running.clone();

    let train_handle = std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_training_loop(
                trainer, envs, shared_train, train_running_clone,
                total_timesteps, num_envs, n_steps,
            );
        }));
        if let Err(e) = result {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            eprintln!("[Train] PANIC: {}", msg);
            std::fs::write("train_error.txt", &msg).ok();
        }
    });

    if show_viewer {
        // Run viewer on main thread (required by winit)
        let shared_viewer = shared.clone();

        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([960.0, 540.0])
                .with_title("Minecraft Dungeons AI — Rust Edition"),
            ..Default::default()
        };

        let result = eframe::run_native(
            "minecraft-dungeons-ai",
            options,
            Box::new(move |cc| {
                cc.egui_ctx.set_pixels_per_point(1.5);
                let mut app = viewer::ViewerApp::new();
                app.shared = Some(shared_viewer);
                Ok(Box::new(app))
            }),
        );

        train_running.store(false, Ordering::Relaxed);

        if let Err(e) = result {
            eprintln!("[Viewer] Error: {}", e);
        }
    } else {
        train_handle.join().ok();
    }
}

fn run_test(config_path: &str, steps: u32, use_real_maps: bool) {
    let config = config::Config::load(config_path);
    let ec = &config.env;

    println!("[Test] Running {} steps with random actions", steps);

    let resolution = (config.capture.resolution[0] as usize, config.capture.resolution[1] as usize);
    let mut env = env::DungeonEnv::new(
        resolution,
        ec.frame_stack,
        ec.max_episode_steps,
        ec.num_enemies_phase1,
        ec.num_enemies_phase2,
        ec.num_items,
        ec.view_radius,
        ec.action_repeat,
        ec.phase1_clears,
        use_real_maps,
    );

    let (_obs, info) = env.reset();
    println!("[Test] Initial HP: {:.1}%", info.hp * 100.0);

    let mut total_reward = 0.0f32;
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for step in 0..steps {
        let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
        let action = sim::Action {
            movement: [angle.cos(), angle.sin()],
            attack: rng.gen_bool(0.3),
            artifact: 0,
            dodge: rng.gen_bool(0.05),
            potion: rng.gen_bool(0.01),
        };

        let (_, reward, terminated, truncated, info) = env.step(action);
        total_reward += reward;

        if step % 100 == 0 {
            println!(
                "[Step {:>4}] Reward: {:>6.1} | HP: {:.0}% | Kills: {} | Exit: {:.1}",
                step, total_reward, info.hp * 100.0, info.kills, info.exit_dist
            );
        }

        if terminated || truncated {
            println!("[Step {:>4}] Episode ended! Victory: {}, Steps: {}", step, info.victory, info.step);
            let (_, _) = env.reset();
            total_reward = 0.0;
        }
    }

    println!("[Test] Done!");
}

fn run_training_loop(
    mut trainer: train::PPOTrainer,
    mut envs: Vec<env::DungeonEnv>,
    shared: Option<SharedViewerState>,
    running: Arc<AtomicBool>,
    total_timesteps: u64,
    num_envs: usize,
    n_steps: usize,
) {
    println!("[Train] Starting training loop...");
    println!("[Train] Press Ctrl+C to stop.");

    let mut fps_counter: u64 = 0;
    let mut fps_timer = Instant::now();
    let mut current_fps = 0.0f64;
    let steps_per_iter = (num_envs * n_steps) as u64;
    let save_every = 50_000u64;
    let mut global_best_times: HashMap<String, f32> = HashMap::new();

    loop {
        if !running.load(Ordering::Relaxed) {
            break;
        }

        let (avg_reward, last_values) = trainer.collect_rollout(&mut envs, shared.as_ref());

        for env in &envs {
            for (map, &steps) in &env.sim.best_times {
                let time = steps as f32 * 0.1;
                match global_best_times.get(map) {
                    Some(&best) if time < best => { global_best_times.insert(map.clone(), time); }
                    None => { global_best_times.insert(map.clone(), time); }
                    _ => {}
                }
            }
        }

        let stats = trainer.update(&last_values);

        fps_counter += steps_per_iter;
        let elapsed = fps_timer.elapsed();
        if elapsed >= Duration::from_secs(1) {
            current_fps = fps_counter as f64 / elapsed.as_secs_f64();
            fps_counter = 0;
            fps_timer = Instant::now();
        }

        let total = trainer.total_steps;
        println!(
            "[Step {:>8}] Reward: {:>7.1} | P_loss: {:>7.4} | V_loss: {:>7.4} | Ent: {:>5.3} | SPS: {:.0}",
            total, avg_reward, stats.policy_loss, stats.value_loss, stats.entropy, current_fps
        );

        if total / save_every > (total.saturating_sub(num_envs as u64 * n_steps as u64)) / save_every {
            trainer.save(&format!("models/checkpoint_{}.safetensors", total));
        }

        if let Some(ref s) = shared {
            s.push_reward(avg_reward);
            s.push_loss(stats.value_loss);

            if let Some(env0) = envs.get_mut(0) {
                let frame = env0.sim.render_frame_full();
                let (ow, oh) = (env0.output_w, env0.output_h);
                s.push_frame(frame, ow, oh);

                let map_frame = env0.sim.render_full_map(200, 120);
                s.push_map(map_frame, 200, 120);

                let hp = env0.sim.player.hp / env0.sim.player.max_hp;
                let attack_cd = if env0.sim.player.attack_cooldown > 0.0 { 1.0 } else { 0.0 };
                let time = env0.sim.episode_steps as f32 * 0.1;
                let map_name = match &env0.sim.generator {
                    sim::MapGenerator::Real(r) => r.current_map_name.clone(),
                    sim::MapGenerator::Synthetic(g) => g.current_map_name.clone(),
                };
                let best_time = *global_best_times.get(&map_name).unwrap_or(&f32::INFINITY);
                s.push_stats(
                    hp, attack_cd, time, best_time,
                    current_fps as f32, total, avg_reward,
                    env0.phase, &map_name,
                    stats.policy_loss, stats.value_loss, stats.entropy,
                    num_envs,
                );
            }
        }

        if total >= total_timesteps {
            break;
        }
    }

    println!("[Train] Training complete! Total steps: {}", trainer.total_steps);
    trainer.save("models/final_model.safetensors");
}
