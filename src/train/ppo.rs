use rand::prelude::SliceRandom;
use candle_core::{Tensor, DType, Device};
use candle_nn::{AdamW, ParamsAdamW, Optimizer};

use crate::model::MinecraftDungeonsPolicy;
use crate::model::policy::ActionSample;
use super::buffer::RolloutBuffer;

pub struct PPOTrainer {
    pub policy: MinecraftDungeonsPolicy,
    pub buffer: RolloutBuffer,
    pub lr: f64,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub clip_range: f32,
    pub ent_coef: f32,
    pub vf_coef: f32,
    pub max_grad_norm: f32,
    pub n_epochs: usize,
    pub n_steps: usize,
    pub batch_size: usize,
    pub total_steps: u64,
    pub update_count: u64,
    optimizer: AdamW,
    device: Device,
    screen_c: usize,
    screen_h: usize,
    screen_w: usize,
}

impl PPOTrainer {
    pub fn new(
        policy: MinecraftDungeonsPolicy,
        lr: f64,
        gamma: f64,
        gae_lambda: f64,
        clip_range: f64,
        ent_coef: f64,
        vf_coef: f64,
        max_grad_norm: f64,
        n_epochs: usize,
        n_steps: usize,
        batch_size: usize,
        device: Device,
        screen_c: usize,
        screen_h: usize,
        screen_w: usize,
    ) -> Self {
        let opt_params = ParamsAdamW {
            lr,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };
        let optimizer = AdamW::new(policy.varmap.all_vars(), opt_params).unwrap();

        Self {
            policy,
            buffer: RolloutBuffer::new(),
            lr,
            gamma: gamma as f32,
            gae_lambda: gae_lambda as f32,
            clip_range: clip_range as f32,
            ent_coef: ent_coef as f32,
            vf_coef: vf_coef as f32,
            max_grad_norm: max_grad_norm as f32,
            n_epochs,
            n_steps,
            batch_size,
            total_steps: 0,
            update_count: 0,
            optimizer,
            device,
            screen_c,
            screen_h,
            screen_w,
        }
    }

    pub fn collect_rollout(
        &mut self,
        envs: &mut [crate::env::dungeon_env::DungeonEnv],
        viewer: Option<&crate::viewer::SharedViewerState>,
    ) -> (f32, Vec<f32>) {
        let n = envs.len();
        let mut all_obs: Vec<crate::env::dungeon_env::Observation> = Vec::with_capacity(n);

        for env in envs.iter_mut() {
            let obs = env.current_obs();
            all_obs.push(obs);
        }

        let mut total_reward = 0.0f32;
        let rollout_start = std::time::Instant::now();

        for step in 0..self.n_steps {
            let screens: Vec<&Vec<u8>> = all_obs.iter().map(|o| &o.screen).collect();
            let memories: Vec<[f32; 3]> = all_obs.iter().map(|o| o.memory).collect();

            let screen_tensor = build_screen_tensor(&screens, &self.device, self.screen_c, self.screen_h, self.screen_w);
            let memory_tensor = build_memory_tensor(&memories, &self.device);

            let (sample, log_probs, value_tensor) = self.policy.sample_action(&screen_tensor, &memory_tensor).unwrap();
            let value = value_tensor.to_vec1::<f32>().unwrap();
            let lp_vec = log_probs.to_vec1::<f32>().unwrap();

            let move_vecs = sample.movement.to_vec2::<f32>().unwrap();
            let attack_vec = sample.attack.to_vec1::<u32>().unwrap();
            let artifact_vec = sample.artifact.to_vec1::<u32>().unwrap();
            let dodge_vec = sample.dodge.to_vec1::<u32>().unwrap();
            let potion_vec = sample.potion.to_vec1::<u32>().unwrap();

            for i in 0..n {
                let action = crate::sim::Action {
                    movement: [move_vecs[i][0], move_vecs[i][1]],
                    attack: attack_vec[i] == 1,
                    artifact: artifact_vec[i] as u8,
                    dodge: dodge_vec[i] == 1,
                    potion: potion_vec[i] == 1,
                };

                let (next_obs, reward, terminated, truncated, _info) = envs[i].step(action);
                let done = terminated || truncated;

                self.buffer.add(
                    all_obs[i].screen.clone(),
                    all_obs[i].memory,
                    move_vecs[i].clone(),
                    attack_vec[i] as u8,
                    artifact_vec[i] as u8,
                    dodge_vec[i] as u8,
                    potion_vec[i] as u8,
                    lp_vec[i],
                    reward,
                    done,
                    value[i],
                );

                total_reward += reward;
                self.total_steps += 1;

                if done {
                    let (new_obs, _) = envs[i].reset_same_map();
                    all_obs[i] = new_obs;
                } else {
                    all_obs[i] = next_obs;
                }
            }

            if let Some(v) = viewer {
                if (step + 1) % 64 == 0 || step == self.n_steps - 1 {
                    if let Some(env0) = envs.first() {
                        let frame = env0.sim.render_frame();
                        let (ow, oh) = (env0.output_w, env0.output_h);
                        v.push_frame(frame, ow, oh);
                        let map_frame = env0.sim.render_full_map(200, 120);
                        v.push_map(map_frame, 200, 120);
                        let hp = env0.sim.player.hp / env0.sim.player.max_hp;
                        let attack_cd = if env0.sim.player.attack_cooldown > 0.0 { 1.0 } else { 0.0 };
                        let time = env0.sim.episode_steps as f32 * 0.1;
                        let map_name = match &env0.sim.generator {
                            crate::sim::MapGenerator::Real(r) => r.current_map_name.clone(),
                            crate::sim::MapGenerator::Synthetic(g) => g.current_map_name.clone(),
                        };
                        let elapsed_s = rollout_start.elapsed().as_secs_f64();
                        let steps_done = ((step + 1) * n) as f64;
                        let sps = if elapsed_s > 0.0 { steps_done / elapsed_s } else { 0.0 };
                        v.push_stats(
                            hp, attack_cd, time, f32::INFINITY,
                            sps as f32, self.total_steps, total_reward / steps_done as f32,
                            env0.phase, &map_name,
                            0.0, 0.0, 0.0,
                            n,
                        );
                    }
                }
            }
        }

        let screens: Vec<&Vec<u8>> = all_obs.iter().map(|o| &o.screen).collect();
        let memories: Vec<[f32; 3]> = all_obs.iter().map(|o| o.memory).collect();
        let screen_tensor = build_screen_tensor(&screens, &self.device, self.screen_c, self.screen_h, self.screen_w);
        let memory_tensor = build_memory_tensor(&memories, &self.device);
        let last_value = self.policy.forward(&screen_tensor, &memory_tensor).unwrap()
            .value.to_vec1::<f32>().unwrap();

        (total_reward / n as f32, last_value)
    }

    pub fn update(&mut self, last_values: &[f32]) -> UpdateStats {
        let (advantages, returns) = self.buffer.compute_returns(
            self.gamma, self.gae_lambda, last_values,
        );

        let n = self.buffer.len();
        if n == 0 {
            return UpdateStats { policy_loss: 0.0, value_loss: 0.0, entropy: 0.0, learning_rate: self.lr };
        }

        let adv_mean = advantages.iter().sum::<f32>() / n as f32;
        let adv_var = advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f32>() / n as f32;
        let adv_std = adv_var.sqrt().max(1e-8);
        let normalized_adv: Vec<f32> = advantages.iter().map(|a| (a - adv_mean) / adv_std).collect();

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut total_entropy = 0.0f32;
        let mut num_batches = 0;

        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();

        for _ in 0..self.n_epochs {
            indices.shuffle(&mut rng);

            for chunk in indices.chunks(self.batch_size) {
                if chunk.is_empty() { continue; }

                let batch = self.buffer.get_batch(chunk);
                let batch_adv: Vec<f32> = chunk.iter().map(|&i| normalized_adv[i]).collect();
                let batch_ret: Vec<f32> = chunk.iter().map(|&i| returns[i]).collect();

                let screens: Vec<&Vec<u8>> = batch.screens.iter().collect();
                let screen_tensor = build_screen_tensor(&screens, &self.device, self.screen_c, self.screen_h, self.screen_w);
                let memory_tensor = build_memory_tensor(&batch.memories, &self.device);
                let move_actions = build_move_tensor(&batch.move_actions, &self.device);

                let attack_actions = Tensor::new(batch.attack_actions.as_slice(), &self.device).unwrap();
                let artifact_actions = Tensor::new(batch.artifact_actions.as_slice(), &self.device).unwrap();
                let dodge_actions = Tensor::new(batch.dodge_actions.as_slice(), &self.device).unwrap();
                let potion_actions = Tensor::new(batch.potion_actions.as_slice(), &self.device).unwrap();

                let actions = ActionSample {
                    movement: move_actions,
                    attack: attack_actions,
                    artifact: artifact_actions,
                    dodge: dodge_actions,
                    potion: potion_actions,
                };

                let old_log_probs = Tensor::new(batch.old_log_probs.as_slice(), &self.device).unwrap();
                let adv_tensor = Tensor::new(batch_adv.as_slice(), &self.device).unwrap();
                let ret_tensor = Tensor::new(batch_ret.as_slice(), &self.device).unwrap();

                let (new_log_probs, values, entropy) = self.policy.evaluate_actions(
                    &screen_tensor, &memory_tensor, &actions,
                ).unwrap();

                let ratio = new_log_probs.sub(&old_log_probs).unwrap().exp().unwrap();
                let surr1 = ratio.mul(&adv_tensor).unwrap();

                let clip_low = Tensor::new(1.0 - self.clip_range, &self.device).unwrap();
                let clip_high = Tensor::new(1.0 + self.clip_range, &self.device).unwrap();
                let ratio_clipped = ratio.broadcast_maximum(&clip_low).unwrap().broadcast_minimum(&clip_high).unwrap();
                let surr2 = ratio_clipped.mul(&adv_tensor).unwrap();

                let policy_loss = surr1.minimum(&surr2).unwrap().mean(0).unwrap();
                let value_loss = values.sub(&ret_tensor).unwrap().sqr().unwrap().mean(0).unwrap();
                let entropy_mean = entropy.mean(0).unwrap();

                let total_loss = policy_loss.broadcast_mul(&Tensor::new(-1.0f32, &self.device).unwrap()).unwrap()
                    .broadcast_add(&value_loss.broadcast_mul(&Tensor::new(self.vf_coef, &self.device).unwrap()).unwrap()).unwrap()
                    .broadcast_add(&entropy_mean.broadcast_mul(&Tensor::new(-self.ent_coef, &self.device).unwrap()).unwrap()).unwrap();

                self.optimizer.backward_step(&total_loss).unwrap();

                let batch_pl = policy_loss.to_scalar::<f32>().unwrap();
                let batch_vl = value_loss.to_scalar::<f32>().unwrap();
                let batch_ent = entropy_mean.to_scalar::<f32>().unwrap();

                self.update_count += 1;
                total_policy_loss += batch_pl;
                total_value_loss += batch_vl;
                total_entropy += batch_ent;
                num_batches += 1;
            }
        }

        self.buffer.clear();

        UpdateStats {
            policy_loss: total_policy_loss / num_batches.max(1) as f32,
            value_loss: total_value_loss / num_batches.max(1) as f32,
            entropy: total_entropy / num_batches.max(1) as f32,
            learning_rate: self.lr,
        }
    }

    pub fn save(&self, path: &str) {
        super::checkpoint::save_checkpoint(&self.policy, path);
    }

    pub fn load(&mut self, path: &str) {
        super::checkpoint::load_checkpoint(&mut self.policy, path);
    }
}

pub struct UpdateStats {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub learning_rate: f64,
}

fn build_screen_tensor(screens: &[&Vec<u8>], device: &Device, c: usize, h: usize, w: usize) -> Tensor {
    let batch_size = screens.len();
    if batch_size == 0 {
        return Tensor::zeros((1, c, h, w), DType::F32, device).unwrap();
    }
    let mut data: Vec<u8> = Vec::with_capacity(batch_size * c * h * w);
    for screen in screens {
        data.extend_from_slice(screen);
    }
    let scale = Tensor::new(1.0f32 / 255.0f32, device).unwrap();
    Tensor::from_slice(&data, (batch_size, c, h, w), device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .broadcast_mul(&scale)
        .unwrap()
}

fn build_memory_tensor(memories: &[[f32; 3]], device: &Device) -> Tensor {
    let flat: Vec<f32> = memories.iter().flat_map(|m| m.iter().copied()).collect();
    Tensor::new(flat.as_slice(), device).unwrap().reshape((memories.len(), 3)).unwrap()
}

fn build_move_tensor(moves: &[Vec<f32>], device: &Device) -> Tensor {
    let flat: Vec<f32> = moves.iter().flat_map(|m| m.iter().copied()).collect();
    Tensor::new(flat.as_slice(), device).unwrap().reshape((moves.len(), 2)).unwrap()
}
