
pub struct RolloutBuffer {
    pub screens: Vec<Vec<u8>>,
    pub memories: Vec<[f32; 3]>,
    pub move_actions: Vec<Vec<f32>>,
    pub attack_actions: Vec<u8>,
    pub artifact_actions: Vec<u8>,
    pub dodge_actions: Vec<u8>,
    pub potion_actions: Vec<u8>,
    pub log_probs: Vec<f32>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub values: Vec<f32>,
}

impl RolloutBuffer {
    pub fn new() -> Self {
        Self {
            screens: Vec::new(),
            memories: Vec::new(),
            move_actions: Vec::new(),
            attack_actions: Vec::new(),
            artifact_actions: Vec::new(),
            dodge_actions: Vec::new(),
            potion_actions: Vec::new(),
            log_probs: Vec::new(),
            rewards: Vec::new(),
            dones: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn add(
        &mut self,
        screen: Vec<u8>,
        memory: [f32; 3],
        move_action: Vec<f32>,
        attack: u8,
        artifact: u8,
        dodge: u8,
        potion: u8,
        log_prob: f32,
        reward: f32,
        done: bool,
        value: f32,
    ) {
        self.screens.push(screen);
        self.memories.push(memory);
        self.move_actions.push(move_action);
        self.attack_actions.push(attack);
        self.artifact_actions.push(artifact);
        self.dodge_actions.push(dodge);
        self.potion_actions.push(potion);
        self.log_probs.push(log_prob);
        self.rewards.push(reward);
        self.dones.push(done);
        self.values.push(value);
    }

    pub fn clear(&mut self) {
        self.screens = Vec::new();
        self.memories = Vec::new();
        self.move_actions = Vec::new();
        self.attack_actions = Vec::new();
        self.artifact_actions = Vec::new();
        self.dodge_actions = Vec::new();
        self.potion_actions = Vec::new();
        self.log_probs = Vec::new();
        self.rewards = Vec::new();
        self.dones = Vec::new();
        self.values = Vec::new();
    }

    pub fn len(&self) -> usize {
        self.rewards.len()
    }

    pub fn compute_returns(&self, gamma: f32, gae_lambda: f32, last_values: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = self.rewards.len();
        let n_envs = last_values.len();
        let n_steps = n / n_envs;
        let mut advantages = vec![0.0f32; n];
        let mut returns = vec![0.0f32; n];

        for env in 0..n_envs {
            let mut last_gae = 0.0f32;
            for t in (0..n_steps).rev() {
                let idx = t * n_envs + env;
                let next_value = if t < n_steps - 1 {
                    self.values[(t + 1) * n_envs + env]
                } else {
                    last_values[env]
                };
                let next_non_terminal = if self.dones[idx] { 0.0f32 } else { 1.0f32 };
                let delta = self.rewards[idx] + gamma * next_value * next_non_terminal - self.values[idx];
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae;
                advantages[idx] = last_gae;
            }
        }

        for t in 0..n {
            returns[t] = advantages[t] + self.values[t];
        }

        (advantages, returns)
    }

    pub fn get_batch(&self, indices: &[usize]) -> BatchData {
        let screens: Vec<Vec<u8>> = indices.iter().map(|&i| self.screens[i].clone()).collect();
        let memories: Vec<[f32; 3]> = indices.iter().map(|&i| self.memories[i]).collect();
        let move_actions: Vec<Vec<f32>> = indices.iter().map(|&i| self.move_actions[i].clone()).collect();
        let attack_actions: Vec<u8> = indices.iter().map(|&i| self.attack_actions[i]).collect();
        let artifact_actions: Vec<u8> = indices.iter().map(|&i| self.artifact_actions[i]).collect();
        let dodge_actions: Vec<u8> = indices.iter().map(|&i| self.dodge_actions[i]).collect();
        let potion_actions: Vec<u8> = indices.iter().map(|&i| self.potion_actions[i]).collect();
        let old_log_probs: Vec<f32> = indices.iter().map(|&i| self.log_probs[i]).collect();

        BatchData {
            screens,
            memories,
            move_actions,
            attack_actions,
            artifact_actions,
            dodge_actions,
            potion_actions,
            old_log_probs,
        }
    }
}

pub struct BatchData {
    pub screens: Vec<Vec<u8>>,
    pub memories: Vec<[f32; 3]>,
    pub move_actions: Vec<Vec<f32>>,
    pub attack_actions: Vec<u8>,
    pub artifact_actions: Vec<u8>,
    pub dodge_actions: Vec<u8>,
    pub potion_actions: Vec<u8>,
    pub old_log_probs: Vec<f32>,
}
