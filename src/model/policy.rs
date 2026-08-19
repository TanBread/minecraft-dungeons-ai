use candle_core::{Tensor, DType, Result as CResult};
use candle_nn::{VarBuilder, VarMap, Linear, Module};
use super::cnn::CNNFeatureExtractor;

const MOVE_STD: f32 = 0.3;

pub struct MinecraftDungeonsPolicy {
    cnn: CNNFeatureExtractor,
    memory_encoder_1: Linear,
    memory_encoder_2: Linear,
    shared_1: Linear,
    shared_2: Linear,
    movement_head_1: Linear,
    movement_head_2: Linear,
    attack_head: Linear,
    artifact_head: Linear,
    dodge_head: Linear,
    potion_head: Linear,
    value_head: Linear,
    pub varmap: VarMap,
}

impl MinecraftDungeonsPolicy {
    pub fn new(
        screen_channels: usize,
        screen_shape: (usize, usize),
        memory_dim: usize,
        hidden_dim: usize,
        device: candle_core::Device,
    ) -> CResult<Self> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let cnn = CNNFeatureExtractor::new(vs.pp("cnn"), screen_channels)?;

        let dummy = Tensor::zeros((1, screen_channels, screen_shape.0, screen_shape.1), DType::F32, &device)?;
        let cnn_out = cnn.forward(&dummy)?;
        let flat_size = cnn_out.dims()[1];

        let memory_encoder_1 = candle_nn::linear(memory_dim, 256, vs.pp("mem1"))?;
        let memory_encoder_2 = candle_nn::linear(256, 256, vs.pp("mem2"))?;

        let combined_dim = flat_size + 256;
        let shared_1 = candle_nn::linear(combined_dim, hidden_dim, vs.pp("shared1"))?;
        let shared_2 = candle_nn::linear(hidden_dim, hidden_dim, vs.pp("shared2"))?;

        let movement_head_1 = candle_nn::linear(hidden_dim, 512, vs.pp("move1"))?;
        let movement_head_2 = candle_nn::linear(512, 2, vs.pp("move2"))?;

        let attack_head = candle_nn::linear(hidden_dim, 2, vs.pp("attack"))?;
        let artifact_head = candle_nn::linear(hidden_dim, 4, vs.pp("artifact"))?;
        let dodge_head = candle_nn::linear(hidden_dim, 2, vs.pp("dodge"))?;
        let potion_head = candle_nn::linear(hidden_dim, 2, vs.pp("potion"))?;
        let value_head = candle_nn::linear(hidden_dim, 1, vs.pp("value"))?;

        Ok(Self {
            cnn, memory_encoder_1, memory_encoder_2, shared_1, shared_2,
            movement_head_1, movement_head_2, attack_head, artifact_head,
            dodge_head, potion_head, value_head, varmap,
        })
    }

    pub fn forward(&self, screen: &Tensor, memory: &Tensor) -> CResult<PolicyOutput> {
        let screen_features = self.cnn.forward(screen)?;
        let mem = self.memory_encoder_1.forward(memory)?.relu()?;
               let mem = self.memory_encoder_2.forward(&mem)?;

        let combined = Tensor::cat(&[&screen_features, &mem], 1)?;
        let h = self.shared_1.forward(&combined)?.relu()?;
        let h = self.shared_2.forward(&h)?.relu()?;

        let move_raw = self.movement_head_1.forward(&h)?.relu()?;
        let move_mean = self.movement_head_2.forward(&move_raw)?.tanh()?;

        let attack_logits = self.attack_head.forward(&h)?;
        let artifact_logits = self.artifact_head.forward(&h)?;
        let dodge_logits = self.dodge_head.forward(&h)?;
        let potion_logits = self.potion_head.forward(&h)?;
        let value = self.value_head.forward(&h)?.squeeze(1)?;

        Ok(PolicyOutput { move_mean, attack_logits, artifact_logits, dodge_logits, potion_logits, value })
    }

    pub fn sample_action(&self, screen: &Tensor, memory: &Tensor) -> CResult<(ActionSample, Tensor, Tensor)> {
        let output = self.forward(screen, memory)?;
        let bs = screen.dim(0)?;

        let move_std = Tensor::new(MOVE_STD, &self.device())?.broadcast_as(output.move_mean.shape())?;
        let noise = Tensor::randn(0.0f32, 1.0f32, (bs, 2), &self.device())?;
        let noise_scaled = noise.broadcast_mul(&move_std)?;
        let movement = output.move_mean.broadcast_add(&noise_scaled)?.clamp(-1.0, 1.0)?;

        let move_diff = movement.broadcast_sub(&output.move_mean)?;
        let move_diff_sq = move_diff.broadcast_mul(&move_diff)?;
        let move_log_prob = move_diff_sq
            .broadcast_mul(&Tensor::new(-0.5f32, &self.device())?)?
            .broadcast_div(&move_std.broadcast_mul(&move_std)?)?
            .sum(1)?;

        let attack = categorical_sample(&output.attack_logits, &self.device())?;
        let artifact = categorical_sample(&output.artifact_logits, &self.device())?;
        let dodge = categorical_sample(&output.dodge_logits, &self.device())?;
        let potion = categorical_sample(&output.potion_logits, &self.device())?;

        let attack_log_prob = categorical_log_prob(&output.attack_logits, &attack)?;
        let artifact_log_prob = categorical_log_prob(&output.artifact_logits, &artifact)?;
        let dodge_log_prob = categorical_log_prob(&output.dodge_logits, &dodge)?;
        let potion_log_prob = categorical_log_prob(&output.potion_logits, &potion)?;

        let total_log_prob = move_log_prob
            .broadcast_add(&attack_log_prob)?
            .broadcast_add(&artifact_log_prob)?
            .broadcast_add(&dodge_log_prob)?
            .broadcast_add(&potion_log_prob)?;

        Ok((ActionSample { movement, attack, artifact, dodge, potion }, total_log_prob, output.value))
    }

    pub fn evaluate_actions(
        &self,
        screen: &Tensor,
        memory: &Tensor,
        actions: &ActionSample,
    ) -> CResult<(Tensor, Tensor, Tensor)> {
        let output = self.forward(screen, memory)?;
        let device = self.device();

        let move_std = Tensor::new(MOVE_STD, &device)?.broadcast_as(output.move_mean.shape())?;
        let move_diff = actions.movement.broadcast_sub(&output.move_mean)?;
        let move_diff_sq = move_diff.broadcast_mul(&move_diff)?;
        let move_log_prob = move_diff_sq
            .broadcast_mul(&Tensor::new(-0.5f32, &device)?)?
            .broadcast_div(&move_std.broadcast_mul(&move_std)?)?
            .sum(1)?;

        let move_entropy = {
            let sigma_sq = move_std.broadcast_mul(&move_std)?;
            let two_pi_e = Tensor::new(std::f32::consts::TAU * std::f32::consts::E, &device)?;
            let inner = sigma_sq.broadcast_mul(&two_pi_e)?;
            let log_inner = inner.log()?;
            log_inner.sum(1)?.broadcast_mul(&Tensor::new(0.5f32, &device)?)?
        };

        let attack_log_prob = categorical_log_prob(&output.attack_logits, &actions.attack)?;
        let artifact_log_prob = categorical_log_prob(&output.artifact_logits, &actions.artifact)?;
        let dodge_log_prob = categorical_log_prob(&output.dodge_logits, &actions.dodge)?;
        let potion_log_prob = categorical_log_prob(&output.potion_logits, &actions.potion)?;

        let total_log_prob = move_log_prob
            .broadcast_add(&attack_log_prob)?
            .broadcast_add(&artifact_log_prob)?
            .broadcast_add(&dodge_log_prob)?
            .broadcast_add(&potion_log_prob)?;

        let attack_entropy = categorical_entropy(&output.attack_logits)?;
        let artifact_entropy = categorical_entropy(&output.artifact_logits)?;
        let dodge_entropy = categorical_entropy(&output.dodge_logits)?;
        let potion_entropy = categorical_entropy(&output.potion_logits)?;

        let total_entropy = move_entropy
            .broadcast_add(&attack_entropy)?
            .broadcast_add(&artifact_entropy)?
            .broadcast_add(&dodge_entropy)?
            .broadcast_add(&potion_entropy)?;

        Ok((total_log_prob, output.value, total_entropy))
    }

    pub fn save(&self, path: &str) -> CResult<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> CResult<()> {
        self.varmap.load(path)?;
        Ok(())
    }

    fn device(&self) -> candle_core::Device {
        self.varmap.all_vars().first().map(|v| v.device().clone()).unwrap_or(candle_core::Device::Cpu)
    }
}

pub struct PolicyOutput {
    pub move_mean: Tensor,
    pub attack_logits: Tensor,
    pub artifact_logits: Tensor,
    pub dodge_logits: Tensor,
    pub potion_logits: Tensor,
    pub value: Tensor,
}

pub struct ActionSample {
    pub movement: Tensor,
    pub attack: Tensor,
    pub artifact: Tensor,
    pub dodge: Tensor,
    pub potion: Tensor,
}

fn log_softmax(logits: &Tensor) -> CResult<Tensor> {
    let max = logits.max(1)?;
    let shifted = logits.broadcast_sub(&max.unsqueeze(1)?)?;
    let exp = shifted.exp()?;
    let sum_exp = exp.sum(1)?.unsqueeze(1)?;
    let log_sum_exp = sum_exp.log()?;
    shifted.broadcast_sub(&log_sum_exp)
}

fn categorical_sample(logits: &Tensor, device: &candle_core::Device) -> CResult<Tensor> {
    let u = Tensor::rand(1e-8f32, 1.0f32, logits.shape(), device)?;
    let log_u = u.log()?;
    let neg_log_u = log_u.neg()?;
    let g = neg_log_u.log()?.neg()?;
    let scores = logits.broadcast_add(&g)?;
    scores.argmax(1)
}

fn categorical_log_prob(logits: &Tensor, indices: &Tensor) -> CResult<Tensor> {
    let log_probs = log_softmax(logits)?;
    let indices_i64 = indices.to_dtype(candle_core::DType::I64)?;
    let indices_exp = indices_i64.unsqueeze(1)?;
    let gathered = log_probs.gather(&indices_exp, 1)?;
    gathered.squeeze(1)
}

fn categorical_entropy(logits: &Tensor) -> CResult<Tensor> {
    let log_probs = log_softmax(logits)?;
    let probs = log_probs.exp()?;
    let entropy = log_probs.broadcast_mul(&probs)?.sum(1)?.neg()?;
    Ok(entropy)
}
