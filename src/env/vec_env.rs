use rayon::prelude::*;
use super::dungeon_env::{DungeonEnv, Observation, Info};

pub struct VecEnv {
    pub envs: Vec<DungeonEnv>,
}

impl VecEnv {
    pub fn new(envs: Vec<DungeonEnv>) -> Self {
        Self { envs }
    }

    pub fn reset_all(&mut self) -> Vec<(Observation, Info)> {
        self.envs.par_iter_mut().map(|env| env.reset()).collect()
    }

    pub fn step_all(&mut self, actions: Vec<super::super::sim::Action>) -> Vec<(Observation, f32, bool, bool, Info)> {
        self.envs
            .par_iter_mut()
            .zip(actions)
            .map(|(env, action)| env.step(action))
            .collect()
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }
}
