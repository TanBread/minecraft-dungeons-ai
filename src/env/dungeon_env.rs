use super::super::sim::{DungeonSimulator, Action, MapGenerator, DungeonGenerator, RealMapGenerator};

pub struct DungeonEnv {
    pub sim: DungeonSimulator,
    pub frame_stack: usize,
    pub max_episode_steps: u32,
    pub action_repeat: usize,
    pub step_count: u32,
    pub episode_reward: f32,
    pub frame_buffer: Vec<Vec<u8>>,
    pub last_frame: Vec<u8>,
    pub output_w: usize,
    pub output_h: usize,
    pub phase: u32,
    pub phase1_clears: u32,
    pub phase1_cleared: u32,
    pub map_num: u32,
    pub wins_on_map: u32,
    pub wins_needed: u32,
    pub num_enemies_phase1: usize,
    pub num_enemies_phase2: usize,
}

impl DungeonEnv {
    pub fn new(
        resolution: (usize, usize),
        frame_stack: usize,
        max_episode_steps: u32,
        num_enemies_phase1: usize,
        num_enemies_phase2: usize,
        num_items: usize,
        view_radius: i32,
        action_repeat: usize,
        phase1_clears: u32,
        use_real_maps: bool,
    ) -> Self {
        let generator: MapGenerator = if use_real_maps {
            MapGenerator::Real(RealMapGenerator::new("maps/grids"))
        } else {
            MapGenerator::Synthetic(DungeonGenerator::new(40, 40, 8))
        };

        let mut sim = DungeonSimulator::new(
            generator,
            num_enemies_phase1,
            num_items,
            view_radius,
            8,
            resolution.0,
            resolution.1,
        );
        let last_frame = sim.reset();

        println!("[DungeonEnv] PHASE {} — Exploration (no enemies)", 1);
        println!("[DungeonEnv] Map #0 generated");

        Self {
            sim,
            frame_stack,
            max_episode_steps,
            action_repeat,
            step_count: 0,
            episode_reward: 0.0,
            frame_buffer: vec![last_frame.clone(); frame_stack],
            last_frame,
            output_w: resolution.0,
            output_h: resolution.1,
            phase: 1,
            phase1_clears,
            phase1_cleared: 0,
            map_num: 0,
            wins_on_map: 0,
            wins_needed: 100,
            num_enemies_phase1,
            num_enemies_phase2,
        }
    }

    pub fn reset(&mut self) -> (Observation, Info) {
        self.step_count = 0;
        self.episode_reward = 0.0;
        self.frame_buffer.clear();
        let frame = self.sim.reset();
        self.last_frame = frame.clone();
        for _ in 0..self.frame_stack {
            self.frame_buffer.push(frame.clone());
        }
        (self.obs(), self.info())
    }

    pub fn reset_same_map(&mut self) -> (Observation, Info) {
        self.step_count = 0;
        self.episode_reward = 0.0;
        self.frame_buffer.clear();
        let frame = self.sim.reset_same_map();
        self.last_frame = frame.clone();
        for _ in 0..self.frame_stack {
            self.frame_buffer.push(frame.clone());
        }
        (self.obs(), self.info())
    }

    pub fn step(&mut self, action: Action) -> (Observation, f32, bool, bool, Info) {
        self.step_count += 1;
        let mut total_reward = 0.0f32;

        for _ in 0..self.action_repeat {
            let (_, reward, _) = self.sim.step(&action);
            total_reward += reward;
            self.last_frame = self.sim.render_frame();
            if self.sim.done {
                break;
            }
        }

        self.episode_reward += total_reward;
        self.frame_buffer.push(self.last_frame.clone());
        if self.frame_buffer.len() > self.frame_stack {
            self.frame_buffer.remove(0);
        }

        let terminated = self.sim.done;
        let truncated = self.step_count >= self.max_episode_steps;

        if terminated && self.sim.victory {
            self.wins_on_map += 1;
            println!("[Phase {}] Map #{} cleared! ({}/{})", self.phase, self.map_num, self.wins_on_map, self.wins_needed);
            if self.wins_on_map >= self.wins_needed {
                self.map_num += 1;
                self.wins_on_map = 0;

                match &mut self.sim.generator {
                    MapGenerator::Real(r) => {
                        r.cached_grid = None;
                    }
                    MapGenerator::Synthetic(g) => {
                        *g = DungeonGenerator::new(40, 40, 8);
                    }
                }

                if self.phase == 1 {
                    self.phase1_cleared += self.wins_needed;
                    if self.phase1_cleared >= self.phase1_clears * self.wins_needed {
                        self.advance_to_phase2();
                    }
                }

                println!("[Phase {}] New map #{} generated!", self.phase, self.map_num);
            }
        }

        (self.obs(), total_reward, terminated, truncated, self.info())
    }

    fn advance_to_phase2(&mut self) {
        self.phase = 2;
        self.map_num = 0;
        self.wins_on_map = 0;
        self.sim.num_enemies = self.num_enemies_phase2;
        if matches!(&self.sim.generator, MapGenerator::Real(_)) {
        } else {
            self.sim.generator = MapGenerator::Synthetic(DungeonGenerator::new(40, 40, 8));
        }
        println!("\n{:=<50}", "");
        println!("[DungeonEnv] PHASE 2 — COMBAT! Enemies enabled ({})", self.num_enemies_phase2);
        println!("{:=<50}\n", "");
    }

    fn obs(&self) -> Observation {
        let h = self.output_h;
        let w = self.output_w;
        let c = self.frame_stack * 3;

        let mut screen = vec![0u8; c * h * w];
        for (i, frame) in self.frame_buffer.iter().enumerate() {
            for y in 0..h {
                for x in 0..w {
                    let src_idx = (y * w + x) * 3;
                    let dst_base = i * 3 * h * w + y * w + x;
                    if src_idx + 2 < frame.len() {
                        screen[dst_base] = frame[src_idx];
                        screen[dst_base + h * w] = frame[src_idx + 1];
                        screen[dst_base + 2 * h * w] = frame[src_idx + 2];
                    }
                }
            }
        }

        let total_enemies = self.sim.enemies.len().max(1);
        let alive_enemies = self.sim.enemies.iter().filter(|e| e.alive).count();
        let kills_ratio = (self.sim.enemies.len() - alive_enemies) as f32 / total_enemies as f32;
        let total_items = self.sim.items.len().max(1);
        let collected = self.sim.items.iter().filter(|i| i.collected).count();
        let items_ratio = collected as f32 / total_items as f32;
        let hp = self.sim.player.hp / self.sim.player.max_hp;

        Observation {
            screen,
            memory: [hp, kills_ratio, items_ratio],
        }
    }

    pub fn current_obs(&self) -> Observation {
        self.obs()
    }

    fn info(&self) -> Info {
        Info {
            step: self.step_count,
            total_reward: self.episode_reward,
            hp: self.sim.player.hp / self.sim.player.max_hp,
            kills: self.sim.kills,
            victory: self.sim.victory,
            exit_dist: self.sim.exit_distance(),
            phase: self.phase,
            map_num: self.map_num,
            wins_on_map: self.wins_on_map,
        }
    }
}

pub struct Observation {
    pub screen: Vec<u8>,
    pub memory: [f32; 3],
}

pub struct Info {
    pub step: u32,
    pub total_reward: f32,
    pub hp: f32,
    pub kills: usize,
    pub victory: bool,
    pub exit_dist: f32,
    pub phase: u32,
    pub map_num: u32,
    pub wins_on_map: u32,
}
