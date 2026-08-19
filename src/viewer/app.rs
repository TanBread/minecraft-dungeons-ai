use eframe::egui;
use super::panels;

pub struct ViewerApp {
    pub frame_data: Vec<u8>,
    pub frame_w: usize,
    pub frame_h: usize,
    pub frame_dirty: bool,
    pub texture_handle: Option<egui::TextureHandle>,
    pub map_data: Vec<u8>,
    pub map_w: usize,
    pub map_h: usize,
    pub map_dirty: bool,
    pub map_texture_handle: Option<egui::TextureHandle>,
    pub hp: f32,
    pub attack_cooldown: f32,
    pub time_elapsed: f32,
    pub best_time: f32,
    pub fps: f32,
    pub step: u64,
    pub avg_reward: f32,
    pub num_envs: usize,
    pub reward_history: Vec<f32>,
    pub loss_history: Vec<f32>,
    pub phase: u32,
    pub map_name: String,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub shared: Option<super::app::SharedViewerState>,
}

#[derive(Clone)]
pub struct SharedViewerState {
    pub(crate) inner: std::sync::Arc<std::sync::Mutex<ViewSnapshot>>,
}

struct ViewSnapshot {
    frame_data: Vec<u8>,
    frame_w: usize,
    frame_h: usize,
    map_data: Vec<u8>,
    map_w: usize,
    map_h: usize,
    hp: f32,
    attack_cooldown: f32,
    time_elapsed: f32,
    best_time: f32,
    fps: f32,
    step: u64,
    avg_reward: f32,
    phase: u32,
    map_name: String,
    policy_loss: f32,
    value_loss: f32,
    entropy: f32,
    reward_history: Vec<f32>,
    loss_history: Vec<f32>,
    num_envs: usize,
}

impl SharedViewerState {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(ViewSnapshot {
                frame_data: vec![0; 162 * 90 * 3],
                frame_w: 162,
                frame_h: 90,
                map_data: vec![0; 120 * 80 * 3],
                map_w: 120,
                map_h: 80,
                hp: 1.0,
                attack_cooldown: 0.0,
                time_elapsed: 0.0,
                best_time: f32::INFINITY,
                fps: 0.0,
                step: 0,
                avg_reward: 0.0,
                phase: 1,
                map_name: String::new(),
                policy_loss: 0.0,
                value_loss: 0.0,
                entropy: 0.0,
                reward_history: Vec::new(),
                loss_history: Vec::new(),
                num_envs: 48,
            }))
        }
    }

    pub fn push_frame(&self, frame: Vec<u8>, w: usize, h: usize) {
        if let Ok(mut s) = self.inner.lock() {
            s.frame_data = frame;
            s.frame_w = w;
            s.frame_h = h;
        }
    }

    pub fn push_map(&self, map: Vec<u8>, w: usize, h: usize) {
        if let Ok(mut s) = self.inner.lock() {
            s.map_data = map;
            s.map_w = w;
            s.map_h = h;
        }
    }

    pub fn push_stats(
        &self,
        hp: f32,
        attack_cooldown: f32,
        time: f32,
        best: f32,
        fps: f32,
        step: u64,
        avg_reward: f32,
        phase: u32,
        map_name: &str,
        policy_loss: f32,
        value_loss: f32,
        entropy: f32,
        num_envs: usize,
    ) {
        if let Ok(mut s) = self.inner.lock() {
            s.hp = hp;
            s.attack_cooldown = attack_cooldown;
            s.time_elapsed = time;
            s.best_time = best;
            s.fps = fps;
            s.step = step;
            s.avg_reward = avg_reward;
            s.phase = phase;
            s.map_name = map_name.to_string();
            s.policy_loss = policy_loss;
            s.value_loss = value_loss;
            s.entropy = entropy;
            s.num_envs = num_envs;
        }
    }

    pub fn push_reward(&self, reward: f32) {
        if let Ok(mut s) = self.inner.lock() {
            s.reward_history.push(reward);
            if s.reward_history.len() > 200 {
                s.reward_history.remove(0);
            }
        }
    }

    pub fn push_loss(&self, loss: f32) {
        if let Ok(mut s) = self.inner.lock() {
            s.loss_history.push(loss);
            if s.loss_history.len() > 200 {
                s.loss_history.remove(0);
            }
        }
    }

    fn snapshot(&self) -> Option<ViewSnapshot> {
        self.inner.lock().ok().map(|s| ViewSnapshot {
            frame_data: s.frame_data.clone(),
            frame_w: s.frame_w,
            frame_h: s.frame_h,
            map_data: s.map_data.clone(),
            map_w: s.map_w,
            map_h: s.map_h,
            hp: s.hp,
            attack_cooldown: s.attack_cooldown,
            time_elapsed: s.time_elapsed,
            best_time: s.best_time,
            fps: s.fps,
            step: s.step,
            avg_reward: s.avg_reward,
            phase: s.phase,
            map_name: s.map_name.clone(),
            policy_loss: s.policy_loss,
            value_loss: s.value_loss,
            entropy: s.entropy,
            reward_history: s.reward_history.clone(),
            loss_history: s.loss_history.clone(),
            num_envs: s.num_envs,
        })
    }
}

impl ViewerApp {
    pub fn new() -> Self {
        Self {
            frame_data: vec![0; 162 * 90 * 3],
            frame_w: 162,
            frame_h: 90,
            frame_dirty: true,
            texture_handle: None,
            map_data: vec![0; 120 * 80 * 3],
            map_w: 120,
            map_h: 80,
            map_dirty: true,
            map_texture_handle: None,
            hp: 1.0,
            attack_cooldown: 0.0,
            time_elapsed: 0.0,
            best_time: f32::INFINITY,
            fps: 0.0,
            step: 0,
            avg_reward: 0.0,
            num_envs: 2,
            reward_history: Vec::new(),
            loss_history: Vec::new(),
            phase: 1,
            map_name: String::new(),
            policy_loss: 0.0,
            value_loss: 0.0,
            entropy: 0.0,
            shared: None,
        }
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(ref shared) = self.shared {
            if let Some(snap) = shared.snapshot() {
                if snap.frame_data != self.frame_data || snap.frame_w != self.frame_w || snap.frame_h != self.frame_h {
                    self.frame_data = snap.frame_data;
                    self.frame_w = snap.frame_w;
                    self.frame_h = snap.frame_h;
                    self.frame_dirty = true;
                }
                if snap.map_data != self.map_data || snap.map_w != self.map_w || snap.map_h != self.map_h {
                    self.map_data = snap.map_data;
                    self.map_w = snap.map_w;
                    self.map_h = snap.map_h;
                    self.map_dirty = true;
                }
                self.hp = snap.hp;
                self.attack_cooldown = snap.attack_cooldown;
                self.time_elapsed = snap.time_elapsed;
                self.best_time = snap.best_time;
                self.fps = snap.fps;
                self.step = snap.step;
                self.avg_reward = snap.avg_reward;
                self.phase = snap.phase;
                self.map_name = snap.map_name;
                self.policy_loss = snap.policy_loss;
                self.value_loss = snap.value_loss;
                self.entropy = snap.entropy;
                self.reward_history = snap.reward_history;
                self.loss_history = snap.loss_history;
                self.num_envs = snap.num_envs;
            }
        }

        ctx.request_repaint();

        egui::SidePanel::right("sidebar")
            .resizable(true)
            .default_width(200.0)
            .show(ctx, |ui| {
                panels::render_sidebar(ui, self);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            panels::render_main_view(ui, self);
        });
    }
}
