use eframe::egui;
use super::panels;

pub struct ViewerApp {
    pub step: u64,
    pub map_name: String,
    pub entropy: f32,
    pub shared: Option<SharedViewerState>,
}

#[derive(Clone)]
pub struct SharedViewerState {
    inner: std::sync::Arc<std::sync::Mutex<ViewSnapshot>>,
}

struct ViewSnapshot {
    step: u64,
    map_name: String,
    entropy: f32,
}

impl SharedViewerState {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(ViewSnapshot {
                step: 0,
                map_name: String::new(),
                entropy: 0.0,
            }))
        }
    }

    pub fn push_stats(
        &self,
        _hp: f32,
        _attack_cooldown: f32,
        _time: f32,
        _best: f32,
        _fps: f32,
        step: u64,
        _avg_reward: f32,
        _phase: u32,
        map_name: &str,
        _policy_loss: f32,
        _value_loss: f32,
        entropy: f32,
        _num_envs: usize,
    ) {
        if let Ok(mut s) = self.inner.lock() {
            s.step = step;
            s.map_name = map_name.to_string();
            if entropy > -1.0 { s.entropy = entropy; }
        }
    }

    pub fn push_frame(&self, _frame: Vec<u8>, _w: usize, _h: usize) {}
    pub fn push_map(&self, _map: Vec<u8>, _w: usize, _h: usize) {}
    pub fn push_reward(&self, _reward: f32) {}
    pub fn push_loss(&self, _loss: f32) {}

    fn snapshot(&self) -> Option<ViewSnapshot> {
        self.inner.lock().ok().map(|s| ViewSnapshot {
            step: s.step,
            map_name: s.map_name.clone(),
            entropy: s.entropy,
        })
    }
}

impl ViewerApp {
    pub fn new() -> Self {
        Self {
            step: 0,
            map_name: String::new(),
            entropy: 0.0,
            shared: None,
        }
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(ref shared) = self.shared {
            if let Some(snap) = shared.snapshot() {
                self.step = snap.step;
                self.map_name = snap.map_name;
                self.entropy = snap.entropy;
            }
        }

        ctx.request_repaint();

        egui::CentralPanel::default().show(ctx, |ui| {
            panels::render_sidebar(ui, self);
        });
    }
}
