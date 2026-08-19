use eframe::egui;
use super::app::ViewerApp;

fn format_step(steps: u64) -> String {
    if steps >= 1_000_000_000 {
        format!("{:.2}B", steps as f64 / 1_000_000_000.0)
    } else if steps >= 1_000_000 {
        format!("{:.2}M", steps as f64 / 1_000_000.0)
    } else if steps >= 1_000 {
        format!("{:.1}K", steps as f64 / 1_000.0)
    } else {
        format!("{}", steps)
    }
}

pub fn render_sidebar(ui: &mut egui::Ui, app: &mut ViewerApp) {
    ui.set_min_width(ui.available_width());

    let w = ui.available_width();

    ui.group(|ui| {
        ui.set_min_width(w);

        egui::Grid::new("stats_grid").num_columns(2).show(ui, |ui| {
            ui.label("STEP");
            ui.label(format!("{}", format_step(app.step)));
            ui.end_row();

            ui.label("SPS");
            ui.label(format!("{:.0} | {:.0}/m | {:.0}/h", app.fps, app.fps * 60.0, app.fps * 3600.0));
            ui.end_row();

            ui.label("ENVS");
            ui.label(format!("{}", app.num_envs));
            ui.end_row();

            ui.label("PHASE");
            ui.label(format!("{} — {}", app.phase, app.map_name));
            ui.end_row();

            ui.label("TIME");
            let mins = (app.time_elapsed / 60.0) as u32;
            let secs = (app.time_elapsed % 60.0) as u32;
            let time_color = if app.best_time < f32::INFINITY && app.time_elapsed < app.best_time {
                egui::Color32::from_rgb(0, 200, 0)
            } else if app.best_time < f32::INFINITY {
                egui::Color32::from_rgb(200, 0, 0)
            } else {
                egui::Color32::from_rgb(200, 200, 200)
            };
            ui.colored_label(time_color, format!("{}:{:02}", mins, secs));
            ui.end_row();

            ui.label("TTB");
            if app.best_time < f32::INFINITY {
                let bmins = (app.best_time / 60.0) as u32;
                let bsecs = (app.best_time % 60.0) as u32;
                ui.label(format!("{}:{:02}", bmins, bsecs));
            } else {
                ui.label("--:--");
            }
            ui.end_row();

            ui.label("ENTROPY");
            ui.label(format!("{:.3}", app.entropy));
            ui.end_row();
        });
    });

    ui.add_space(4.0);

    // Collapsible: Full Map
    egui::CollapsingHeader::new("FULL MAP").default_open(false).show(ui, |ui| {
        if !app.map_data.is_empty() && app.map_w > 0 && app.map_h > 0 {
            if app.map_dirty {
                let tex: egui::ColorImage = egui::ColorImage::from_rgb(
                    [app.map_w, app.map_h],
                    &app.map_data,
                );
                app.map_texture_handle = Some(ui.ctx().load_texture(
                    "full_map", tex, egui::TextureOptions::NEAREST,
                ));
                app.map_dirty = false;
            }
            if let Some(ref handle) = app.map_texture_handle {
                let available = ui.available_size();
                let scale = (available.x / app.map_w as f32).min(available.y / app.map_h as f32);
                let size = egui::vec2(app.map_w as f32 * scale, app.map_h as f32 * scale);
                let resp = ui.allocate_rect(
                    egui::Rect::from_min_size(ui.cursor().min, size),
                    egui::Sense::hover(),
                );
                ui.painter().image(
                    handle.id(), resp.rect,
                    egui::Rect::from_min_size(egui::Pos2::ZERO, handle.size_vec2()),
                    egui::Color32::WHITE,
                );
            }
        }
    });

    // Collapsible: Reward Chart
    egui::CollapsingHeader::new("REWARD").default_open(false).show(ui, |ui| {
        if app.reward_history.len() > 1 {
            let available = ui.available_size();
            let (rect, _) = ui.allocate_at_least(available, egui::Sense::hover());
            ui.painter().rect_filled(rect, 2.0, egui::Color32::from_rgb(20, 20, 20));

            let min_r = app.reward_history.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_r = app.reward_history.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = (max_r - min_r).abs().max(1.0);

            for i in 1..app.reward_history.len() {
                let x1 = rect.left() + (i as f32 - 1.0) / (app.reward_history.len() as f32 - 1.0) * rect.width();
                let x2 = rect.left() + i as f32 / (app.reward_history.len() as f32 - 1.0) * rect.width();
                let y1 = rect.center().y - (app.reward_history[i-1] - min_r) / range * rect.height() * 0.4;
                let y2 = rect.center().y - (app.reward_history[i] - min_r) / range * rect.height() * 0.4;

                let color = if app.reward_history[i] >= 0.0 {
                    egui::Color32::from_rgb(0, 200, 0)
                } else {
                    egui::Color32::from_rgb(200, 0, 0)
                };

                ui.painter().line_segment(
                    [egui::pos2(x1, y1), egui::pos2(x2, y2)],
                    egui::Stroke::new(1.5_f32, color),
                );
            }
        }
    });

    // Collapsible: Value Loss
    egui::CollapsingHeader::new("VALUE LOSS").default_open(false).show(ui, |ui| {
        if app.loss_history.len() > 1 {
            let available = ui.available_size();
            let (rect, _) = ui.allocate_at_least(available, egui::Sense::hover());
            ui.painter().rect_filled(rect, 2.0, egui::Color32::from_rgb(20, 20, 20));

            let min_l = app.loss_history.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_l = app.loss_history.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = (max_l - min_l).abs().max(1e-6);

            for i in 1..app.loss_history.len() {
                let x1 = rect.left() + (i as f32 - 1.0) / (app.loss_history.len() as f32 - 1.0) * rect.width();
                let x2 = rect.left() + i as f32 / (app.loss_history.len() as f32 - 1.0) * rect.width();
                let y1 = rect.bottom() - (app.loss_history[i-1] - min_l) / range * rect.height() * 0.8;
                let y2 = rect.bottom() - (app.loss_history[i] - min_l) / range * rect.height() * 0.8;

                ui.painter().line_segment(
                    [egui::pos2(x1, y1), egui::pos2(x2, y2)],
                    egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(100, 150, 255)),
                );
            }
        }
    });
}

pub fn render_main_view(ui: &mut egui::Ui, app: &mut ViewerApp) {
    if !app.frame_data.is_empty() && app.frame_w > 0 && app.frame_h > 0 {
        if app.frame_dirty {
            let tex: egui::ColorImage = egui::ColorImage::from_rgb(
                [app.frame_w, app.frame_h],
                &app.frame_data,
            );
            app.texture_handle = Some(ui.ctx().load_texture(
                "frame", tex, egui::TextureOptions::NEAREST,
            ));
            app.frame_dirty = false;
        }

        if let Some(ref handle) = app.texture_handle {
            let available = ui.available_size();
            let scale = (available.x / app.frame_w as f32).min(available.y / app.frame_h as f32);
            let size = egui::vec2(
                app.frame_w as f32 * scale,
                app.frame_h as f32 * scale,
            );
            let resp = ui.allocate_rect(
                egui::Rect::from_min_size(ui.cursor().min, size),
                egui::Sense::hover(),
            );
            ui.painter().image(
                handle.id(),
                resp.rect,
                egui::Rect::from_min_size(egui::Pos2::ZERO, handle.size_vec2()),
                egui::Color32::WHITE,
            );
        }
    } else {
        ui.centered_and_justified(|ui| {
            ui.label("Waiting for training data...");
        });
    }
}
