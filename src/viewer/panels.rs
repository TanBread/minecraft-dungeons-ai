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

    ui.heading("Training Stats");
    ui.add_space(4.0);

    egui::Grid::new("stats_grid").num_columns(2).show(ui, |ui| {
        ui.label("STEPS");
        ui.label(format!("{}", format_step(app.step)));
        ui.end_row();

        ui.label("MAP");
        ui.label(&app.map_name);
        ui.end_row();

        ui.label("ENTROPY");
        ui.label(format!("{:.3}", app.entropy));
        ui.end_row();
    });
}

pub fn render_main_view(ui: &mut egui::Ui, _app: &mut ViewerApp) {
    ui.centered_and_justified(|ui| {
        ui.label("Training in progress...");
    });
}
