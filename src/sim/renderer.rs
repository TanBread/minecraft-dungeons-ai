use ndarray::Array2;
use super::entities::{Tile, Player, Enemy, Item, ItemKind};

// Colors as RGB
const COLOR_WALL: [u8; 3] = [30, 40, 50];
const COLOR_FLOOR: [u8; 3] = [80, 85, 90];
const COLOR_FLOOR_ALT: [u8; 3] = [70, 75, 80];
const COLOR_DOOR: [u8; 3] = [150, 100, 40];
const COLOR_EXIT: [u8; 3] = [120, 220, 0];
const COLOR_PLAYER: [u8; 3] = [255, 200, 0];
const COLOR_ENEMY: [u8; 3] = [220, 0, 0];
const COLOR_ENEMY_CHASE: [u8; 3] = [255, 0, 0];
const COLOR_ITEM_HEALTH: [u8; 3] = [0, 255, 0];
const COLOR_ITEM_AMMO: [u8; 3] = [255, 200, 0];
const COLOR_ITEM_EMERALD: [u8; 3] = [0, 180, 255];
const COLOR_FOG: [u8; 3] = [10, 15, 20];

const TILE_COLORS: [[u8; 3]; 4] = [
    COLOR_WALL,
    COLOR_FLOOR,
    COLOR_DOOR,
    COLOR_EXIT,
];

pub struct Renderer {
    pub view_radius: i32,
    pub tile_size: i32,
    pub output_w: usize,
    pub output_h: usize,
    fog_mask: Vec<Vec<f32>>,
    fog_mask_up: Vec<Vec<f32>>,
    in_range: Vec<Vec<bool>>,
    view_diameter: usize,
}

impl Renderer {
    pub fn new(view_radius: i32, tile_size: i32, output_w: usize, output_h: usize) -> Self {
        let vd = (view_radius * 2 + 1) as usize;
        let ts = tile_size as usize;
        let vpx = vd * ts;

        // Pre-compute distance fog mask
        let mut fog_mask = vec![vec![0.0f32; vd]; vd];
        let mut fog_mask_up = vec![vec![0.0f32; vpx]; vpx];
        let mut in_range = vec![vec![false; vd]; vd];

        for yi in 0..vd {
            for xi in 0..vd {
                let yy = yi as f32 - view_radius as f32;
                let xx = xi as f32 - view_radius as f32;
                let dist = (xx * xx + yy * yy).sqrt();
                fog_mask[yi][xi] = (1.0 - dist / (view_radius as f32 + 1.0)).clamp(0.5, 1.0);
                in_range[yi][xi] = dist <= view_radius as f32 + 0.5;

                // Upsample
                for ty in 0..ts {
                    for tx in 0..ts {
                        fog_mask_up[yi * ts + ty][xi * ts + tx] = fog_mask[yi][xi];
                    }
                }
            }
        }

        Self {
            view_radius,
            tile_size,
            output_w,
            output_h,
            fog_mask,
            fog_mask_up,
            in_range,
            view_diameter: vd,
        }
    }

    pub fn render(
        &self,
        grid: &Array2<u8>,
        player: &Player,
        enemies: &[Enemy],
        items: &[Item],
        _exit_pos: (usize, usize),
        explored: &Array2<bool>,
        visited: &Array2<bool>,
    ) -> Vec<u8> {
        let h = grid.nrows();
        let w = grid.ncols();
        let px = player.x as isize;
        let py = player.y as isize;
        let vr = self.view_radius as isize;
        let ts = self.tile_size as usize;
        let vd = self.view_diameter;

        // 1. Extract view region and build tile colors
        let x0 = (px - vr).max(0) as usize;
        let y0 = (py - vr).max(0) as usize;
        let x1 = ((px + vr + 1) as usize).min(w);
        let y1 = ((py + vr + 1) as usize).min(h);
        let rh = y1 - y0;
        let rw = x1 - x0;

        let ox = vr as usize - (px as usize).saturating_sub(x0);
        let oy = vr as usize - (py as usize).saturating_sub(y0);

        let mut tile_colors = vec![[0u8; 3]; vd * vd];

        // Place region tiles
        for dy in 0..rh {
            for dx in 0..rw {
                let tile = grid[[y0 + dy, x0 + dx]];
                let idx = (oy + dy) * vd + (ox + dx);
                let tile_idx = (tile as usize).min(3);
                tile_colors[idx] = TILE_COLORS[tile_idx];
            }
        }

        // Checkerboard for floors
        for dy in 0..rh {
            for dx in 0..rw {
                let gy = y0 + dy;
                let gx = x0 + dx;
                let idx = (oy + dy) * vd + (ox + dx);
                if grid[[gy, gx]] == Tile::Floor as u8 {
                    tile_colors[idx] = if (gx + gy) % 2 == 0 { COLOR_FLOOR } else { COLOR_FLOOR_ALT };
                }
            }
        }

        // Unexplored = fog
        for dy in 0..rh {
            for dx in 0..rw {
                let idx = (oy + dy) * vd + (ox + dx);
                if !explored[[y0 + dy, x0 + dx]] {
                    tile_colors[idx] = COLOR_FOG;
                }
            }
        }

        // 2. Tile each tile into ts x ts pixels
        let vpx = vd * ts;
        let mut view = vec![[0u8; 3]; vpx * vpx];
        for ty in 0..vpx {
            for tx in 0..vpx {
                let tile_y = ty / ts;
                let tile_x = tx / ts;
                let color = tile_colors[tile_y * vd + tile_x];
                view[ty * vpx + tx] = color;
            }
        }

        // 3. Apply distance fog
        for y in 0..vpx {
            for x in 0..vpx {
                let f = self.fog_mask_up[y][x];
                let idx = y * vpx + x;
                view[idx][0] = (view[idx][0] as f32 * f) as u8;
                view[idx][1] = (view[idx][1] as f32 * f) as u8;
                view[idx][2] = (view[idx][2] as f32 * f) as u8;
            }
        }

        // 4. Draw items
        for item in items {
            if item.collected {
                continue;
            }
            let dx = item.x - player.x;
            let dy = item.y - player.y;
            if dx * dx + dy * dy > ((vr + 1) as f32).powi(2) {
                continue;
            }
            let ix = ((dx + vr as f32) * ts as f32 + ts as f32 / 2.0) as usize;
            let iy = ((dy + vr as f32) * ts as f32 + ts as f32 / 2.0) as usize;
            let c = match item.kind {
                ItemKind::Health => COLOR_ITEM_HEALTH,
                ItemKind::Ammo => COLOR_ITEM_AMMO,
                ItemKind::Emerald => COLOR_ITEM_EMERALD,
            };
            draw_circle(&mut view, vpx, vpx, ix, iy, ts.max(3) / 3, c);
        }

        // 5. Draw enemies
        for enemy in enemies {
            if !enemy.alive {
                continue;
            }
            let dx = enemy.x - player.x;
            let dy = enemy.y - player.y;
            if dx * dx + dy * dy > ((vr + 1) as f32).powi(2) {
                continue;
            }
            let ex = ((dx + vr as f32) * ts as f32 + ts as f32 / 2.0) as usize;
            let ey = ((dy + vr as f32) * ts as f32 + ts as f32 / 2.0) as usize;
            let dist = (dx * dx + dy * dy).sqrt();
            let c = if dist < enemy.detection_range { COLOR_ENEMY_CHASE } else { COLOR_ENEMY };
            let r = (ts / 2).max(2);
            draw_circle(&mut view, vpx, vpx, ex, ey, r, c);
        }

        // 6. Draw player
        let center = vr as usize * ts + ts / 2;
        draw_circle(&mut view, vpx, vpx, center, center, (ts / 2).max(2), COLOR_PLAYER);

        // 7. Resize to output
        let mut frame = bilinear_resize(&view, vpx, vpx, self.output_w, self.output_h);

        // 8. Draw minimap overlay (bottom-right)
        let mm_size = 120usize.min(self.output_h / 3);
        let mm_scale = h.max(w) / mm_size + 1;
        let mm_w = w / mm_scale;
        let mm_h = h / mm_scale;
        let mut minimap = vec![[10u8; 3]; mm_w * mm_h];

        for my in 0..mm_h {
            for mx in 0..mm_w {
                let gy = my * mm_scale;
                let gx = mx * mm_scale;
                if gy < h && gx < w && explored[[gy, gx]] {
                    let tile = grid[[gy, gx]];
                    let c = if tile == Tile::Wall as u8 {
                        [30, 30, 30]
                    } else if tile == Tile::Floor as u8 || tile == Tile::Door as u8 {
                        [60, 60, 60]
                    } else if tile == Tile::Exit as u8 {
                        [0, 100, 60]
                    } else {
                        [10, 10, 10]
                    };
                    minimap[my * mm_w + mx] = c;

                    // Visited tint
                    if visited[[gy, gx]] && (tile == Tile::Floor as u8 || tile == Tile::Exit as u8) {
                        minimap[my * mm_w + mx] = [100, 140, 100];
                    }
                }
            }
        }

        // Player dot on minimap
        let mm_px = (px as usize / mm_scale).min(mm_w - 1);
        let mm_py = (py as usize / mm_scale).min(mm_h - 1);
        if mm_py < mm_h && mm_px < mm_w {
            minimap[mm_py * mm_w + mm_px] = [0, 200, 255];
        }

        // Overlay minimap onto frame
        let fy0 = self.output_h.saturating_sub(mm_h + 4);
        let fx0 = self.output_w.saturating_sub(mm_w + 4);
        for my in 0..mm_h {
            for mx in 0..mm_w {
                let fy = fy0 + my;
                let fx = fx0 + mx;
                if fy < self.output_h && fx < self.output_w {
                    let idx = (fy * self.output_w + fx) * 3;
                    let bg_r = frame[idx];
                    let bg_g = frame[idx + 1];
                    let bg_b = frame[idx + 2];
                    let mm = minimap[my * mm_w + mx];
                    frame[idx] = ((bg_r as f32 * 0.3) as u8).wrapping_add(mm[0] / 2);
                    frame[idx + 1] = ((bg_g as f32 * 0.3) as u8).wrapping_add(mm[1] / 2);
                    frame[idx + 2] = ((bg_b as f32 * 0.3) as u8).wrapping_add(mm[2] / 2);
                }
            }
        }

        frame
    }

    /// Render the full explored map (for the sidebar).
    pub fn render_full_map(
        &self,
        grid: &Array2<u8>,
        player: &Player,
        _exit_pos: (usize, usize),
        explored: &Array2<bool>,
        visited: &Array2<bool>,
        output_w: usize,
        output_h: usize,
    ) -> Vec<u8> {
        let h = grid.nrows();
        let w = grid.ncols();
        let mut minimap = vec![[10u8; 3]; w * h];

        for y in 0..h {
            for x in 0..w {
                if !explored[[y, x]] {
                    continue;
                }
                let tile = grid[[y, x]];
                let c = if tile == Tile::Wall as u8 {
                    [30, 30, 30]
                } else if tile == Tile::Floor as u8 || tile == Tile::Door as u8 {
                    [60, 60, 60]
                } else if tile == Tile::Exit as u8 {
                    [0, 100, 60]
                } else {
                    [10, 10, 10]
                };
                minimap[y * w + x] = c;

                if visited[[y, x]] && (tile == Tile::Floor as u8 || tile == Tile::Exit as u8) {
                    minimap[y * w + x] = [100, 140, 100];
                }
            }
        }

        // Player dot
        let px = player.x as usize;
        let py = player.y as usize;
        if py < h && px < w {
            minimap[py * w + px] = [0, 200, 255];
        }

        bilinear_resize(&minimap, w, h, output_w, output_h)
    }
}

fn draw_circle(buffer: &mut [[u8; 3]], width: usize, height: usize, cx: usize, cy: usize, radius: usize, color: [u8; 3]) {
    let r2 = radius * radius;
    for dy in 0..=radius * 2 {
        for dx in 0..=radius * 2 {
            let ddx = dx as isize - radius as isize;
            let ddy = dy as isize - radius as isize;
            if (ddx * ddx + ddy * ddy) as usize <= r2 {
                let x = cx + dx;
                let y = cy + dy;
                if x < width && y < height {
                    buffer[y * width + x] = color;
                }
            }
        }
    }
}

fn bilinear_resize(
    src: &[[u8; 3]],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h * 3];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let sx = (x as f64 * src_w as f64 / dst_w as f64) as usize;
            let sy = (y as f64 * src_h as f64 / dst_h as f64) as usize;
            let sx = sx.min(src_w - 1);
            let sy = sy.min(src_h - 1);
            let c = src[sy * src_w + sx];
            let idx = (y * dst_w + x) * 3;
            dst[idx] = c[0];
            dst[idx + 1] = c[1];
            dst[idx + 2] = c[2];
        }
    }
    dst
}
