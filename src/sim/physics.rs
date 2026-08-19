use ndarray::Array2;
use super::entities::Tile;

const MOVEMENT_SPEED: f32 = 0.5;
const SLIDE_ANGLES: [f32; 5] = [1.57, -1.57, 0.78, -0.78, 3.14];

pub fn is_walkable(grid: &Array2<u8>, w: usize, h: usize, x: f32, y: f32) -> bool {
    let offsets: [f32; 3] = [-0.3, 0.0, 0.3];
    for &ox in &offsets {
        for &oy in &offsets {
            let gx = (x + ox) as isize;
            let gy = (y + oy) as isize;
            if gx < 0 || gy < 0 || gx >= w as isize || gy >= h as isize {
                return false;
            }
            if grid[[gy as usize, gx as usize]] == Tile::Wall as u8 {
                return false;
            }
        }
    }
    true
}

/// Move the player, handling wall collision and perpendicular sliding.
/// Returns (new_x, new_y, actually_moved).
pub fn move_player(
    grid: &Array2<u8>,
    w: usize,
    h: usize,
    x: f32,
    y: f32,
    dx: f32,
    dy: f32,
) -> (f32, f32, bool) {
    let mag = (dx * dx + dy * dy).sqrt();
    let (ndx, ndy) = if mag > 1.0 {
        (dx / mag, dy / mag)
    } else {
        (dx, dy)
    };

    let new_x = x + ndx * MOVEMENT_SPEED;
    let new_y = y + ndy * MOVEMENT_SPEED;

    // Try full movement
    if is_walkable(grid, w, h, new_x, new_y) {
        return (new_x, new_y, true);
    }
    // Slide along X only
    if is_walkable(grid, w, h, new_x, y) {
        return (new_x, y, true);
    }
    // Slide along Y only
    if is_walkable(grid, w, h, x, new_y) {
        return (x, new_y, true);
    }

    // Both axes blocked — try perpendicular angles
    let base_angle = ndy.atan2(ndx);
    for &offset in &SLIDE_ANGLES {
        let slide_angle = base_angle + offset;
        let sx = x + slide_angle.cos() * 0.4;
        let sy = y + slide_angle.sin() * 0.4;
        if is_walkable(grid, w, h, sx, sy) {
            return (sx, sy, true);
        }
    }

    (x, y, false)
}

pub fn distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    (dx * dx + dy * dy).sqrt()
}
