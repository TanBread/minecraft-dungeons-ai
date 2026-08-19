use ndarray::Array2;

pub fn reveal_area(explored: &mut Array2<bool>, cx: isize, cy: isize, radius: i32, w: usize, h: usize) {
    for dy in -(radius as isize)..=(radius as isize) {
        for dx in -(radius as isize)..=(radius as isize) {
            let gx = cx + dx;
            let gy = cy + dy;
            if gx >= 0 && gy >= 0 && (gx as usize) < w && (gy as usize) < h {
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                if dist <= radius as f32 + 0.5 {
                    explored[[gy as usize, gx as usize]] = true;
                }
            }
        }
    }
}

pub fn mark_visited(visited: &mut Array2<bool>, cx: isize, cy: isize, w: usize, h: usize) {
    for dy in -1..=1 {
        for dx in -1..=1 {
            let gx = cx + dx;
            let gy = cy + dy;
            if gx >= 0 && gy >= 0 && (gx as usize) < w && (gy as usize) < h {
                visited[[gy as usize, gx as usize]] = true;
            }
        }
    }
}
