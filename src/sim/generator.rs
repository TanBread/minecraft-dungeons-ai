use std::path::{Path, PathBuf};
use rand::Rng;
use ndarray::Array2;
use super::entities::Tile;

pub enum MapGenerator {
    Synthetic(super::dungeon::DungeonGenerator),
    Real(RealMapGenerator),
}

impl MapGenerator {
    pub fn generate(&mut self) -> (Array2<u8>, (usize, usize), (usize, usize)) {
        match self {
            MapGenerator::Synthetic(g) => g.generate(),
            MapGenerator::Real(g) => g.generate(),
        }
    }

    pub fn generate_same(&mut self) -> (Array2<u8>, (usize, usize), (usize, usize)) {
        match self {
            MapGenerator::Synthetic(g) => g.generate(),
            MapGenerator::Real(g) => g.generate_same(),
        }
    }

    pub fn current_map_name(&self) -> &str {
        match self {
            MapGenerator::Synthetic(g) => &g.current_map_name,
            MapGenerator::Real(g) => &g.current_map_name,
        }
    }

    pub fn get_room(&self, x: f32, y: f32) -> isize {
        match self {
            MapGenerator::Synthetic(g) => g.get_room(x, y),
            MapGenerator::Real(g) => g.get_room(x, y),
        }
    }
}

pub struct RealMapGenerator {
    pub map_files: Vec<PathBuf>,
    pub map_id: u32,
    pub current_map_name: String,
    pub rooms: Vec<(usize, usize, usize, usize)>,
    pub cached_grid: Option<Array2<u8>>,
}

impl RealMapGenerator {
    pub fn new(maps_dir: &str) -> Self {
        let mut map_files: Vec<PathBuf> = Vec::new();
        let dir = Path::new(maps_dir);
        if dir.is_dir() {
            for entry in std::fs::read_dir(dir).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("npy") {
                    map_files.push(path);
                }
            }
            map_files.sort();
        }
        println!("[RealMapGenerator] Loaded {} real maps", map_files.len());
        Self {
            map_id: 0,
            current_map_name: "unknown".into(),
            map_files,
            rooms: Vec::new(),
            cached_grid: None,
        }
    }

    pub fn generate(&mut self) -> (Array2<u8>, (usize, usize), (usize, usize)) {
        self.map_id += 1;
        let mut rng = rand::thread_rng();

        if self.map_files.is_empty() {
            // Fallback: tiny room
            let mut grid = Array2::from_elem((20, 20), Tile::Wall as u8);
            for y in 5..15 {
                for x in 5..15 {
                    grid[[y, x]] = Tile::Floor as u8;
                }
            }
            self.current_map_name = "fallback".into();
            return (grid, (10, 10), (12, 12));
        }

        let map_path = &self.map_files[rng.gen_range(0..self.map_files.len())];
        self.current_map_name = map_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let raw = load_npy(map_path);
        let (h, w) = raw.dim();

        // Convert: 0=void->Wall, 1=wall->Wall, 2=floor->Floor
        let mut grid = Array2::from_elem((h, w), Tile::Wall as u8);
        for y in 0..h {
            for x in 0..w {
                if raw[[y, x]] == 2 {
                    grid[[y, x]] = Tile::Floor as u8;
                }
            }
        }

        // Find all floor tiles
        let mut floor_tiles: Vec<(usize, usize)> = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if grid[[y, x]] == Tile::Floor as u8 {
                    floor_tiles.push((x, y));
                }
            }
        }

        if floor_tiles.is_empty() {
            for y in 5..15.min(h) {
                for x in 5..15.min(w) {
                    grid[[y, x]] = Tile::Floor as u8;
                    floor_tiles.push((x, y));
                }
            }
        }

        // Player start: random floor tile
        let idx = rng.gen_range(0..floor_tiles.len());
        let player_pos = floor_tiles[idx];

        // Exit: farthest floor tile from player
        let exit_pos = floor_tiles
            .iter()
            .max_by_key(|&&(fx, fy)| {
                let dx = fx as f64 - player_pos.0 as f64;
                let dy = fy as f64 - player_pos.1 as f64;
                ((dx * dx + dy * dy) * 1000.0) as u64
            })
            .copied()
            .unwrap_or((player_pos.0 + 1, player_pos.1 + 1));

        grid[[exit_pos.1, exit_pos.0]] = Tile::Exit as u8;

        self.rooms = detect_rooms(&grid, h, w);

        self.cached_grid = Some(grid.clone());

        (grid, player_pos, exit_pos)
    }

    pub fn generate_same(&mut self) -> (Array2<u8>, (usize, usize), (usize, usize)) {
        self.map_id += 1;
        let mut rng = rand::thread_rng();

        let grid = match &self.cached_grid {
            Some(cached) => cached.clone(),
            None => return self.generate(),
        };

        let (h, w) = grid.dim();

        let mut floor_tiles: Vec<(usize, usize)> = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if grid[[y, x]] == Tile::Floor as u8 {
                    floor_tiles.push((x, y));
                }
            }
        }

        if floor_tiles.is_empty() {
            return self.generate();
        }

        let idx = rng.gen_range(0..floor_tiles.len());
        let player_pos = floor_tiles[idx];

        let exit_pos = floor_tiles
            .iter()
            .max_by_key(|&&(fx, fy)| {
                let dx = fx as f64 - player_pos.0 as f64;
                let dy = fy as f64 - player_pos.1 as f64;
                ((dx * dx + dy * dy) * 1000.0) as u64
            })
            .copied()
            .unwrap_or((player_pos.0 + 1, player_pos.1 + 1));

        let mut grid = grid;
        for y in 0..h {
            for x in 0..w {
                if grid[[y, x]] == Tile::Exit as u8 {
                    grid[[y, x]] = Tile::Floor as u8;
                }
            }
        }
        grid[[exit_pos.1, exit_pos.0]] = Tile::Exit as u8;

        self.rooms = detect_rooms(&grid, h, w);

        (grid, player_pos, exit_pos)
    }

    pub fn get_room(&self, x: f32, y: f32) -> isize {
        for (i, &(rx, ry, rw, rh)) in self.rooms.iter().enumerate() {
            if rx as f32 <= x && x <= (rx + rw) as f32 && ry as f32 <= y && y <= (ry + rh) as f32 {
                return i as isize;
            }
        }
        -1
    }
}

fn detect_rooms(grid: &Array2<u8>, h: usize, w: usize) -> Vec<(usize, usize, usize, usize)> {
    let sector_size = 8.max(h.min(w) / 8);
    let mut rooms = Vec::new();

    for sy in (0..h).step_by(sector_size) {
        for sx in (0..w).step_by(sector_size) {
            let ey = (sy + sector_size).min(h);
            let ex = (sx + sector_size).min(w);
            let mut floor_count = 0;
            let total = (ey - sy) * (ex - sx);
            for y in sy..ey {
                for x in sx..ex {
                    if grid[[y, x]] == Tile::Floor as u8 {
                        floor_count += 1;
                    }
                }
            }
            if floor_count as f64 / total as f64 > 0.3 {
                rooms.push((sx, sy, ex - sx, ey - sy));
            }
        }
    }

    if rooms.is_empty() {
        rooms.push((0, 0, w, h));
    }
    rooms
}

fn load_npy(path: &Path) -> Array2<u8> {
    let data = std::fs::read(path).expect("Failed to read .npy file");
    // Parse numpy .npy format
    // Magic: \x93NUMPY
    // Version: 1 byte major, 1 byte minor
    // Header length: 2 bytes (v1) or 4 bytes (v2)
    if data.len() < 6 || &data[..6] != b"\x93NUMPY" {
        panic!("Not a valid .npy file");
    }
    let major = data[6];
    let (header_len, header_start) = if major == 1 {
        let len = u16::from_le_bytes([data[8], data[9]]) as usize;
        (len, 10)
    } else {
        let len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        (len, 12)
    };

    let header_str = std::str::from_utf8(&data[header_start..header_start + header_len])
        .expect("Invalid header encoding");

    // Parse shape from header like "shape": (H, W),
    let shape_start = header_str.find("shape").expect("No shape in header");
    let paren_start = header_str[shape_start..].find('(').unwrap() + shape_start;
    let paren_end = header_str[paren_start..].find(')').unwrap() + paren_start;
    let shape_str = &header_str[paren_start + 1..paren_end];
    let parts: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let (h, w) = if parts.len() >= 2 {
        (parts[0], parts[1])
    } else if parts.len() == 1 {
        (parts[0], 1)
    } else {
        (1, 1)
    };

    // Parse dtype
    let dtype = if header_str.contains("'<u1'") || header_str.contains("'uint8'") {
        "u8"
    } else if header_str.contains("'<i4'") || header_str.contains("'int32'") {
        "i32"
    } else if header_str.contains("'<u2'") || header_str.contains("'uint16'") {
        "u16"
    } else {
        "u8" // default
    };

    let data_start = header_start + header_len;
    let raw_data = &data[data_start..];

    match dtype {
        "u8" => {
            let values: Vec<u8> = raw_data.iter().copied().take(h * w).collect();
            Array2::from_shape_vec((h, w), values).unwrap()
        }
        "u16" => {
            let values: Vec<u8> = raw_data
                .chunks_exact(2)
                .take(h * w)
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as u8)
                .collect();
            Array2::from_shape_vec((h, w), values).unwrap()
        }
        "i32" => {
            let values: Vec<u8> = raw_data
                .chunks_exact(4)
                .take(h * w)
                .map(|c| {
                    let v = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    v.clamp(0, 255) as u8
                })
                .collect();
            Array2::from_shape_vec((h, w), values).unwrap()
        }
        _ => {
            let values: Vec<u8> = raw_data.iter().copied().take(h * w).collect();
            Array2::from_shape_vec((h, w), values).unwrap()
        }
    }
}
