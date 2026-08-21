use std::collections::HashMap;
use ndarray::Array2;
use rand::Rng;

use super::entities::{Tile, Player, Enemy, Item, ItemKind};
use super::generator::MapGenerator;
use super::renderer::Renderer;
use super::{physics, combat, items as item_mod, fog};

pub struct DungeonGenerator {
    pub w: usize,
    pub h: usize,
    pub room_attempts: usize,
    pub current_map_name: String,
}

impl DungeonGenerator {
    pub fn new(w: usize, h: usize, room_attempts: usize) -> Self {
        Self {
            w,
            h,
            room_attempts,
            current_map_name: "synthetic".into(),
        }
    }

    pub fn generate(&mut self) -> (Array2<u8>, (usize, usize), (usize, usize)) {
        let mut rng = rand::thread_rng();
        let mut grid = Array2::from_elem((self.h, self.w), Tile::Wall as u8);

        let mut rooms: Vec<(usize, usize, usize, usize)> = Vec::new();

        for _ in 0..self.room_attempts {
            let rw = rng.gen_range(5..12).min(self.w.saturating_sub(2));
            let rh = rng.gen_range(5..10).min(self.h.saturating_sub(2));
            let rx = rng.gen_range(1..self.w.saturating_sub(rw + 1));
            let ry = rng.gen_range(1..self.h.saturating_sub(rh + 1));

            let mut overlaps = false;
            for &(ox, oy, ow, oh) in &rooms {
                if rx < ox + ow + 1 && rx + rw + 1 > ox && ry < oy + oh + 1 && ry + rh + 1 > oy {
                    overlaps = true;
                    break;
                }
            }
            if overlaps {
                continue;
            }

            for y in ry..ry + rh {
                for x in rx..rx + rw {
                    grid[[y, x]] = Tile::Floor as u8;
                }
            }
            rooms.push((rx, ry, rw, rh));
        }

        // Connect rooms with corridors
        for i in 1..rooms.len() {
            let (ax, ay, aw, ah) = rooms[i - 1];
            let (bx, by, bw_r, bh_r) = rooms[i];
            let cx1 = ax + aw / 2;
            let cy1 = ay + ah / 2;
            let cx2 = bx + bw_r / 2;
            let cy2 = by + bh_r / 2;

            let mut x = cx1;
            let mut y = cy1;
            while x != cx2 {
                grid[[y.min(self.h - 1), x.min(self.w - 1)]] = Tile::Floor as u8;
                if x < cx2 { x += 1; } else { x -= 1; }
            }
            while y != cy2 {
                grid[[y.min(self.h - 1), x.min(self.w - 1)]] = Tile::Floor as u8;
                if y < cy2 { y += 1; } else { y -= 1; }
            }
        }

        // Find floor tiles
        let mut floor_tiles: Vec<(usize, usize)> = Vec::new();
        for y in 0..self.h {
            for x in 0..self.w {
                if grid[[y, x]] == Tile::Floor as u8 {
                    floor_tiles.push((x, y));
                }
            }
        }

        if floor_tiles.is_empty() {
            // Fallback: create a small room
            for y in 5..15.min(self.h) {
                for x in 5..15.min(self.w) {
                    grid[[y, x]] = Tile::Floor as u8;
                    floor_tiles.push((x, y));
                }
            }
        }

        let player_pos = floor_tiles[rng.gen_range(0..floor_tiles.len())];

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

        self.current_map_name = format!("synthetic_{}", rng.gen_range(0..10000));

        (grid, player_pos, exit_pos)
    }

    pub fn get_room(&self, _x: f32, _y: f32) -> isize {
        -1
    }
}

pub struct DungeonSimulator {
    pub grid: Array2<u8>,
    pub grid_w: usize,
    pub grid_h: usize,
    pub player: Player,
    pub enemies: Vec<Enemy>,
    pub items: Vec<Item>,
    pub exit_pos: (usize, usize),
    pub explored: Array2<bool>,
    pub visited: Array2<bool>,
    pub step_count: u32,
    pub episode_steps: u32,
    pub done: bool,
    pub kills: usize,
    pub victory: bool,
    pub num_enemies: usize,
    pub num_items: usize,
    pub generator: MapGenerator,
    pub renderer: Renderer,
    pub best_times: HashMap<String, u32>,
    // Stuck detection
    stuck_count: u32,
    last_pos: (f32, f32),
    force_random: u32,
    prev_dist_to_exit: f32,
}

impl DungeonSimulator {
    pub fn new(
        generator: MapGenerator,
        num_enemies: usize,
        num_items: usize,
        view_radius: i32,
        tile_size: i32,
        output_w: usize,
        output_h: usize,
    ) -> Self {
        let renderer = Renderer::new(view_radius, tile_size, output_w, output_h);
        let grid = Array2::from_elem((40, 40), Tile::Wall as u8);

        Self {
            grid: grid.clone(),
            grid_w: 40,
            grid_h: 40,
            player: Player::new(0.0, 0.0),
            enemies: Vec::new(),
            items: Vec::new(),
            exit_pos: (0, 0),
            explored: grid.mapv(|_| false),
            visited: grid.mapv(|_| false),
            step_count: 0,
            episode_steps: 0,
            done: false,
            kills: 0,
            victory: false,
            num_enemies,
            num_items,
            generator,
            renderer,
            best_times: HashMap::new(),
            stuck_count: 0,
            last_pos: (0.0, 0.0),
            force_random: 0,
            prev_dist_to_exit: -1.0,
        }
    }

    pub fn reset(&mut self) -> Vec<u8> {
        let (grid, player_pos, exit_pos) = self.generator.generate();
        self.grid = grid;
        self.grid_h = self.grid.nrows();
        self.grid_w = self.grid.ncols();
        self.player = Player::new(player_pos.0 as f32, player_pos.1 as f32);
        self.exit_pos = exit_pos;
        self.explored = Array2::from_elem((self.grid_h, self.grid_w), false);
        self.visited = Array2::from_elem((self.grid_h, self.grid_w), false);
        self.step_count = 0;
        self.episode_steps = 0;
        self.done = false;
        self.kills = 0;
        self.victory = false;
        self.stuck_count = 0;
        self.last_pos = (self.player.x, self.player.y);
        self.force_random = 0;
        self.prev_dist_to_exit = -1.0;

        let px = self.player.x as isize;
        let py = self.player.y as isize;
        fog::mark_visited(&mut self.visited, px, py, self.grid_w, self.grid_h);
        fog::reveal_area(&mut self.explored, px, py, self.renderer.view_radius, self.grid_w, self.grid_h);

        // Spawn enemies
        self.enemies.clear();
        let mut rng = rand::thread_rng();
        for _ in 0..self.num_enemies {
            for _ in 0..1000 {
                let ex = rng.gen_range(1..self.grid_w.saturating_sub(1)).max(1);
                let ey = rng.gen_range(1..self.grid_h.saturating_sub(1)).max(1);
                if self.grid[[ey, ex]] == Tile::Floor as u8 {
                    let dist = physics::distance(ex as f32, ey as f32, player_pos.0 as f32, player_pos.1 as f32);
                    if dist > 5.0 {
                        let speed = rng.gen_range(0.02..0.05);
                        let max_hp = rng.gen_range(20.0..60.0);
                        self.enemies.push(Enemy::new(ex as f32, ey as f32, max_hp, speed, rng.gen_range(5.0..15.0)));
                        break;
                    }
                }
            }
        }

        // Spawn items
        self.items.clear();
        let kinds = [ItemKind::Health, ItemKind::Ammo, ItemKind::Emerald];
        for _ in 0..self.num_items {
            for _ in 0..1000 {
                let ix = rng.gen_range(1..self.grid_w.saturating_sub(1)).max(1);
                let iy = rng.gen_range(1..self.grid_h.saturating_sub(1)).max(1);
                if self.grid[[iy, ix]] == Tile::Floor as u8 {
                    let dist = physics::distance(ix as f32, iy as f32, player_pos.0 as f32, player_pos.1 as f32);
                    if dist > 3.0 {
                        let kind = kinds[rng.gen_range(0..3)];
                        self.items.push(Item::new(ix as f32, iy as f32, kind));
                        break;
                    }
                }
            }
        }

        self.render_frame()
    }

    pub fn reset_same_map(&mut self) -> Vec<u8> {
        let (grid, player_pos, exit_pos) = self.generator.generate_same();
        self.grid = grid;
        self.grid_h = self.grid.nrows();
        self.grid_w = self.grid.ncols();
        self.player = Player::new(player_pos.0 as f32, player_pos.1 as f32);
        self.exit_pos = exit_pos;
        self.explored = Array2::from_elem((self.grid_h, self.grid_w), false);
        self.visited = Array2::from_elem((self.grid_h, self.grid_w), false);
        self.step_count = 0;
        self.episode_steps = 0;
        self.done = false;
        self.kills = 0;
        self.victory = false;
        self.stuck_count = 0;
        self.last_pos = (self.player.x, self.player.y);
        self.force_random = 0;
        self.prev_dist_to_exit = -1.0;

        let px = self.player.x as isize;
        let py = self.player.y as isize;
        fog::mark_visited(&mut self.visited, px, py, self.grid_w, self.grid_h);
        fog::reveal_area(&mut self.explored, px, py, self.renderer.view_radius, self.grid_w, self.grid_h);

        self.enemies.clear();
        let mut rng = rand::thread_rng();
        for _ in 0..self.num_enemies {
            for _ in 0..1000 {
                let ex = rng.gen_range(1..self.grid_w.saturating_sub(1)).max(1);
                let ey = rng.gen_range(1..self.grid_h.saturating_sub(1)).max(1);
                if self.grid[[ey, ex]] == Tile::Floor as u8 {
                    let dist = physics::distance(ex as f32, ey as f32, player_pos.0 as f32, player_pos.1 as f32);
                    if dist > 5.0 {
                        let speed = rng.gen_range(0.02..0.05);
                        let max_hp = rng.gen_range(20.0..60.0);
                        self.enemies.push(Enemy::new(ex as f32, ey as f32, max_hp, speed, rng.gen_range(5.0..15.0)));
                        break;
                    }
                }
            }
        }

        self.items.clear();
        let kinds = [ItemKind::Health, ItemKind::Ammo, ItemKind::Emerald];
        for _ in 0..self.num_items {
            for _ in 0..1000 {
                let ix = rng.gen_range(1..self.grid_w.saturating_sub(1)).max(1);
                let iy = rng.gen_range(1..self.grid_h.saturating_sub(1)).max(1);
                if self.grid[[iy, ix]] == Tile::Floor as u8 {
                    let dist = physics::distance(ix as f32, iy as f32, player_pos.0 as f32, player_pos.1 as f32);
                    if dist > 3.0 {
                        let kind = kinds[rng.gen_range(0..3)];
                        self.items.push(Item::new(ix as f32, iy as f32, kind));
                        break;
                    }
                }
            }
        }

        self.render_frame()
    }

    pub fn step(&mut self, action: &Action) -> (Vec<u8>, f32, StepInfo) {
        if self.done {
            return (self.render_frame(), 0.0, self.info());
        }

        self.step_count += 1;
        self.episode_steps += 1;
        let mut reward = 0.0f32;
        let mut rng = rand::thread_rng();

        // Movement
        let mut dx = action.movement[0];
        let mut dy = action.movement[1];

        if self.force_random > 0 {
            self.force_random -= 1;
            let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            dx = angle.cos();
            dy = angle.sin();
        }

        let (new_x, new_y, _moved) = physics::move_player(
            &self.grid, self.grid_w, self.grid_h,
            self.player.x, self.player.y, dx, dy,
        );
        self.player.x = new_x;
        self.player.y = new_y;

        // Stuck detection
        let cur_pos = (self.player.x, self.player.y);
        if (cur_pos.0 - self.last_pos.0).abs() < 0.01 && (cur_pos.1 - self.last_pos.1).abs() < 0.01 {
            self.stuck_count += 1;
            if self.stuck_count > 12 {
                // Nudge: random directions with increasing force
                let nudge_angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
                let force = (self.stuck_count as f32 / 6.0).min(2.0);
                let nx = self.player.x + nudge_angle.cos() * force;
                let ny = self.player.y + nudge_angle.sin() * force;
                if physics::is_walkable(&self.grid, self.grid_w, self.grid_h, nx, ny) {
                    self.player.x = nx;
                    self.player.y = ny;
                }
                self.force_random = 8;
            }
        } else {
            self.stuck_count = 0;
        }
        self.last_pos = cur_pos;

        // Dense reward: distance to exit (shaping)
        let new_dist_to_exit = physics::distance(
            self.exit_pos.0 as f32, self.exit_pos.1 as f32,
            self.player.x, self.player.y,
        );
        let old_dist = self.prev_dist_to_exit;
        self.prev_dist_to_exit = new_dist_to_exit;
        if old_dist >= 0.0 {
            // Positive reward for getting closer, negative for moving away
            reward += (old_dist - new_dist_to_exit) * 0.5;
        }

        // Small time penalty to encourage speed
        reward -= 0.001;

        // Attack
        if action.attack && self.player.attack_cooldown <= 0.0 {
            self.player.attack_cooldown = 0.25;
            for enemy in self.enemies.iter_mut() {
                if !enemy.alive { continue; }
                let dist = physics::distance(enemy.x, enemy.y, self.player.x, self.player.y);
                if dist < 2.5 {
                    enemy.hp -= 25.0;
                    if enemy.hp <= 0.0 {
                        enemy.alive = false;
                        self.kills += 1;
                    }
                }
            }
        }

        // Potion
        if action.potion && self.player.potion_cooldown <= 0.0 {
            self.player.potion_cooldown = 30.0;
            self.player.hp = (self.player.hp + 40.0).min(self.player.max_hp);
        }

        // Dodge
        if action.dodge && self.player.dodge_cooldown <= 0.0 {
            self.player.dodge_cooldown = 1.0;
            self.player.damage_cooldown = self.player.damage_cooldown.max(0.4);
        }

        // Artifact
        if action.artifact > 0 && self.player.artifact_cooldown <= 0.0 {
            self.player.artifact_cooldown = 5.0;
            for enemy in self.enemies.iter_mut() {
                if !enemy.alive { continue; }
                let dist = physics::distance(enemy.x, enemy.y, self.player.x, self.player.y);
                if dist < 3.0 {
                    enemy.hp -= 15.0;
                    if enemy.hp <= 0.0 {
                        enemy.alive = false;
                        self.kills += 1;
                    }
                }
            }
        }

        // Tick cooldowns
        let dt = 1.0 / 60.0;
        self.player.attack_cooldown = (self.player.attack_cooldown - dt).max(0.0);
        self.player.damage_cooldown = (self.player.damage_cooldown - dt).max(0.0);
        self.player.potion_cooldown = (self.player.potion_cooldown - dt).max(0.0);
        self.player.dodge_cooldown = (self.player.dodge_cooldown - dt).max(0.0);
        self.player.artifact_cooldown = (self.player.artifact_cooldown - dt).max(0.0);

        // Enemy AI
        for enemy in self.enemies.iter_mut() {
            if !enemy.alive { continue; }
            combat::update_enemy(&self.grid, self.grid_w, self.grid_h, &self.player, enemy);
        }

        // Enemy attacks
        combat::enemy_attacks_player(&mut self.player, &self.enemies, dt);

        // Item collection
        item_mod::collect_items(&mut self.player, &mut self.items);

        // Reveal fog
        let px = self.player.x as isize;
        let py = self.player.y as isize;
        fog::reveal_area(&mut self.explored, px, py, self.renderer.view_radius, self.grid_w, self.grid_h);
        fog::mark_visited(&mut self.visited, px, py, self.grid_w, self.grid_h);

        // Check death
        if self.player.hp <= 0.0 {
            self.player.hp = 0.0;
            self.done = true;
            reward -= 50.0;
        }

        // Check exit — speedrun scoring
        let dist_to_exit = physics::distance(
            self.exit_pos.0 as f32, self.exit_pos.1 as f32,
            self.player.x, self.player.y,
        );
        if dist_to_exit < 1.0 {
            self.done = true;
            self.victory = true;
            let map_name = self.generator.current_map_name().to_string();
            let best = *self.best_times.get(&map_name).unwrap_or(&self.episode_steps);
            if self.episode_steps <= best {
                self.best_times.insert(map_name, self.episode_steps);
                reward += 200.0;
            } else {
                reward += 100.0;
            }
        }

        (self.render_frame(), reward, self.info())
    }

    pub fn render_frame(&self) -> Vec<u8> {
        self.renderer.render(
            &self.grid, &self.player, &self.enemies, &self.items,
            self.exit_pos, &self.explored, &self.visited, true,
        )
    }

    pub fn render_frame_full(&self) -> Vec<u8> {
        self.renderer.render(
            &self.grid, &self.player, &self.enemies, &self.items,
            self.exit_pos, &self.explored, &self.visited, false,
        )
    }

    pub fn render_full_map(&self, output_w: usize, output_h: usize) -> Vec<u8> {
        self.renderer.render_full_map(
            &self.grid, &self.player, self.exit_pos,
            &self.explored, &self.visited, output_w, output_h,
        )
    }

    pub fn exit_distance(&self) -> f32 {
        physics::distance(
            self.exit_pos.0 as f32, self.exit_pos.1 as f32,
            self.player.x, self.player.y,
        )
    }

    fn info(&self) -> StepInfo {
        let _total_enemies = self.enemies.len().max(1);
        let alive_enemies = self.enemies.iter().filter(|e| e.alive).count();
        let _total_items = self.items.len().max(1);
        let collected = self.items.iter().filter(|i| i.collected).count();

        StepInfo {
            step: self.step_count,
            hp: self.player.hp / self.player.max_hp,
            kills: self.kills,
            total_kills: self.enemies.len() - alive_enemies,
            victory: self.victory,
            items_collected: collected,
            total_items: self.items.len(),
            exit_dist: physics::distance(
                self.exit_pos.0 as f32, self.exit_pos.1 as f32,
                self.player.x, self.player.y,
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Action {
    pub movement: [f32; 2],
    pub attack: bool,
    pub artifact: u8,
    pub dodge: bool,
    pub potion: bool,
}

impl Default for Action {
    fn default() -> Self {
        Self {
            movement: [0.0, 0.0],
            attack: false,
            artifact: 0,
            dodge: false,
            potion: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepInfo {
    pub step: u32,
    pub hp: f32,
    pub kills: usize,
    pub total_kills: usize,
    pub victory: bool,
    pub items_collected: usize,
    pub total_items: usize,
    pub exit_dist: f32,
}
