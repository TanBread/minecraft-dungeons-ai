use ndarray::Array2;
use super::entities::{Player, Enemy};
use super::physics::{is_walkable, distance};

pub fn update_enemy(
    grid: &Array2<u8>,
    w: usize,
    h: usize,
    player: &Player,
    enemy: &mut Enemy,
) {
    let dist = distance(enemy.x, enemy.y, player.x, player.y);

    if dist < enemy.detection_range {
        // Chase player
        let dx = player.x - enemy.x;
        let dy = player.y - enemy.y;
        let mag = dist.max(0.01);
        let new_x = enemy.x + (dx / mag) * enemy.speed;
        let new_y = enemy.y + (dy / mag) * enemy.speed;

        if is_walkable(grid, w, h, new_x, new_y) {
            enemy.x = new_x;
            enemy.y = new_y;
        } else if is_walkable(grid, w, h, new_x, enemy.y) {
            enemy.x = new_x;
        } else if is_walkable(grid, w, h, enemy.x, new_y) {
            enemy.y = new_y;
        }
    } else {
        // Patrol
        use rand::Rng;
        let mut rng = rand::thread_rng();
        enemy.patrol_timer -= 1;
        if enemy.patrol_timer <= 0 {
            let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            enemy.patrol_dx = angle.cos() * enemy.speed * 0.5;
            enemy.patrol_dy = angle.sin() * enemy.speed * 0.5;
            enemy.patrol_timer = rng.gen_range(20..=60);
        }

        let new_x = enemy.x + enemy.patrol_dx;
        let new_y = enemy.y + enemy.patrol_dy;
        if is_walkable(grid, w, h, new_x, new_y) {
            enemy.x = new_x;
            enemy.y = new_y;
        } else {
            enemy.patrol_timer = 0;
        }
    }
}

pub fn enemy_attacks_player(player: &mut Player, enemies: &[Enemy], _dt: f32) {
    for enemy in enemies {
        if !enemy.alive {
            continue;
        }
        let dist = distance(enemy.x, enemy.y, player.x, player.y);
        if dist < enemy.attack_range && player.damage_cooldown <= 0.0 {
            player.hp -= enemy.damage;
            player.damage_cooldown = 0.5;
        }
    }
}
