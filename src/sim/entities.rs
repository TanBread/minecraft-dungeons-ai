#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Tile {
    Wall = 0,
    Floor = 1,
    Door = 2,
    Exit = 3,
}

impl Tile {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Tile::Wall,
            1 => Tile::Floor,
            2 => Tile::Door,
            3 => Tile::Exit,
            _ => Tile::Wall,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Player {
    pub x: f32,
    pub y: f32,
    pub hp: f32,
    pub max_hp: f32,
    pub attack_cooldown: f32,
    pub damage_cooldown: f32,
    pub potion_cooldown: f32,
    pub dodge_cooldown: f32,
    pub artifact_cooldown: f32,
}

impl Player {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            hp: 100.0,
            max_hp: 100.0,
            attack_cooldown: 0.0,
            damage_cooldown: 0.0,
            potion_cooldown: 0.0,
            dodge_cooldown: 0.0,
            artifact_cooldown: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Enemy {
    pub x: f32,
    pub y: f32,
    pub hp: f32,
    pub max_hp: f32,
    pub alive: bool,
    pub speed: f32,
    pub damage: f32,
    pub attack_range: f32,
    pub detection_range: f32,
    pub attack_cooldown: f32,
    pub patrol_dx: f32,
    pub patrol_dy: f32,
    pub patrol_timer: i32,
}

impl Enemy {
    pub fn new(x: f32, y: f32, hp: f32, speed: f32, damage: f32) -> Self {
        Self {
            x,
            y,
            hp,
            max_hp: hp,
            alive: true,
            speed,
            damage,
            attack_range: 1.5,
            detection_range: 8.0,
            attack_cooldown: 0.0,
            patrol_dx: 0.0,
            patrol_dy: 0.0,
            patrol_timer: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    Health,
    Ammo,
    Emerald,
}

#[derive(Debug, Clone)]
pub struct Item {
    pub x: f32,
    pub y: f32,
    pub kind: ItemKind,
    pub collected: bool,
}

impl Item {
    pub fn new(x: f32, y: f32, kind: ItemKind) -> Self {
        Self { x, y, kind, collected: false }
    }
}
