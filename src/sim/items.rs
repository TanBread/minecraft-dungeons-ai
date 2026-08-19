use super::entities::{Player, Item, ItemKind};
use super::physics::distance;

pub fn collect_items(player: &mut Player, items: &mut [Item]) {
    for item in items.iter_mut() {
        if item.collected {
            continue;
        }
        let dist = distance(item.x, item.y, player.x, player.y);
        if dist < 0.8 {
            item.collected = true;
            if item.kind == ItemKind::Health {
                player.hp = player.hp.min(player.max_hp + 30.0);
            }
        }
    }
}
