{
  "ok": true,
  "request_id": "req_20260403_072846_3562_11",
  "data": {
    "phase": "run",
    "screen": "COMBAT",
    "in_run": true,
    "run_over": false,
    "in_combat": true,
    "turn": 1,
    "combat": {
      "player": {
        "current_hp": 80,
        "max_hp": 80,
        "block": 0,
        "energy": 3,
        "stars": 0,
        "focus": 0,
        "powers": [],
        "base_orb_slots": 0,
        "orb_capacity": 0,
        "empty_orb_slots": 0,
        "orbs": []
      },
      "hand": [
        {
          "index": 0,
          "card_id": "STRIKE_IRONCLAD",
          "name": "打击",
          "upgraded": false,
          "target_type": "AnyEnemy",
          "requires_target": true,
          "target_index_space": "enemies",
          "valid_target_indices": [
            0,
            1
          ],
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "造成6点伤害。",
          "key_words": [],
          "playable": true,
          "unplayable_reason": null
        },
        {
          "index": 1,
          "card_id": "DEFEND_IRONCLAD",
          "name": "防御",
          "upgraded": false,
          "target_type": "Self",
          "requires_target": false,
          "target_index_space": null,
          "valid_target_indices": [],
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "获得5点格挡。",
          "key_words": [
            "格挡"
          ],
          "playable": true,
          "unplayable_reason": null
        },
        {
          "index": 2,
          "card_id": "STRIKE_IRONCLAD",
          "name": "打击",
          "upgraded": false,
          "target_type": "AnyEnemy",
          "requires_target": true,
          "target_index_space": "enemies",
          "valid_target_indices": [
            0,
            1
          ],
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "造成6点伤害。",
          "key_words": [],
          "playable": true,
          "unplayable_reason": null
        },
        {
          "index": 3,
          "card_id": "STRIKE_IRONCLAD",
          "name": "打击",
          "upgraded": false,
          "target_type": "AnyEnemy",
          "requires_target": true,
          "target_index_space": "enemies",
          "valid_target_indices": [
            0,
            1
          ],
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "造成6点伤害。",
          "key_words": [],
          "playable": true,
          "unplayable_reason": null
        },
        {
          "index": 4,
          "card_id": "DEFEND_IRONCLAD",
          "name": "防御",
          "upgraded": false,
          "target_type": "Self",
          "requires_target": false,
          "target_index_space": null,
          "valid_target_indices": [],
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "获得5点格挡。",
          "key_words": [
            "格挡"
          ],
          "playable": true,
          "unplayable_reason": null
        }
      ],
      "enemies": [
        {
          "index": 0,
          "enemy_id": "TOADPOLE",
          "name": "蟾蜍蝌蚪",
          "current_hp": 23,
          "max_hp": 23,
          "block": 0,
          "is_alive": true,
          "is_hittable": true,
          "powers": [],
          "intent": "SPIKEN_MOVE",
          "move_id": "SPIKEN_MOVE",
          "intents": [
            {
              "index": 0,
              "intent_type": "Buff",
              "label": null,
              "damage": null,
              "hits": null,
              "total_damage": null,
              "status_card_count": null
            }
          ]
        },
        {
          "index": 1,
          "enemy_id": "TOADPOLE",
          "name": "蟾蜍蝌蚪",
          "current_hp": 22,
          "max_hp": 22,
          "block": 0,
          "is_alive": true,
          "is_hittable": true,
          "powers": [],
          "intent": "WHIRL_MOVE",
          "move_id": "WHIRL_MOVE",
          "intents": [
            {
              "index": 0,
              "intent_type": "Attack",
              "label": "7",
              "damage": 7,
              "hits": 1,
              "total_damage": 7,
              "status_card_count": null
            }
          ]
        }
      ],
      "potions": [
        {
          "index": 0,
          "potion_id": null,
          "name": null,
          "description": null,
          "rarity": null,
          "occupied": false,
          "usage": null,
          "target_type": null,
          "is_queued": false,
          "requires_target": false,
          "target_index_space": null,
          "valid_target_indices": [],
          "can_use": false,
          "can_discard": false
        },
        {
          "index": 1,
          "potion_id": null,
          "name": null,
          "description": null,
          "rarity": null,
          "occupied": false,
          "usage": null,
          "target_type": null,
          "is_queued": false,
          "requires_target": false,
          "target_index_space": null,
          "valid_target_indices": [],
          "can_use": false,
          "can_discard": false
        },
        {
          "index": 2,
          "potion_id": null,
          "name": null,
          "description": null,
          "rarity": null,
          "occupied": false,
          "usage": null,
          "target_type": null,
          "is_queued": false,
          "requires_target": false,
          "target_index_space": null,
          "valid_target_indices": [],
          "can_use": false,
          "can_discard": false
        }
      ],
      "enemy_type": "monster"
    },
    "run": {
      "character_id": "IRONCLAD",
      "character_name": "铁甲战士",
      "floor": 2,
      "current_hp": 80,
      "max_hp": 80,
      "gold": 99,
      "max_energy": 3,
      "base_orb_slots": 0,
      "deck": [
        {
          "index": 0,
          "card_id": "STRIKE_IRONCLAD",
          "name": "打击",
          "upgraded": false,
          "card_type": "Attack",
          "rarity": "Basic",
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "造成6点伤害。"
        },
        {
          "index": 1,
          "card_id": "STRIKE_IRONCLAD",
          "name": "打击",
          "upgraded": false,
          "card_type": "Attack",
          "rarity": "Basic",
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "造成6点伤害。"
        },
        {
          "index": 2,
          "card_id": "STRIKE_IRONCLAD",
          "name": "打击",
          "upgraded": false,
          "card_type": "Attack",
          "rarity": "Basic",
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "造成6点伤害。"
        },
        {
          "index": 3,
          "card_id": "STRIKE_IRONCLAD",
          "name": "打击",
          "upgraded": false,
          "card_type": "Attack",
          "rarity": "Basic",
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "造成6点伤害。"
        },
        {
          "index": 4,
          "card_id": "STRIKE_IRONCLAD",
          "name": "打击",
          "upgraded": false,
          "card_type": "Attack",
          "rarity": "Basic",
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "造成6点伤害。"
        },
        {
          "index": 5,
          "card_id": "DEFEND_IRONCLAD",
          "name": "防御",
          "upgraded": false,
          "card_type": "Skill",
          "rarity": "Basic",
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "获得5点格挡。"
        },
        {
          "index": 6,
          "card_id": "DEFEND_IRONCLAD",
          "name": "防御",
          "upgraded": false,
          "card_type": "Skill",
          "rarity": "Basic",
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "获得5点格挡。"
        },
        {
          "index": 7,
          "card_id": "DEFEND_IRONCLAD",
          "name": "防御",
          "upgraded": false,
          "card_type": "Skill",
          "rarity": "Basic",
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "获得5点格挡。"
        },
        {
          "index": 8,
          "card_id": "DEFEND_IRONCLAD",
          "name": "防御",
          "upgraded": false,
          "card_type": "Skill",
          "rarity": "Basic",
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 1,
          "star_cost": 0,
          "rules_text": "获得5点格挡。"
        },
        {
          "index": 9,
          "card_id": "BASH",
          "name": "痛击",
          "upgraded": false,
          "card_type": "Attack",
          "rarity": "Basic",
          "costs_x": false,
          "star_costs_x": false,
          "energy_cost": 2,
          "star_cost": 0,
          "rules_text": "造成8点伤害。 给予2层易伤。"
        }
      ],
      "relics": [
        {
          "index": 0,
          "relic_id": "BURNING_BLOOD",
          "name": "燃烧之血",
          "description": "在战斗结束时，回复6点生命。",
          "stack": null,
          "is_melted": false
        },
        {
          "index": 1,
          "relic_id": "BOOMING_CONCH",
          "name": "轰鸣海螺",
          "description": "在精英战的战斗开始时，额外抽2张牌。",
          "stack": null,
          "is_melted": false
        }
      ],
      "players": [
        {
          "player_id": "1",
          "slot_index": 0,
          "is_local": true,
          "is_connected": true,
          "character_id": "IRONCLAD",
          "character_name": "铁甲战士",
          "current_hp": 80,
          "max_hp": 80,
          "gold": 99,
          "is_alive": true
        }
      ],
      "potions": [
        {
          "index": 0,
          "potion_id": null,
          "name": null,
          "description": null,
          "rarity": null,
          "occupied": false,
          "usage": null,
          "target_type": null,
          "is_queued": false,
          "requires_target": false,
          "target_index_space": null,
          "valid_target_indices": [],
          "can_use": false,
          "can_discard": false
        },
        {
          "index": 1,
          "potion_id": null,
          "name": null,
          "description": null,
          "rarity": null,
          "occupied": false,
          "usage": null,
          "target_type": null,
          "is_queued": false,
          "requires_target": false,
          "target_index_space": null,
          "valid_target_indices": [],
          "can_use": false,
          "can_discard": false
        },
        {
          "index": 2,
          "potion_id": null,
          "name": null,
          "description": null,
          "rarity": null,
          "occupied": false,
          "usage": null,
          "target_type": null,
          "is_queued": false,
          "requires_target": false,
          "target_index_space": null,
          "valid_target_indices": [],
          "can_use": false,
          "can_discard": false
        }
      ]
    },
    "map": null,
    "selection": null,
    "event": null,
    "reward": null,
    "shop": null,
    "game_over": null,
    "can_act": true,
    "block_reason": null,
    "selected_character_id": null,
    "can_choose_character": false,
    "can_start_run": false,
    "legal_actions": [
      "end_turn",
      "play_card"
    ]
  }
}