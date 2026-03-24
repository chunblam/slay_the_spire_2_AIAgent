"""
src/env/state_encoder.py

将 STS2AIAgent 返回的 JSON 游戏状态编码为神经网络可用的张量
"""

from typing import Dict, List, Optional
import numpy as np
import gymnasium as gym


# ─── 常量定义 ─────────────────────────────────────────────────────────────────

MAX_HAND = 10          # 最大手牌数
MAX_MONSTERS = 5       # 最大敌人数
MAX_RELICS = 30        # 最大遗物数
MAX_DECK = 60          # 最大牌组大小（统计特征）
MAX_ENERGY = 10        # 最大能量

CARD_FEATURE_DIM = 8   # 每张卡的特征维度
MONSTER_FEATURE_DIM = 8  # 每个敌人的特征维度
PLAYER_FEATURE_DIM = 10  # 玩家状态特征维度
RELIC_FEATURE_DIM = 1  # 遗物用 one-hot 编码

# 归一化常数
MAX_HP = 100.0
MAX_GOLD = 1000.0
MAX_FLOOR = 55.0
MAX_BLOCK = 100.0
MAX_DAMAGE = 100.0


class StateEncoder:
    """
    将游戏 JSON 状态编码为固定维度的 numpy 数组字典

    设计原则:
    - 手牌、敌人等变长序列用固定长度+padding处理
    - 所有数值归一化到 [0, 1]
    - 布尔值编码为 0/1
    - 卡牌类型、稀有度等类别变量用 one-hot 或 embedding index
    """

    # 卡牌类型映射
    CARD_TYPE_MAP = {"ATTACK": 0, "SKILL": 1, "POWER": 2, "STATUS": 3, "CURSE": 4}
    CARD_COST_SPECIAL = {"X": -1, "UNPLAYABLE": -2}

    def get_observation_space(self) -> gym.spaces.Dict:
        """返回 Gymnasium 观测空间定义"""
        return gym.spaces.Dict({
            # 玩家状态: [hp_ratio, block_ratio, energy_ratio, gold_ratio,
            #            floor_ratio, num_relics, num_cards_in_deck,
            #            is_in_combat, buffs_count, debuffs_count]
            "player": gym.spaces.Box(
                low=0.0, high=1.0, shape=(PLAYER_FEATURE_DIM,), dtype=np.float32
            ),
            # 手牌矩阵: [MAX_HAND × CARD_FEATURE_DIM]
            # 每张卡: [cost_norm, damage_norm, block_norm, card_type_oh×5,
            #          is_ethereal, is_exhaust, is_innate]  (dim=8)
            "hand": gym.spaces.Box(
                low=0.0, high=1.0, shape=(MAX_HAND, CARD_FEATURE_DIM), dtype=np.float32
            ),
            # 手牌有效 mask: 1=有卡且可出, 0=空槽或无法出
            "hand_mask": gym.spaces.Box(
                low=0, high=1, shape=(MAX_HAND,), dtype=np.float32
            ),
            # 敌人矩阵: [MAX_MONSTERS × MONSTER_FEATURE_DIM]
            # 每个敌人: [hp_ratio, block_ratio, intent_damage_norm,
            #            intent_type_oh×4, alive]  (dim=8)
            "monsters": gym.spaces.Box(
                low=0.0, high=1.0, shape=(MAX_MONSTERS, MONSTER_FEATURE_DIM), dtype=np.float32
            ),
            # 遗物存在 mask: one-hot 向量，1=拥有该遗物
            "relics": gym.spaces.Box(
                low=0, high=1, shape=(MAX_RELICS,), dtype=np.float32
            ),
            # 牌组统计: [total, attacks, skills, powers, curses,
            #            status, upgrade_ratio, avg_cost]
            "deck_stats": gym.spaces.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32
            ),
            # 当前 screen type 编码
            "screen_type": gym.spaces.Discrete(16),
        })

    def encode(self, state: Dict) -> Dict:
        """将完整游戏状态 JSON 编码为观测字典"""
        screen_type = self._encode_screen_type(state.get("screen_type", "NONE"))
        combat = state.get("combat", {})
        player_data = combat.get("player", {})

        player_vec = self._encode_player(state, player_data, combat)
        hand_mat, hand_mask = self._encode_hand(combat.get("hand", []), player_data)
        monsters_mat = self._encode_monsters(combat.get("monsters", []))
        relics_vec = self._encode_relics(state.get("relics", []))
        deck_stats = self._encode_deck(state.get("deck", []))

        return {
            "player": player_vec.astype(np.float32),
            "hand": hand_mat.astype(np.float32),
            "hand_mask": hand_mask.astype(np.float32),
            "monsters": monsters_mat.astype(np.float32),
            "relics": relics_vec.astype(np.float32),
            "deck_stats": deck_stats.astype(np.float32),
            "screen_type": np.array(screen_type, dtype=np.int64),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 各子模块编码器
    # ──────────────────────────────────────────────────────────────────────────

    def _encode_player(self, state: Dict, player: Dict, combat: Dict) -> np.ndarray:
        hp = player.get("hp", 0)
        max_hp = max(player.get("max_hp", 1), 1)
        block = player.get("block", 0)
        energy = combat.get("energy", 0)
        gold = state.get("gold", 0)
        floor_ = state.get("floor", 0)
        relics = state.get("relics", [])
        deck = state.get("deck", [])
        buffs = player.get("buffs", [])
        debuffs = [b for b in buffs if b.get("amount", 0) < 0]
        pos_buffs = [b for b in buffs if b.get("amount", 0) >= 0]

        return np.array([
            hp / max_hp,                          # HP 比例
            min(block / MAX_BLOCK, 1.0),          # 格挡比例
            energy / MAX_ENERGY,                  # 能量比例
            min(gold / MAX_GOLD, 1.0),            # 金币比例
            floor_ / MAX_FLOOR,                   # 楼层比例
            min(len(relics) / MAX_RELICS, 1.0),   # 遗物数量比例
            min(len(deck) / MAX_DECK, 1.0),       # 牌组大小比例
            float(bool(combat)),                  # 是否在战斗中
            min(len(pos_buffs) / 10.0, 1.0),      # 正面 buff 数量
            min(len(debuffs) / 10.0, 1.0),        # 负面 debuff 数量
        ])

    def _encode_hand(self, hand: List[Dict], player: Dict) -> tuple:
        """编码手牌矩阵及可出牌 mask"""
        mat = np.zeros((MAX_HAND, CARD_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros(MAX_HAND, dtype=np.float32)
        energy = player.get("energy", 0) if player else 0  # 注意: energy 在 combat 层级

        for i, card in enumerate(hand[:MAX_HAND]):
            cost = card.get("cost", 0)
            cost_val = cost if isinstance(cost, int) else 0
            damage = card.get("damage", 0) or 0
            block = card.get("block", 0) or 0
            card_type = card.get("type", "ATTACK")

            # 基础特征
            type_idx = self.CARD_TYPE_MAP.get(card_type, 0)
            type_oh = np.zeros(5)
            type_oh[type_idx] = 1.0

            mat[i] = [
                cost_val / 3.0,                   # 费用归一化
                min(damage / MAX_DAMAGE, 1.0),    # 伤害归一化
                min(block / MAX_BLOCK, 1.0),      # 格挡归一化
                float(card.get("is_ethereal", False)),
                float(card.get("exhaust", False)),
                float(card.get("innate", False)),
                0.0,  # 保留维度
                0.0,  # 保留维度
            ]

            # 是否可出: 费用 <= 当前能量 或 特殊费用
            playable = cost_val <= energy or cost == "X"
            mask[i] = float(playable)

        return mat, mask

    def _encode_monsters(self, monsters: List[Dict]) -> np.ndarray:
        """编码敌人矩阵"""
        mat = np.zeros((MAX_MONSTERS, MONSTER_FEATURE_DIM), dtype=np.float32)

        alive_monsters = [m for m in monsters if m.get("hp", 0) > 0]
        for i, monster in enumerate(alive_monsters[:MAX_MONSTERS]):
            hp = monster.get("hp", 0)
            max_hp = max(monster.get("max_hp", 1), 1)
            block = monster.get("block", 0)
            intent = monster.get("intent", {})
            intent_type = intent.get("type", "NONE")
            intent_dmg = intent.get("damage", 0) or 0

            # Intent type 简单编码
            intent_map = {"ATTACK": 0, "DEFEND": 1, "BUFF": 2, "DEBUFF": 3,
                          "SLEEP": 4, "ESCAPE": 5, "NONE": 6, "UNKNOWN": 7}
            intent_idx = intent_map.get(intent_type, 7)
            intent_oh = np.zeros(4)
            intent_oh[min(intent_idx, 3)] = 1.0

            mat[i] = [
                hp / max_hp,                         # HP 比例
                min(block / MAX_BLOCK, 1.0),          # 格挡比例
                min(intent_dmg / MAX_DAMAGE, 1.0),    # 意图伤害
                1.0,                                  # alive
                intent_oh[0],
                intent_oh[1],
                intent_oh[2],
                intent_oh[3],
            ]

        return mat

    def _encode_relics(self, relics: List[Dict]) -> np.ndarray:
        """遗物 one-hot 编码（基于哈希，固定维度）"""
        vec = np.zeros(MAX_RELICS, dtype=np.float32)
        for relic in relics[:MAX_RELICS]:
            name = relic.get("id", relic.get("name", ""))
            # 简单哈希到槽位
            slot = hash(name) % MAX_RELICS
            vec[slot] = 1.0
        return vec

    def _encode_deck(self, deck: List[Dict]) -> np.ndarray:
        """牌组统计特征"""
        if not deck:
            return np.zeros(8, dtype=np.float32)

        total = len(deck)
        attacks = sum(1 for c in deck if c.get("type") == "ATTACK")
        skills = sum(1 for c in deck if c.get("type") == "SKILL")
        powers = sum(1 for c in deck if c.get("type") == "POWER")
        curses = sum(1 for c in deck if c.get("type") in ("CURSE", "STATUS"))
        status = sum(1 for c in deck if c.get("type") == "STATUS")
        upgraded = sum(1 for c in deck if c.get("upgraded", False))

        costs = [c.get("cost", 0) for c in deck if isinstance(c.get("cost"), int)]
        avg_cost = np.mean(costs) / 3.0 if costs else 0.0

        return np.array([
            min(total / MAX_DECK, 1.0),
            attacks / max(total, 1),
            skills / max(total, 1),
            powers / max(total, 1),
            curses / max(total, 1),
            status / max(total, 1),
            upgraded / max(total, 1),
            avg_cost,
        ], dtype=np.float32)

    def _encode_screen_type(self, screen_type: str) -> int:
        screen_map = {
            "NONE": 0, "COMBAT": 1, "MAP": 2, "CARD_REWARD": 3,
            "REST": 4, "SHOP": 5, "EVENT": 6, "CHEST": 7,
            "CARD_SELECT": 8, "GRID": 9, "BOSS_REWARD": 10,
            "COMPLETE": 11, "GAME_OVER": 12, "DEATH": 13,
            "LOADING": 14, "OTHER": 15,
        }
        return screen_map.get(screen_type, 15)