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

    @staticmethod
    def _norm_token(value: Optional[str]) -> str:
        if value is None:
            return ""
        s = str(value).strip()
        if not s:
            return ""
        return s.replace("-", "_").replace(" ", "_").upper()

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
        hand_mat, hand_mask = self._encode_hand(combat.get("hand", []), player_data, combat)
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
            float(bool(state.get("in_combat", False))),  # 是否在战斗中
            min(len(pos_buffs) / 10.0, 1.0),      # 正面 buff 数量
            min(len(debuffs) / 10.0, 1.0),        # 负面 debuff 数量
        ])

    def _encode_hand(self, hand: List[Dict], player: Dict, combat: Optional[Dict] = None) -> tuple:
        """编码手牌矩阵及可出牌 mask。

        注意：`hand_mask` 是模型输入特征，不是动作采样掩码。
        真实动作掩码由 action_space.get_valid_action_mask() 构建。
        """
        mat = np.zeros((MAX_HAND, CARD_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros(MAX_HAND, dtype=np.float32)
        combat = combat or {}
        energy = 0
        if isinstance(combat, dict):
            energy = int(combat.get("energy", 0) or 0)
        if energy <= 0 and player:
            energy = int(player.get("energy", 0) or 0)

        for i, card in enumerate(hand[:MAX_HAND]):
            if not isinstance(card, dict):
                continue
            
            # API 标准字段: energy_cost（不再用 cost 别名）
            cost_val = int(card.get("energy_cost", 0) or 0)
            costs_x = bool(card.get("costs_x", False))
            damage = int(card.get("damage", 0) or 0)
            block = int(card.get("block", 0) or 0)

            # 手牌中通常没有 card_type，但如果有就用；encoder 不负责补充 card_type
            mat[i] = [
                cost_val / 3.0,                      # 费用归一化 (energy_cost / 3)
                min(damage / MAX_DAMAGE, 1.0),       # 伤害归一化
                min(block / MAX_BLOCK, 1.0),         # 格挡归一化
                float(card.get("upgraded", False)),  # 是否已升级
                float(costs_x),                      # 是否为X费
                float(card.get("star_costs_x", False)),  # 是否为星星X费
                float(card.get("requires_target", False)),  # 是否需要目标
                float(card.get("playable", False)),  # 是否可打出
            ]

            # 优先使用 API 的 playable 字段；不存在时才回退使用能量规则
            playable_raw = card.get("playable")
            if playable_raw is None:
                playable = (cost_val <= energy) or costs_x
            else:
                playable = bool(playable_raw)
            mask[i] = float(playable)

        return mat, mask

    def _encode_monsters(self, monsters: List[Dict]) -> np.ndarray:
        """编码敌人矩阵。API 使用 intents[] 中的 intent_type, hits, damage, total_damage 等字段。"""
        mat = np.zeros((MAX_MONSTERS, MONSTER_FEATURE_DIM), dtype=np.float32)

        # API 字段: is_alive（不再推导）
        alive_monsters = [m for m in monsters if m.get("is_alive", False)]
        for i, monster in enumerate(alive_monsters[:MAX_MONSTERS]):
            # API 字段: current_hp（不是 hp）
            current_hp = int(monster.get("current_hp", 0) or 0)
            max_hp = max(int(monster.get("max_hp", 1) or 1), 1)
            block = int(monster.get("block", 0) or 0)
            
            # 从 intents[] 提取主意图
            intents = monster.get("intents")
            if not isinstance(intents, list):
                intents = []
            primary_intent = intents[0] if intents and isinstance(intents[0], dict) else {}

            # API 字段: intent_type（不是 type）、hits（不是 times）
            intent_type = primary_intent.get("intent_type", "UNKNOWN")
            intent_type_norm = self._norm_token(intent_type)
            
            total_damage = primary_intent.get("total_damage")
            if total_damage is not None:
                intent_dmg = float(total_damage or 0)
            else:
                hits = int(primary_intent.get("hits", 1) or 1)
                damage = float(primary_intent.get("damage", 0) or 0)
                intent_dmg = damage * hits

            # Intent type 映射
            intent_map = {
                "ATTACK": 0,
                "ATTACK_BUFF": 0,
                "ATTACK_DEBUFF": 0,
                "DEFEND": 1,
                "BUFF": 2,
                "DEBUFF": 3,
                "SLEEP": 4,
                "ESCAPE": 5,
                "STATUS_CARD": 6,
                "UNKNOWN": 7,
            }
            intent_idx = intent_map.get(intent_type_norm, 7)
            intent_oh = np.zeros(4)
            intent_oh[min(intent_idx, 3)] = 1.0

            mat[i] = [
                current_hp / max_hp,                 # HP 比例
                min(block / MAX_BLOCK, 1.0),         # 格挡比例
                min(intent_dmg / MAX_DAMAGE, 1.0),   # 意图伤害
                1.0,                                 # 是否存活
                intent_oh[0],
                intent_oh[1],
                intent_oh[2],
                intent_oh[3],
            ]

        return mat

    def _encode_relics(self, relics: List[Dict]) -> np.ndarray:
        """遗物 one-hot 编码（基于 relic_id 哈希）。API 字段: relic_id, name, description, stack, is_melted。"""
        vec = np.zeros(MAX_RELICS, dtype=np.float32)
        for relic in relics[:MAX_RELICS]:
            if not isinstance(relic, dict):
                continue
            # API 字段: relic_id（优先）或 name（备选）
            relic_id = relic.get("relic_id") or relic.get("name", "")
            if not relic_id:
                continue
            # 基于 relic_id 的哈希编码到固定槽位
            slot = hash(relic_id) % MAX_RELICS
            vec[slot] = 1.0
        return vec

    def _encode_deck(self, deck: List[Dict]) -> np.ndarray:
        """牌组统计特征。API 字段: card_type（不是 type）、energy_cost（不是 cost）、rarity、star_costs_x 等。"""
        if not deck:
            return np.zeros(8, dtype=np.float32)

        total = len(deck)
        
        # API 字段：card_type（这是 deck 中的标准字段）
        norm_types = []
        for c in deck:
            if not isinstance(c, dict):
                continue
            card_type = c.get("card_type", "")
            if card_type:
                norm_types.append(self._norm_token(card_type))
        
        attacks = sum(1 for t in norm_types if t == "ATTACK")
        skills = sum(1 for t in norm_types if t == "SKILL")
        powers = sum(1 for t in norm_types if t == "POWER")
        curses = sum(1 for t in norm_types if t in ("CURSE", "STATUS"))
        status = sum(1 for t in norm_types if t == "STATUS")
        upgraded = sum(1 for c in deck if isinstance(c, dict) and c.get("upgraded", False))

        # API 字段: energy_cost（不再用 cost 别名）
        costs = []
        for c in deck:
            if not isinstance(c, dict):
                continue
            cost = c.get("energy_cost", 0)
            if isinstance(cost, (int, float)):
                costs.append(int(cost))
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
        st = self._norm_token(screen_type)
        screen_map = {
            "NONE": 0, "COMBAT": 1, "MAP": 2, "CARD_REWARD": 3,
            "REST": 4, "SHOP": 5, "EVENT": 6, "CHEST": 7,
            "CARD_SELECT": 8, "GRID": 9, "BOSS_REWARD": 10,
            "COMPLETE": 11, "GAME_OVER": 12, "DEATH": 13,
            "LOADING": 14, "OTHER": 15,
            # STS2AIAgent screen names / env-normalized variants
            "REWARD": 3,
            "CARD_SELECTION": 8,
            "CHOOSE_CARD_BUNDLE": 8,
            "CARDS_VIEW": 9,
            "CHARACTER_SELECT": 15,
            "MAIN_MENU": 0,
            "MODAL": 15,
            "UNKNOWN": 15,
            "MULTIPLAYER_LOBBY": 15,
        }
        return screen_map.get(st, 15)