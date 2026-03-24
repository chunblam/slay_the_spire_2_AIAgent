"""
src/env/action_space.py

将 RL 离散动作 ID 映射为 STS2AIAgent Session API 的动作请求。
规范见 docs/STS2AIAgent-API-中文调用文档.md
"""
from typing import Dict, List, Optional, Set


def _first_alive_target_index(monsters: List[dict]) -> Optional[int]:
    for i, m in enumerate(monsters or []):
        if m.get("hp", 0) > 0 and m.get("is_alive", True):
            return i
    return None


class STS2ActionSpace:
    """
    动作空间设计（按 screen_type 分组，与 sts2_env 归一化后的语义一致）

    战斗中 (COMBAT):
        0 ~ max_hand-1        → play_card（card_index + 可选 target_index）
        max_hand ~ +potions-1 → use_potion（option_index + 可选 target_index）
        last                  → end_turn

    奖励选择 (CARD_REWARD):
        0=跳过，1..n → choose_reward_card(option_index 0-based)

    地图导航 (MAP):
        0 ~ 6                 → choose_map_node（index 对应 next_options）

    休息 (REST):
        0~3                   → choose_rest_option（index）

    商店 (SHOP):
        先尝试 open_shop_inventory，再按候选购买动作执行

    事件 (EVENT):
        0~4                   → choose_event_option
    卡牌选择 (CARD_SELECT):
        0~9                   → select_deck_card(option_index)
    """

    def __init__(
        self,
        max_hand_size: int = 10,
        max_potions: int = 5,
    ):
        self.max_hand = max_hand_size
        self.max_potions = max_potions
        self.total_actions = max(16, 10)
        self._shop_done_floors: Set[int] = set()

    def decode(self, action_id: int, state: Dict) -> Dict:
        """
        返回可直接 POST 的 JSON 对象（含 \"action\" 字段）。
        """
        screen = state.get("screen_type", "NONE")

        if screen == "COMBAT":
            candidate = self._decode_combat(action_id, state)
        elif screen == "CARD_REWARD":
            candidate = self._decode_card_reward(action_id, state)
        elif screen == "REWARD":
            candidate = self._decode_reward(action_id, state)
        elif screen == "MAP":
            candidate = self._decode_map(action_id, state)
        elif screen == "REST":
            candidate = self._decode_rest(action_id, state)
        elif screen == "CARD_SELECT":
            candidate = self._decode_card_select(action_id, state)
        elif screen == "SHOP":
            candidate = self._decode_shop(action_id, state)
        elif screen == "EVENT":
            candidate = self._decode_event(action_id, state)
        elif screen == "CHEST":
            candidate = self._decode_chest(action_id, state)
        else:
            candidate = self._fallback_from_legal_actions(state)
        return self._ensure_legal(candidate, state)

    def _ensure_legal(self, candidate: Dict, state: Dict) -> Dict:
        legal = [str(a) for a in (state.get("legal_actions") or [])]
        action_name = str(candidate.get("action", ""))
        if not legal or not action_name:
            return candidate
        if action_name in legal:
            return candidate
        return self._fallback_from_legal_actions(state)

    @staticmethod
    def _fallback_from_legal_actions(state: Dict) -> Dict:
        legal = [str(a) for a in (state.get("legal_actions") or [])]
        if not legal:
            return {"action": "end_turn"}

        if "close_shop_inventory" in legal:
            return {"action": "close_shop_inventory"}
        if "menu_back" in legal:
            return {"action": "menu_back"}
        if "collect_rewards_and_proceed" in legal:
            return {"action": "collect_rewards_and_proceed"}
        if "claim_reward" in legal:
            return {"action": "claim_reward", "option_index": 0}
        if "choose_reward_card" in legal:
            return {"action": "choose_reward_card", "option_index": 0}
        if "skip_reward_cards" in legal:
            return {"action": "skip_reward_cards"}
        if "choose_map_node" in legal:
            return {"action": "choose_map_node", "option_index": 0}
        if "choose_event_option" in legal:
            return {"action": "choose_event_option", "option_index": 0}
        if "choose_rest_option" in legal:
            return {"action": "choose_rest_option", "option_index": 0}
        if "open_chest" in legal:
            return {"action": "open_chest"}
        if "choose_treasure_relic" in legal:
            return {"action": "choose_treasure_relic", "option_index": 0}
        if "open_shop_inventory" in legal:
            return {"action": "open_shop_inventory"}
        if "proceed" in legal:
            return {"action": "proceed"}
        if "end_turn" in legal:
            return {"action": "end_turn"}
        return {"action": legal[0]}

    def _decode_combat(self, action_id: int, state: Dict) -> Dict:
        combat = state.get("combat", {})
        hand = combat.get("hand", [])
        potions = state.get("potions", [])
        monsters = combat.get("monsters", [])
        target_index = _first_alive_target_index(monsters)

        if action_id < self.max_hand:
            if action_id < len(hand):
                body: Dict = {"action": "play_card", "card_index": action_id}
                if target_index is not None and bool(hand[action_id].get("requires_target", False)):
                    body["target_index"] = target_index
                return body
            return {"action": "end_turn"}

        if action_id < self.max_hand + self.max_potions:
            option_index = action_id - self.max_hand
            if option_index < len(potions):
                body = {"action": "use_potion", "option_index": option_index}
                if target_index is not None:
                    body["target_index"] = target_index
                return body
            return {"action": "end_turn"}

        return {"action": "end_turn"}

    def _decode_card_reward(self, action_id: int, state: Dict) -> Dict:
        cards = state.get("card_reward", {}).get("cards", [])
        if not cards or action_id == 0 or action_id > len(cards):
            return {"action": "skip_reward_cards"}
        return {"action": "choose_reward_card", "option_index": action_id - 1}

    def _decode_reward(self, action_id: int, state: Dict) -> Dict:
        reward = state.get("reward") or {}
        rewards = reward.get("rewards") or []
        if rewards:
            idx = min(max(action_id, 0), len(rewards) - 1)
            return {"action": "claim_reward", "option_index": idx}
        return self._fallback_from_legal_actions(state)

    def _map_next_options(self, state: Dict) -> List:
        m = state.get("map") or {}
        opts = m.get("next_options")
        if opts is not None:
            return list(opts)
        nodes = m.get("next_nodes")
        return list(nodes) if nodes else []

    def _decode_map(self, action_id: int, state: Dict) -> Dict:
        nodes = self._map_next_options(state)
        if not nodes:
            return self._fallback_from_legal_actions(state)
        idx = min(action_id, len(nodes) - 1)
        return {"action": "choose_map_node", "option_index": idx}

    def _decode_rest(self, action_id: int, state: Dict) -> Dict:
        idx = min(max(action_id, 0), 3)
        return {"action": "choose_rest_option", "option_index": idx}

    def _decode_card_select(self, action_id: int, state: Dict) -> Dict:
        cards = (state.get("selection") or {}).get("cards", [])
        idx = min(action_id, len(cards) - 1) if cards else 0
        return {"action": "select_deck_card", "option_index": idx}

    def _decode_shop(self, action_id: int, state: Dict) -> Dict:
        shop = state.get("shop") or {}
        legal = [str(a) for a in (state.get("legal_actions") or [])]
        floor = int(state.get("floor", 0) or 0)

        is_open = bool(shop.get("is_open", shop.get("open", False)))
        can_open = bool(shop.get("can_open", True))
        can_close = bool(shop.get("can_close", False))

        # 纯 flag 机制：同一楼层只允许进入商店一次，避免反复开关商店。
        if is_open:
            self._shop_done_floors.add(floor)
        if floor in self._shop_done_floors and not is_open:
            if "menu_back" in legal:
                return {"action": "menu_back"}
            if "proceed" in legal:
                return {"action": "proceed"}
            return self._fallback_from_legal_actions(state)

        if (not is_open) and can_open and ("open_shop_inventory" in legal or not legal):
            return {"action": "open_shop_inventory"}

        def _is_buyable(x: Dict) -> bool:
            affordable = bool(x.get("affordable", x.get("enough_gold", True)))
            stocked = bool(x.get("stocked", x.get("available", x.get("is_stocked", True))))
            return affordable and stocked

        def _idx(x: Dict, fallback: int) -> int:
            if x.get("index") is not None:
                return int(x.get("index"))
            if x.get("i") is not None:
                return int(x.get("i"))
            return fallback

        cards_raw = list(shop.get("cards") or [])
        relics_raw = list(shop.get("relics") or [])
        potions_raw = list(shop.get("potions") or [])

        cards = [x for x in cards_raw if _is_buyable(x)]
        relics = [x for x in relics_raw if _is_buyable(x)]
        potions = [x for x in potions_raw if _is_buyable(x)]
        candidates: List[Dict] = []
        if "buy_card" in legal or not legal:
            candidates.extend([{"action": "buy_card", "option_index": _idx(x, i)} for i, x in enumerate(cards)])
        if "buy_relic" in legal or not legal:
            candidates.extend([{"action": "buy_relic", "option_index": _idx(x, i)} for i, x in enumerate(relics)])
        if "buy_potion" in legal or not legal:
            candidates.extend([{"action": "buy_potion", "option_index": _idx(x, i)} for i, x in enumerate(potions)])
        removal = shop.get("card_removal") or shop.get("remove") or {}
        if ("remove_card_at_shop" in legal or not legal) and bool(removal.get("available", False)) and bool(removal.get("affordable", True)):
            candidates.append({"action": "remove_card_at_shop"})
        if candidates:
            return candidates[action_id % len(candidates)]

        # 无可买项，标记当前楼层“商店已处理”，后续优先离开不再反复开关商店。
        self._shop_done_floors.add(floor)
        if can_close and ("close_shop_inventory" in legal or not legal):
            return {"action": "close_shop_inventory"}
        if "menu_back" in legal:
            return {"action": "menu_back"}
        return self._fallback_from_legal_actions(state)

    def _decode_event(self, action_id: int, state: Dict) -> Dict:
        event = state.get("event") or {}
        options = event.get("options") or []
        if not options:
            return self._fallback_from_legal_actions(state)
        idx = min(max(action_id, 0), len(options) - 1)
        return {"action": "choose_event_option", "option_index": idx}

    def _decode_chest(self, action_id: int, state: Dict) -> Dict:
        chest = state.get("chest") or {}
        if not bool(chest.get("is_opened", False)):
            return {"action": "open_chest"}
        relics = chest.get("relic_options") or []
        if relics:
            idx = min(max(action_id, 0), len(relics) - 1)
            return {"action": "choose_treasure_relic", "option_index": idx}
        return self._fallback_from_legal_actions(state)

    def get_valid_action_mask(self, state: Dict) -> List[bool]:
        """
        返回当前状态下每个动作是否有效的 mask
        """
        mask = [False] * self.total_actions
        legal_actions = state.get("legal_actions") or []
        if isinstance(legal_actions, list) and legal_actions:
            if "end_turn" in legal_actions:
                mask[self.max_hand + self.max_potions] = True

            if "play_card" in legal_actions:
                hand = (state.get("combat") or {}).get("hand", [])
                for i in range(min(len(hand), self.max_hand, self.total_actions)):
                    mask[i] = True

            if "use_potion" in legal_actions:
                for i in range(min(self.max_potions, self.total_actions - self.max_hand)):
                    mask[self.max_hand + i] = True

            if any(a in legal_actions for a in ("choose_map_node", "choose_rest_option", "choose_event_option",
                                                "select_deck_card", "choose_reward_card", "choose_treasure_relic",
                                                "claim_reward", "skip_reward_cards", "proceed", "confirm_selection",
                                                "open_chest", "open_shop_inventory", "buy_card", "buy_relic",
                                                "buy_potion", "remove_card_at_shop", "collect_rewards_and_proceed",
                                                "menu_new_run", "menu_continue", "menu_choose_character",
                                                "menu_confirm", "menu_back", "menu_return")):
                for i in range(min(8, self.total_actions)):
                    if not mask[i]:
                        mask[i] = True

            if any(mask):
                return mask

        screen = state.get("screen_type", "NONE")
        combat = state.get("combat", {})
        energy = combat.get("energy", 0)
        hand = combat.get("hand", [])

        if screen == "COMBAT":
            for i, card in enumerate(hand[: self.max_hand]):
                cost = card.get("cost", 0)
                cost_val = cost if isinstance(cost, int) else 0
                if bool(card.get("playable", True)) and (cost_val <= energy or cost == "X"):
                    mask[i] = True
            potions = state.get("potions", [])
            for i in range(min(len(potions), self.max_potions)):
                mask[self.max_hand + i] = True
            mask[self.max_hand + self.max_potions] = True

        elif screen in ("CARD_REWARD", "REWARD", "MAP", "REST", "SHOP", "CARD_SELECT", "CHEST"):
            for i in range(min(8, self.total_actions)):
                mask[i] = True
        elif screen == "EVENT":
            mask[0] = True

        return mask
