from typing import Dict, List, Optional


def _get_legal_actions(state: Dict) -> List[str]:
    # Session API contract: action selection is driven by legal_actions.
    legal = state.get("legal_actions") or []
    return [str(a) for a in legal]


def _can_act_now(state: Dict) -> bool:
    return bool(state.get("can_act", True))


def _first_alive_target_index(monsters: List[dict]) -> Optional[int]:
    """STS2AIAgent requires target_index as enemy array index (int)。
    API 字段: is_alive（不再推导），current_hp（不是 hp）。
    """
    for idx, m in enumerate(monsters or []):
        if not isinstance(m, dict):
            continue
        # 使用 API 标准字段 is_alive（不再从 current_hp 推导）
        alive = bool(m.get("is_alive", False))
        if alive:
            return idx
    return None


def _pick_target_index_for_card(card: Dict, monsters: List[dict]) -> Optional[int]:
    if not bool(card.get("requires_target", False)):
        return None

    valid = card.get("valid_target_indices")
    if isinstance(valid, list):
        for v in valid:
            try:
                return int(v)
            except Exception:
                continue

    return _first_alive_target_index(monsters)


def _pick_target_index_for_potion(potion: Dict, monsters: List[dict]) -> Optional[int]:
    if not bool(potion.get("requires_target", False)):
        return None

    valid = potion.get("valid_target_indices")
    if isinstance(valid, list):
        for v in valid:
            try:
                return int(v)
            except Exception:
                continue

    return _first_alive_target_index(monsters)


class STS2ActionSpace:
    def __init__(
        self,
        max_hand_size: int = 10,
        max_potions: int = 5,
    ):
        self.max_hand = max_hand_size
        self.max_potions = max_potions
        # 10 hand slots + 5 potion slots + 1 end-turn slot
        self.total_actions = self.max_hand + self.max_potions + 1
        self._card_select_cursor = 0
        self._last_screen = ""

    def decode(self, action_id: int, state: Dict) -> Dict:
        action_id = max(0, int(action_id))
        screen = str(state.get("screen", "") or "").upper()
        if screen != "CARD_SELECTION" and self._last_screen == "CARD_SELECTION":
            self._card_select_cursor = 0
        self._last_screen = screen
        legal = _get_legal_actions(state)
        if not _can_act_now(state):
            # Env should gate posting when can_act=False; keep decode conservative.
            return self._fallback_from_legal_actions(state)

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
        elif screen == "CARD_SELECTION":
            candidate = self._decode_card_select(action_id, state)
        elif screen == "SHOP":
            candidate = self._decode_shop(action_id, state)
        elif screen == "EVENT":
            candidate = self._decode_event(action_id, state)
        elif screen == "CRYSTAL":
            legal = _get_legal_actions(state)
            if "crystal_auto_play" in legal:
                candidate = {"action": "crystal_auto_play"}
            else:
                candidate = self._fallback_from_legal_actions(state)
        elif screen == "CARD_BUNDLE":
            candidate = self._decode_choose_card_bundle(action_id, state)
        elif screen == "CHEST":
            candidate = self._decode_chest(action_id, state)
        else:
            candidate = self._fallback_from_legal_actions(state)
        return self._ensure_legal(candidate, state)

    def _ensure_legal(self, candidate: Dict, state: Dict) -> Dict:
        legal = _get_legal_actions(state)
        action_name = str(candidate.get("action", ""))
        if not legal or not action_name:
            return candidate
        if action_name in legal:
            return candidate
        return self._fallback_from_legal_actions(state)

    @staticmethod
    def _fallback_from_legal_actions(state: Dict) -> Dict:
        legal = _get_legal_actions(state)
        if not legal:
            return {"action": "proceed"}

        # Menu bootstrap
        if "open_character_select" in legal:
            return {"action": "open_character_select"}
        if "select_character" in legal:
            return {"action": "select_character", "option_index": 0}
        if "embark" in legal:
            return {"action": "embark"}
        if "return_to_main_menu" in legal:
            return {"action": "return_to_main_menu"}

        # Core in-run actions: prioritize non-terminal actions before end_turn.
        if "play_card" in legal:
            return {"action": "play_card", "card_index": 0}
        if "use_potion" in legal:
            return {"action": "use_potion", "option_index": 0}
        if "end_turn" in legal:
            return {"action": "end_turn"}

        if "choose_reward_card" in legal:
            return {"action": "choose_reward_card", "option_index": 0}
        if "skip_reward_cards" in legal:
            return {"action": "skip_reward_cards"}
        if "collect_rewards_and_proceed" in legal:
            return {"action": "collect_rewards_and_proceed"}

        if "claim_reward" in legal:
            return {"action": "claim_reward", "option_index": 0}
        if "choose_map_node" in legal:
            return {"action": "choose_map_node", "option_index": 0}
        if "choose_event_option" in legal:
            return {"action": "choose_event_option", "option_index": 0}
        if "choose_rest_option" in legal:
            return {"action": "choose_rest_option", "option_index": 0}

        if "buy_card" in legal:
            return {"action": "buy_card", "option_index": 0}
        if "buy_relic" in legal:
            return {"action": "buy_relic", "option_index": 0}
        if "buy_potion" in legal:
            return {"action": "buy_potion", "option_index": 0}
        if "remove_card_at_shop" in legal:
            return {"action": "remove_card_at_shop"}
        if "open_shop_inventory" in legal:
            return {"action": "open_shop_inventory"}
        if "close_shop_inventory" in legal:
            return {"action": "close_shop_inventory"}

        if "confirm_selection" in legal:
            selection = state.get("selection") or {}
            min_select = int(selection.get("min_select", 1) or 1)
            selected_count = int(selection.get("selected_count", 0) or 0)
            can_confirm = bool(selection.get("can_confirm", False))
            if can_confirm or selected_count >= min_select:
                return {"action": "confirm_selection"}
        if "select_deck_card" in legal:
            selection = state.get("selection") or {}
            cards = selection.get("cards") if isinstance(selection.get("cards"), list) else []
            selected_count = int(selection.get("selected_count", 0) or 0)
            if cards:
                idx = selected_count % len(cards)
                return {"action": "select_deck_card", "option_index": idx}
            return {"action": "select_deck_card", "option_index": 0}
        if "close_cards_view" in legal:
            return {"action": "close_cards_view"}

        if "open_chest" in legal:
            return {"action": "open_chest"}
        if "choose_treasure_relic" in legal:
            return {"action": "choose_treasure_relic", "option_index": 0}

        if "choose_bundle" in legal:
            return {"action": "choose_bundle", "option_index": 0}

        if "proceed" in legal:
            return {"action": "proceed"}

        if "continue_run" in legal:
            return {"action": "continue_run"}
        if "close_main_menu_submenu" in legal:
            return {"action": "close_main_menu_submenu"}
        if "confirm_modal" in legal:
            return {"action": "confirm_modal"}
        if "dismiss_modal" in legal:
            return {"action": "dismiss_modal"}

        for action_name in legal:
            if action_name != "discard_potion":
                return {"action": action_name}
        return {"action": "proceed"}

    def _decode_choose_card_bundle(self, action_id: int, state: Dict) -> Dict:
        legal = _get_legal_actions(state)
        card_bundle = state.get("card_bundle") or {}

        if "choose_bundle" in legal:
            controls = card_bundle.get("ui_controls") or []
            picks: List[dict] = []
            if isinstance(controls, list):
                for c in controls:
                    if isinstance(c, dict) and c.get("role") == "choose_bundle":
                        picks.append(c)
            pick_count = len(picks)
            if pick_count <= 1:
                idx = 0
            else:
                idx = int(action_id % 2)
                if idx >= pick_count:
                    idx = 0
            return {"action": "choose_bundle", "option_index": idx}

        if "proceed" in legal:
            return {"action": "proceed"}

        return self._fallback_from_legal_actions(state)

    def _decode_combat(self, action_id: int, state: Dict) -> Dict:
        combat = state.get("combat", {})
        hand = combat.get("hand", [])
        potions = state.get("potions", [])
        enemies = combat.get("enemies", combat.get("monsters", []))
        legal = _get_legal_actions(state)

        if action_id < self.max_hand:
            if action_id < len(hand):
                card = hand[action_id] if isinstance(hand[action_id], dict) else {}
                card_index = int(card.get("index", action_id) or action_id)
                body: Dict = {"action": "play_card", "card_index": card_index}
                # Only attach target_index when the selected card requires a target.
                target_index = _pick_target_index_for_card(card, enemies)
                if target_index is not None:
                    body["target_index"] = target_index
                return body
            return self._fallback_from_legal_actions(state)

        if action_id < self.max_hand + self.max_potions:
            slot = action_id - self.max_hand
            if slot < len(potions):
                potion = potions[slot] if isinstance(potions[slot], dict) else {}
                option_index = int(potion.get("index", slot) or slot)
                body = {"action": "use_potion", "option_index": option_index}
                # Only attach target_index when the selected potion requires a target.
                target_index = _pick_target_index_for_potion(potion, enemies)
                if target_index is not None:
                    body["target_index"] = target_index
                return body
            return self._fallback_from_legal_actions(state)

        if "end_turn" in legal:
            return {"action": "end_turn"}
        return self._fallback_from_legal_actions(state)

    def _decode_card_reward(self, action_id: int, state: Dict) -> Dict:
        cards = state.get("card_reward", {}).get("cards", [])
        legal = _get_legal_actions(state)
        if not cards:
            return self._fallback_from_legal_actions(state)

        # action_id in [0, n_cards-1] selects a card; larger ids map to skip when legal.
        if "choose_reward_card" in legal and 0 <= action_id < len(cards):
            card = cards[action_id] if isinstance(cards[action_id], dict) else {}
            option_index = int(card.get("index", action_id) or action_id)
            return {"action": "choose_reward_card", "option_index": option_index}

        if "skip_reward_cards" in legal:
            return {"action": "skip_reward_cards"}

        return {"action": "choose_reward_card", "option_index": 0}

    def _decode_reward(self, action_id: int, state: Dict) -> Dict:
        reward = state.get("reward") or {}
        rewards = reward.get("rewards") or []
        card_options = reward.get("card_options") or []
        legal = _get_legal_actions(state)

        # Reward screen may currently be in card-choice subphase.
        if "choose_reward_card" in legal and isinstance(card_options, list) and card_options:
            if 0 <= action_id < len(card_options):
                card = card_options[action_id] if isinstance(card_options[action_id], dict) else {}
                option_index = int(card.get("index", action_id) or action_id)
                return {"action": "choose_reward_card", "option_index": option_index}
            if "skip_reward_cards" in legal:
                return {"action": "skip_reward_cards"}
            return {"action": "choose_reward_card", "option_index": 0}

        if "skip_reward_cards" in legal and bool(reward.get("pending_card_choice", False)):
            return {"action": "skip_reward_cards"}

        if rewards and "claim_reward" in legal:
            idx = min(max(action_id, 0), len(rewards) - 1)
            reward_item = rewards[idx] if isinstance(rewards[idx], dict) else {}
            option_index = int(reward_item.get("index", idx) or idx)
            return {"action": "claim_reward", "option_index": option_index}
        if "collect_rewards_and_proceed" in legal:
            return {"action": "collect_rewards_and_proceed"}
        if "proceed" in legal:
            return {"action": "proceed"}
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
        idx = min(max(action_id, 0), len(nodes) - 1)
        node = nodes[idx]
        option_index = idx
        if isinstance(node, dict):
            option_index = int(node.get("index", idx) or idx)
        elif isinstance(node, int):
            option_index = node
        return {"action": "choose_map_node", "option_index": option_index}

    def _decode_rest(self, action_id: int, state: Dict) -> Dict:
        """
        Raw API: rest_site.options 来自状态；不要写死“强制锻造/强制选牌升级”的流程。
        这里只负责在可用选项范围内选择 index。
        """
        rest = state.get("rest") or {}
        options = rest.get("options") or []
        legal = _get_legal_actions(state)

        if not isinstance(options, list) or not options or "choose_rest_option" not in legal:
            if "proceed" in legal:
                return {"action": "proceed"}
            return self._fallback_from_legal_actions(state)

        idx = min(max(int(action_id), 0), len(options) - 1)
        option = options[idx] if isinstance(options[idx], dict) else {}
        option_index = int(option.get("index", idx) or idx)
        return {"action": "choose_rest_option", "option_index": option_index}

    def _decode_card_select(self, action_id: int, state: Dict) -> Dict:
        legal = _get_legal_actions(state)
        selection = state.get("selection") or {}

        cards = selection.get("cards", [])
        min_select = int(selection.get("min_select", 1) or 1)
        max_select = int(selection.get("max_select", min_select) or min_select)
        selected_count = int(selection.get("selected_count", 0) or 0)
        requires_confirmation = bool(selection.get("requires_confirmation", False))
        can_confirm = bool(selection.get("can_confirm", False))

        if cards:
            base_idx = max(int(action_id), 0)
            # Prefer cards that are not yet selected to reduce no-op repeat clicks.
            candidate_positions: List[int] = []
            for i, c in enumerate(cards):
                if not isinstance(c, dict):
                    continue
                already_selected = bool(
                    c.get("selected", False)
                    or c.get("is_selected", False)
                    or c.get("chosen", False)
                )
                if not already_selected:
                    candidate_positions.append(i)
            if not candidate_positions:
                candidate_positions = list(range(len(cards)))
            pick_pos = candidate_positions[(base_idx + max(selected_count, 0)) % len(candidate_positions)]
            picked = cards[pick_pos] if isinstance(cards[pick_pos], dict) else {}
            idx = int(picked.get("index", pick_pos) or pick_pos)
        else:
            # Without selection metadata, stay conservative and use option 0.
            idx = 0

        if "confirm_selection" in legal and selected_count >= min_select:
            # Some backends expose confirm_selection before can_confirm flips true.
            # Prefer confirm to avoid repeated select_deck_card loops.
            if can_confirm or selected_count >= max_select or requires_confirmation or "select_deck_card" not in legal:
                return {"action": "confirm_selection"}

        if "select_deck_card" in legal:
            return {"action": "select_deck_card", "option_index": idx}
        if "confirm_selection" in legal:
            return {"action": "confirm_selection"}
        if "close_cards_view" in legal:
            return {"action": "close_cards_view"}

        return self._fallback_from_legal_actions(state)

    def _decode_shop(self, action_id: int, state: Dict) -> Dict:
        shop = state.get("shop") or {}
        legal = _get_legal_actions(state)
        is_open = bool(shop.get("is_open", False))

        # If library is already open, try to make purchases or close it
        if is_open:
            # Try to buy cards
            if "buy_card" in legal:
                cards = shop.get("cards") or []
                buyable = []
                for i, item in enumerate(cards):
                    if not isinstance(item, dict):
                        continue
                    affordable = bool(item.get("affordable", item.get("enough_gold", False)))
                    stocked = bool(item.get("stocked", item.get("available", True)))
                    if affordable and stocked:
                        buyable.append(int(item.get("index", i) or i))
                if buyable:
                    return {"action": "buy_card", "option_index": buyable[action_id % len(buyable)]}

            # Try to buy relics
            if "buy_relic" in legal:
                relics = shop.get("relics") or []
                buyable = []
                for i, item in enumerate(relics):
                    if not isinstance(item, dict):
                        continue
                    affordable = bool(item.get("affordable", item.get("enough_gold", False)))
                    stocked = bool(item.get("stocked", item.get("available", True)))
                    if affordable and stocked:
                        buyable.append(int(item.get("index", i) or i))
                if buyable:
                    return {"action": "buy_relic", "option_index": buyable[action_id % len(buyable)]}

            # Try to buy potions
            if "buy_potion" in legal:
                potions = shop.get("potions") or []
                buyable = []
                for i, item in enumerate(potions):
                    if not isinstance(item, dict):
                        continue
                    affordable = bool(item.get("affordable", item.get("enough_gold", False)))
                    stocked = bool(item.get("stocked", item.get("available", True)))
                    if affordable and stocked:
                        buyable.append(int(item.get("index", i) or i))
                if buyable:
                    return {"action": "buy_potion", "option_index": buyable[action_id % len(buyable)]}

            # Try to remove cards
            if "remove_card_at_shop" in legal:
                return {"action": "remove_card_at_shop"}

            # No more purchases available, close the inventory
            if "close_shop_inventory" in legal:
                return {"action": "close_shop_inventory"}
        else:
            # Library is closed, check for removal or open it
            if "remove_card_at_shop" in legal:
                return {"action": "remove_card_at_shop"}
            
            # Check if there's anything to buy before opening
            cards_available = bool(shop.get("cards") and any(
                bool(item.get("affordable", item.get("enough_gold", False))) and 
                bool(item.get("stocked", item.get("available", True)))
                for item in shop.get("cards", []) if isinstance(item, dict)
            ))
            relics_available = bool(shop.get("relics") and any(
                bool(item.get("affordable", item.get("enough_gold", False))) and 
                bool(item.get("stocked", item.get("available", True)))
                for item in shop.get("relics", []) if isinstance(item, dict)
            ))
            potions_available = bool(shop.get("potions") and any(
                bool(item.get("affordable", item.get("enough_gold", False))) and 
                bool(item.get("stocked", item.get("available", True)))
                for item in shop.get("potions", []) if isinstance(item, dict)
            ))
            
            # Only open if there's something to buy
            if (cards_available or relics_available or potions_available) and "open_shop_inventory" in legal:
                return {"action": "open_shop_inventory"}
            
            # No purchases needed, proceed to leave the shop
            if "proceed" in legal:
                return {"action": "proceed"}

        # Fallback: close if possible, then proceed
        if "close_shop_inventory" in legal and is_open:
            return {"action": "close_shop_inventory"}
        if "proceed" in legal:
            return {"action": "proceed"}
        return self._fallback_from_legal_actions(state)

    def _decode_event(self, action_id: int, state: Dict) -> Dict:
        legal = _get_legal_actions(state)
        event = state.get("event") or {}
        options = event.get("options") or []

        if options and "choose_event_option" in legal:
            idx = min(max(action_id, 0), len(options) - 1)
            opt = options[idx] if isinstance(options[idx], dict) else {}
            option_index = int(opt.get("index", idx) or idx)
            return {"action": "choose_event_option", "option_index": option_index}

        return self._fallback_from_legal_actions(state)

    def _decode_chest(self, action_id: int, state: Dict) -> Dict:
        chest = state.get("chest") or {}
        legal = _get_legal_actions(state)
        relics = chest.get("relic_options") or []
        idx = min(max(action_id, 0), len(relics) - 1) if relics else 0

        if "open_chest" in legal:
            return {"action": "open_chest"}
        if relics and "choose_treasure_relic" in legal:
            relic = relics[idx] if isinstance(relics[idx], dict) else {}
            option_index = int(relic.get("index", idx) or idx)
            return {"action": "choose_treasure_relic", "option_index": option_index}
        if "proceed" in legal:
            return {"action": "proceed"}

        return self._fallback_from_legal_actions(state)

    def get_valid_action_mask(self, state: Dict) -> List[bool]:
        mask = [False] * self.total_actions
        if not _can_act_now(state):
            # Avoid invalid all-False distribution in policy forward.
            mask[0] = True
            return mask
        legal_actions = _get_legal_actions(state)

        def _set_prefix(count: int):
            for i in range(min(max(1, count), self.total_actions)):
                mask[i] = True

        if legal_actions:
            if "end_turn" in legal_actions:
                mask[self.max_hand + self.max_potions] = True

            if "play_card" in legal_actions:
                hand = (state.get("combat") or {}).get("hand", [])
                for i in range(min(len(hand), self.max_hand, self.total_actions)):
                    card = hand[i] if isinstance(hand[i], dict) else {}
                    if bool(card.get("playable", True)):
                        mask[i] = True

            if "use_potion" in legal_actions:
                for i in range(min(self.max_potions, self.total_actions - self.max_hand)):
                    mask[self.max_hand + i] = True

            if "choose_reward_card" in legal_actions:
                options = (state.get("reward") or {}).get("card_options") or (state.get("card_reward") or {}).get("cards") or []
                option_count = len(options) if isinstance(options, list) else 0
                if "skip_reward_cards" in legal_actions:
                    option_count += 1
                _set_prefix(option_count)

            elif "claim_reward" in legal_actions:
                rewards = (state.get("reward") or {}).get("rewards") or []
                _set_prefix(len(rewards) if isinstance(rewards, list) else 1)

            elif "choose_map_node" in legal_actions:
                nodes = self._map_next_options(state)
                _set_prefix(len(nodes))

            elif "choose_rest_option" in legal_actions:
                options = (state.get("rest") or {}).get("options") or []
                _set_prefix(len(options) if isinstance(options, list) else 1)

            elif "choose_event_option" in legal_actions:
                options = (state.get("event") or {}).get("options") or []
                _set_prefix(len(options) if isinstance(options, list) else 1)

            elif "choose_treasure_relic" in legal_actions:
                relics = (state.get("chest") or {}).get("relic_options") or []
                _set_prefix(len(relics) if isinstance(relics, list) else 1)

            elif "choose_bundle" in legal_actions:
                controls = (state.get("card_bundle") or {}).get("ui_controls") or []
                picks = 0
                if isinstance(controls, list):
                    picks = sum(1 for c in controls if isinstance(c, dict) and c.get("role") == "choose_bundle")
                _set_prefix(picks if picks > 0 else 1)

            elif "select_deck_card" in legal_actions:
                cards = (state.get("selection") or {}).get("cards") or []
                _set_prefix(len(cards) if isinstance(cards, list) and cards else 1)

            elif any(
                a in legal_actions for a in (
                    "buy_card", "buy_relic", "buy_potion", "remove_card_at_shop",
                    "open_shop_inventory", "close_shop_inventory", "proceed",
                    "confirm_selection", "collect_rewards_and_proceed", "crystal_auto_play",
                    "open_character_select", "select_character", "embark", "return_to_main_menu",
                )
            ):
                _set_prefix(4)

            if any(mask):
                return mask

            # When legal actions are present but do not map to our policy head
            # (e.g. discard_potion-only transient states), avoid heuristic guesses.
            # Keep at least one valid index to preserve numerical stability.
            mask[0] = True
            return mask

        screen = str(state.get("screen", "") or "").upper()
        combat = state.get("combat", {})
        energy = combat.get("energy", 0)
        hand = combat.get("hand", [])

        if screen == "COMBAT":
            for i, card in enumerate(hand[: self.max_hand]):
                energy_cost = card.get("energy_cost", card.get("cost", 0))
                if isinstance(energy_cost, str):
                    cost_token = energy_cost.strip().upper()
                    cost_val = 0
                    is_x_cost = (cost_token == "X")
                else:
                    cost_val = int(energy_cost or 0)
                    is_x_cost = bool(card.get("costs_x", False))

                if bool(card.get("playable", True)) and (cost_val <= energy or is_x_cost):
                    mask[i] = True
            potions = state.get("potions", [])
            for i in range(min(len(potions), self.max_potions)):
                mask[self.max_hand + i] = True
            mask[self.max_hand + self.max_potions] = True

        elif screen in ("CARD_REWARD", "REWARD", "MAP", "REST", "SHOP", "CARD_SELECTION", "CHEST", "CARD_BUNDLE", "EVENT", "CRYSTAL"):
            for i in range(min(8, self.total_actions)):
                mask[i] = True

        return mask
