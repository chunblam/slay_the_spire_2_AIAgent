import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import requests

from state_encoder import StateEncoder
from action_space import STS2ActionSpace

class STS2Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18080,
        max_hand_size: int = 10,
        max_potions: int = 5,
        render_mode: Optional[str] = None,
        timeout: float = 30.0,
        character_index: int = 0,
        startup_debug: bool = False,
        action_poll_interval: float = 0.5,
        action_min_interval: float = 0.5,
        post_action_settle: float = 0.5,
        action_retry_count: int = 1,
        game_mode: str = "singleplayer",
    ):
        super().__init__()
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.render_mode = render_mode
        self.character_index = max(0, character_index)
        self.startup_debug = startup_debug
        self.action_poll_interval = max(0.1, action_poll_interval)
        self.action_min_interval = max(0.0, action_min_interval)
        self.post_action_settle = max(0.0, post_action_settle)
        self.action_retry_count = max(0, int(action_retry_count))
        self._last_action_at = 0.0

        # Session API is the primary route; /state is kept as a compatibility fallback.
        self._state_endpoint_path = "/api/v1/session/state"
        self._action_endpoint_path = "/api/v1/session/action"
        self._fallback_state_endpoint_path = "/state"

        self.encoder = StateEncoder()
        self.action_handler = STS2ActionSpace(
            max_hand_size=max_hand_size,
            max_potions=max_potions,
        )

        self.observation_space = self.encoder.get_observation_space()
        self.action_space = gym.spaces.Discrete(self.action_handler.total_actions)

        self._current_state: Optional[Dict] = None
        self._episode_reward: float = 0.0
        self._step_count: int = 0
        self._startup_character_selected: bool = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self._episode_reward = 0.0
        self._step_count = 0

        state = self._ensure_run_ready(timeout_sec=120.0)
        self._current_state = state

        obs = self.encoder.encode(state)
        info = self._build_info(state)
        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert self._current_state is not None, "reset() must be called first"

        # Use the previously stabilized state directly to avoid extra GET before dispatch.
        prev_state = self._current_state

        # One preflight check before POST to avoid dispatching during transient phases.
        if not self._can_act_now(prev_state) and not self._is_terminal_state(prev_state):
            refreshed = self._get_state()
            if not self._can_act_now(refreshed) and not self._is_terminal_state(refreshed):
                raise RuntimeError(
                    f"step() called but state not actionable: screen={refreshed.get('screen')} can_act={refreshed.get('can_act')}"
                )
            prev_state = refreshed
            self._current_state = prev_state

        api_call = self.action_handler.decode(action, prev_state)
        # Dispatch exactly once; no retry for gameplay actions.
        new_state = self._post_action_once(api_call)
        new_state = self._wait_until_actionable_or_terminal(new_state, max_wait=30.0)
        self._current_state = new_state
        self._step_count += 1

        reward, done = self._compute_reward(prev_state, new_state)
        self._episode_reward += reward

        obs = self.encoder.encode(new_state)
        info = self._build_info(new_state)
        info["action_executed"] = api_call

        truncated = self._step_count >= 1000
        return obs, reward, done, truncated, info

    def refresh_state(self) -> Tuple[Dict, Dict]:
        st = self._get_state()
        self._current_state = st
        return self.encoder.encode(st), self._build_info(st)

    def step_manual_intervention(
        self,
        prev_state: Dict,
        max_wait: float = 180.0,
        poll: Optional[float] = None,
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        new_state, changed = self._wait_for_manual_state_change(prev_state, max_wait=max_wait, poll=poll)
        self._current_state = new_state
        self._step_count += 1

        reward, done = self._compute_reward(prev_state, new_state)
        self._episode_reward += reward
        obs = self.encoder.encode(new_state)
        info = self._build_info(new_state)
        info["action_executed"] = {"action": "manual_intervention"}
        info["manual_intervention"] = True
        info["manual_intervention_reason"] = "unknown_state_no_available_actions"
        info["manual_intervention_changed"] = changed
        info["manual_state_delta"] = self._build_state_delta(prev_state, new_state)
        truncated = self._step_count >= 1000
        return obs, reward, done, truncated, info

    def render(self):
        if self.render_mode == "human" and self._current_state:
            self._print_state(self._current_state)

    def close(self):
        pass

    @staticmethod
    def _unwrap_envelope(payload: Any) -> Dict:
        if not isinstance(payload, dict):
            return {}
        if "ok" in payload and "data" in payload:
            data = payload.get("data")
            return data if isinstance(data, dict) else {}
        if payload.get("status") == "error":
            if isinstance(payload.get("state"), dict):
                return payload.get("state")
            raise RuntimeError(str(payload.get("error", payload)))
        if isinstance(payload.get("state"), dict):
            return payload.get("state")
        return payload

    @staticmethod
    def _normalize_state(raw_state: Dict, fallback_state: Optional[Dict] = None) -> Dict:
        out: Dict[str, Any] = {}
        fallback_state = fallback_state or {}

        def _pick(key: str, default: Any = None) -> Any:
            if key in raw_state:
                return raw_state.get(key)
            if key in fallback_state:
                return fallback_state.get(key)
            return default

        def _pick_dict(key: str) -> Dict[str, Any]:
            raw_val = raw_state.get(key) if isinstance(raw_state.get(key), dict) else {}
            fb_val = fallback_state.get(key) if isinstance(fallback_state.get(key), dict) else {}
            return {**fb_val, **raw_val}

        # Preserve raw GET /api/v1/session/state screen keyword for downstream reward logic.
        screen_raw = str(_pick("screen", "UNKNOWN") or "UNKNOWN").upper()
        state_type = str(screen_raw).lower()

        legal_actions = [str(a) for a in (_pick("legal_actions") or [])]
        if not legal_actions:
            legal_actions = [str(a) for a in (_pick("available_actions") or [])]

        run = _pick_dict("run")
        # STS2AIAgent uses "combat"; session state combat is partial, so merge fallback/full state.
        combat_raw = raw_state.get("combat") if isinstance(raw_state.get("combat"), dict) else {}
        combat_fb = fallback_state.get("combat") if isinstance(fallback_state.get("combat"), dict) else {}
        combat = {**combat_fb, **combat_raw}
        player_raw = combat_raw.get("player") if isinstance(combat_raw.get("player"), dict) else {}
        player_fb = combat_fb.get("player") if isinstance(combat_fb.get("player"), dict) else {}
        player = {**player_fb, **player_raw}

        out["state_type"] = state_type
        out["screen"] = screen_raw
        out["raw_screen"] = screen_raw
        out["screen_type"] = screen_raw
        session_phase = str(_pick("phase") or "").strip().lower()
        if session_phase:
            out["phase"] = session_phase
        else:
            out["phase"] = "run" if screen_raw not in ("", "MAIN_MENU", "CHARACTER_SELECT", "MODAL", "UNKNOWN") else "menu"

        can_act_raw = _pick("can_act")
        out["can_act"] = bool(can_act_raw) if can_act_raw is not None else bool(legal_actions)
        out["block_reason"] = _pick("block_reason")
        out["available_actions"] = legal_actions
        out["legal_actions"] = legal_actions
        out["in_combat"] = bool(_pick("in_combat", False))

        out["floor"] = int(_pick("floor", run.get("floor", 0)) or 0)
        out["gold"] = int(_pick("gold", run.get("gold", 0)) or 0)
        # API 标准字段：run.deck[], run.relics[], run.potions[]
        raw_deck = run.get("deck", _pick("deck", []))
        raw_relics = run.get("relics", _pick("relics", []))
        raw_potions = run.get("potions", _pick("potions", []))
        if (not isinstance(raw_potions, list)) or (not raw_potions):
            raw_potions = combat.get("potions", [])
        # 验证类型且保留原始 API 格式
        out["deck"] = [c for c in raw_deck if isinstance(c, dict)] if isinstance(raw_deck, list) else []
        out["relics"] = [r for r in raw_relics if isinstance(r, dict)] if isinstance(raw_relics, list) else []
        out["potions"] = [p for p in raw_potions if isinstance(p, dict)] if isinstance(raw_potions, list) else []
        out["game_over"] = _pick("game_over") or {}

        combat_payload: Dict[str, Any] = {}
        combat_payload["energy"] = int(player.get("energy", 0) or 0)
        combat_payload["max_energy"] = int(player.get("max_energy", combat.get("max_energy", 3)) or 3)
        # STS2AIAgent session state exposes turn as top-level field.
        combat_payload["round"] = int(_pick("turn", 0) or 0)
        combat_payload["turn"] = str(combat.get("turn", ""))
        combat_payload["combat_type"] = state_type if state_type in ("monster", "elite", "boss", "combat") else ""
        combat_payload["player"] = {
            "current_hp": int(player.get("current_hp", player.get("hp", 0)) or 0),
            "hp": int(player.get("current_hp", player.get("hp", 0)) or 0),
            "max_hp": int(player.get("max_hp", 1) or 1),
            "block": int(player.get("block", 0) or 0),
            "energy": int(player.get("energy", 0) or 0),
            "buffs": player.get("powers", player.get("buffs", [])) if isinstance(player.get("powers", player.get("buffs", [])), list) else [],
        }

        # API 提供的手牌数据已经是标准格式：card_id, name, upgraded, card_type, energy_cost 等
        # 直接使用 combat.hand，只需要补充派生或缺失的字段
        hand_source = combat.get("hand", [])
        hand_payload = []
        energy = int(player.get("energy", 0) or 0)
        for idx, card in enumerate(hand_source or []):
            if not isinstance(card, dict):
                continue
            # API 字段直接映射，不再做别名转换
            card_energy_cost = int(card.get("energy_cost", 0) or 0)
            # 如果 playable 字段缺失，基于能量推算
            playable = card.get("playable")
            if playable is None:
                playable = (card_energy_cost <= energy) or bool(card.get("costs_x", False))
            
            hand_payload.append({
                "index": card.get("index", idx),
                "card_id": card.get("card_id"),
                "name": card.get("name"),
                "upgraded": bool(card.get("upgraded", False)),
                "target_type": card.get("target_type"),
                "requires_target": bool(card.get("requires_target", False)),
                "costs_x": bool(card.get("costs_x", False)),
                "star_costs_x": bool(card.get("star_costs_x", False)),
                "energy_cost": card_energy_cost,
                "star_cost": int(card.get("star_cost", 0) or 0),
                "playable": bool(playable),
                "unplayable_reason": card.get("unplayable_reason"),
                # 额外推导字段供 encoder 使用
                "damage": int(card.get("damage", 0) or 0),
                "block": int(card.get("block", 0) or 0),
            })
        combat_payload["hand"] = hand_payload

        # API 敌人字段：enemy_id, current_hp, move_id, intents[], powers[] 等
        monsters_payload = []
        for idx, enemy in enumerate((combat.get("enemies") or [])):
            if not isinstance(enemy, dict):
                continue
            # 直接使用 API 字段 intents[]
            intents_payload = []
            raw_intents = enemy.get("intents")
            if isinstance(raw_intents, list):
                for raw_intent in raw_intents:
                    if not isinstance(raw_intent, dict):
                        continue
                    # API 字段：intent_type, hits, damage, total_damage, status_card_count
                    intent_type = raw_intent.get("intent_type", "")
                    damage = int(raw_intent.get("damage", 0) or 0)
                    hits = int(raw_intent.get("hits", 1) or 1)
                    total_damage = raw_intent.get("total_damage")
                    if total_damage is None:
                        total_damage = damage * hits
                    else:
                        total_damage = int(total_damage or 0)
                    intents_payload.append({
                        "index": raw_intent.get("index"),
                        "intent_type": intent_type,
                        "label": raw_intent.get("label"),
                        "damage": damage,
                        "hits": hits,
                        "total_damage": total_damage,
                        "status_card_count": raw_intent.get("status_card_count"),
                    })

            primary_intent = intents_payload[0] if intents_payload else {
                "intent_type": "UNKNOWN",
                "damage": 0,
                "hits": 1,
                "total_damage": 0,
            }

            # 收集 powers
            powers_list = enemy.get("powers", [])
            if not isinstance(powers_list, list):
                powers_list = []

            monsters_payload.append({
                "index": enemy.get("index", idx),
                "enemy_id": enemy.get("enemy_id"),
                "name": enemy.get("name"),
                "current_hp": int(enemy.get("current_hp", 0) or 0),
                "max_hp": int(enemy.get("max_hp", 1) or 1),
                "block": int(enemy.get("block", 0) or 0),
                "is_alive": bool(enemy.get("is_alive", (enemy.get("current_hp", 0) or 0) > 0)),
                "is_hittable": bool(enemy.get("is_hittable", True)),
                "powers": powers_list,
                "move_id": enemy.get("move_id"),
                "intents": intents_payload,
                # 兼容性：primary intent 映射到顶层 intent（但推荐使用 intents[]）
                "intent": {
                    "intent_type": str(primary_intent.get("intent_type", "UNKNOWN")),
                    "damage": int(primary_intent.get("damage", 0) or 0),
                    "hits": int(primary_intent.get("hits", 1) or 1),
                    "total_damage": int(primary_intent.get("total_damage", 0) or 0),
                },
            })
        combat_payload["enemies"] = monsters_payload
        # Temporary compatibility alias for modules not migrated yet.
        combat_payload["monsters"] = monsters_payload
        out["combat"] = combat_payload

        # Reward-like states
        reward = _pick_dict("reward")
        reward_items = reward.get("rewards", reward.get("items", [])) if isinstance(reward, dict) else []
        reward_card_options = reward.get("card_options", []) if isinstance(reward, dict) else []
        reward_alternatives = reward.get("alternatives", []) if isinstance(reward, dict) else []
        out["reward"] = {
            "rewards": reward_items if isinstance(reward_items, list) else [],
            "can_proceed": bool((reward or {}).get("can_proceed", False)),
            "pending_card_choice": bool((reward or {}).get("pending_card_choice", False)),
            "card_options": reward_card_options if isinstance(reward_card_options, list) else [],
            "alternatives": reward_alternatives if isinstance(reward_alternatives, list) else [],
        }

        card_reward = _pick_dict("card_reward")
        card_reward_cards = card_reward.get("cards", []) if isinstance(card_reward, dict) else []
        if not isinstance(card_reward_cards, list) or not card_reward_cards:
            card_reward_cards = reward_card_options if isinstance(reward_card_options, list) else []
        out["card_reward"] = {
            "cards": card_reward_cards if isinstance(card_reward_cards, list) else [],
            "can_skip": bool((card_reward or {}).get("can_skip", bool(reward_alternatives))),
        }

        # Map
        map_payload = _pick_dict("map")
        out["map"] = {
            "next_options": map_payload.get("next_options", map_payload.get("available_nodes", [])) if isinstance(map_payload.get("next_options", map_payload.get("available_nodes", [])), list) else [],
            "nodes": map_payload.get("nodes", []) if isinstance(map_payload.get("nodes", []), list) else [],
        }

        rest_payload = _pick_dict("rest")
        if not rest_payload:
            rest_payload = _pick_dict("rest_site")
        out["rest"] = rest_payload

        # Shop: normalize item categories for decoder
        shop = _pick_dict("shop")
        cards = shop.get("cards") if isinstance(shop.get("cards"), list) else []
        relics = shop.get("relics") if isinstance(shop.get("relics"), list) else []
        potions = shop.get("potions") if isinstance(shop.get("potions"), list) else []
        removal = None
        if not cards and not relics and not potions:
            shop_items = shop.get("items", []) if isinstance(shop.get("items", []), list) else []
            for i, it in enumerate(shop_items):
                if not isinstance(it, dict):
                    continue
                cat = str(it.get("category", "")).lower()
                idx = int(it.get("index", i) or i)
                norm = dict(it)
                norm["index"] = idx
                if cat == "card":
                    cards.append(norm)
                elif cat == "relic":
                    relics.append(norm)
                elif cat == "potion":
                    potions.append(norm)
                elif cat == "card_removal":
                    removal = norm
        if not removal:
            raw_removal = shop.get("card_removal")
            if isinstance(raw_removal, dict):
                removal = raw_removal
        out["shop"] = {
            **shop,
            "cards": cards,
            "relics": relics,
            "potions": potions,
            "card_removal": removal or {},
        }

        out["event"] = _pick_dict("event")

        chest = _pick_dict("chest")
        treasure = _pick_dict("treasure")
        relic_select = _pick_dict("relic_select")
        chest_relics = []
        if isinstance(chest.get("relic_options"), list):
            chest_relics = chest.get("relic_options")
        elif isinstance(treasure.get("relics"), list):
            chest_relics = treasure.get("relics")
        elif isinstance(relic_select.get("relics"), list):
            chest_relics = relic_select.get("relics")
        out["chest"] = {
            "is_opened": bool(chest.get("is_opened", bool(chest_relics))),
            "relic_options": chest_relics,
            "can_proceed": bool(chest.get("can_proceed", treasure.get("can_proceed", relic_select.get("can_skip", False)))),
        }

        selection_payload = _pick_dict("selection")
        card_select = _pick_dict("card_select")
        hand_select = _pick_dict("hand_select")
        select_cards = selection_payload.get("cards") if isinstance(selection_payload.get("cards"), list) else None
        if select_cards is None:
            select_cards = card_select.get("cards") if isinstance(card_select.get("cards"), list) else hand_select.get("cards", [])

        min_select = int(selection_payload.get("min_select", card_select.get("min_select", hand_select.get("min_select", 1))) or 1)
        max_select = int(selection_payload.get("max_select", card_select.get("max_select", hand_select.get("max_select", min_select))) or min_select)
        selected_count = int(selection_payload.get("selected_count", card_select.get("selected_count", hand_select.get("selected_count", 0))) or 0)
        requires_confirmation = bool(selection_payload.get("requires_confirmation", card_select.get("requires_confirmation", hand_select.get("requires_confirmation", False))))
        can_confirm = bool(selection_payload.get("can_confirm", card_select.get("can_confirm", hand_select.get("can_confirm", False))))

        out["selection"] = {
            "kind": selection_payload.get("kind", card_select.get("kind", hand_select.get("kind", ""))),
            "cards": select_cards if isinstance(select_cards, list) else [],
            "prompt": selection_payload.get("prompt", card_select.get("prompt", hand_select.get("prompt", ""))),
            "min_select": max(0, min_select),
            "max_select": max(max(0, min_select), max_select),
            "selected_count": max(0, selected_count),
            "requires_confirmation": requires_confirmation,
            "can_confirm": can_confirm,
            "can_cancel": bool(card_select.get("can_cancel", hand_select.get("can_cancel", False))),
        }

        out["card_bundle"] = _pick("card_bundle") if isinstance(_pick("card_bundle"), dict) else {}

        return out

    def _fetch_state_payload(self, endpoint_path: str) -> Dict:
        resp = requests.get(f"{self.base_url}{endpoint_path}", timeout=self.timeout)
        resp.raise_for_status()
        return self._unwrap_envelope(resp.json())

    @staticmethod
    def _session_combat_state_complete(session_payload: Dict) -> bool:
        """
        Session combat snapshot is considered complete for action decoding when:
        - combat.hand exists (for play_card)
        - if potion actions are legal, combat.potions exists (for use/discard_potion)
        """
        if not isinstance(session_payload, dict):
            return False
        if str(session_payload.get("screen", "")).upper() != "COMBAT":
            return False

        combat = session_payload.get("combat") or {}
        if not isinstance(combat.get("hand"), list):
            return False

        legal = [str(a) for a in (session_payload.get("legal_actions") or [])]
        need_potions = ("use_potion" in legal) or ("discard_potion" in legal)
        if need_potions and not isinstance(combat.get("potions"), list):
            return False

        return True

    def _get_state(self) -> Dict:
        session_payload: Dict = {}
        full_payload: Dict = {}

        try:
            session_payload = self._fetch_state_payload(self._state_endpoint_path)
        except Exception:
            session_payload = {}

        # Compatibility enrichment from /state is only required when session payload
        # is missing critical fields (or when outside combat).
        need_fallback = True
        if session_payload:
            if self._session_combat_state_complete(session_payload):
                need_fallback = False

        if need_fallback:
            try:
                full_payload = self._fetch_state_payload(self._fallback_state_endpoint_path)
            except Exception:
                full_payload = {}

        if not session_payload and not full_payload:
            raise RuntimeError("failed to fetch both session state and fallback state")

        primary = session_payload if session_payload else full_payload
        return self._normalize_state(primary, fallback_state=full_payload)

    def _post_action(self, body: Dict) -> Dict:
        if "action" not in body:
            raise ValueError(f"POST body missing 'action': {body!r}")
        self._throttle_action_if_needed()
        resp = requests.post(f"{self.base_url}{self._action_endpoint_path}", json=body, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"action failed: status={resp.status_code}, body={body}, response={resp.text}"
            )
        payload = resp.json()
        full_data = self._unwrap_envelope(payload)
        try:
            session_data = self._fetch_state_payload(self._state_endpoint_path)
            data = self._normalize_state(session_data, fallback_state=full_data)
        except Exception:
            data = self._normalize_state(full_data)
        self._last_action_at = time.time()
        if self.post_action_settle > 0:
            time.sleep(self.post_action_settle)
        return data

    def _post_action_once(self, body: Dict) -> Dict:
        """
        POST exactly once for gameplay actions to avoid accidental duplicate dispatch.
        For HTTP 409, return latest state and let caller wait for actionable state.
        """
        if "action" not in body:
            raise ValueError(f"POST body missing 'action': {body!r}")
        self._throttle_action_if_needed()
        resp = requests.post(
            f"{self.base_url}{self._action_endpoint_path}",
            json=body,
            timeout=self.timeout,
        )
        self._last_action_at = time.time()

        if resp.status_code == 409:
            current = self._get_state()
            if self.post_action_settle > 0:
                time.sleep(self.post_action_settle)
            return current

        if resp.status_code >= 400:
            raise RuntimeError(
                f"action failed: status={resp.status_code}, body={body}, response={resp.text}"
            )

        payload = resp.json()
        full_data = self._unwrap_envelope(payload)
        try:
            session_data = self._fetch_state_payload(self._state_endpoint_path)
            data = self._normalize_state(session_data, fallback_state=full_data)
        except Exception:
            data = self._normalize_state(full_data)
        if self.post_action_settle > 0:
            time.sleep(self.post_action_settle)
        return data

    def _execute_action_with_recovery(self, body: Dict, max_retries: Optional[int] = None) -> Dict:
        max_retries = self.action_retry_count if max_retries is None else max(0, max_retries)
        last_err = ""
        for _ in range(max_retries + 1):
            try:
                return self._post_action(body)
            except Exception as ex:
                last_err = str(ex)
                time.sleep(self.action_poll_interval)
        raise RuntimeError(f"action failed after retries: {body} | {last_err}")

    def _ensure_run_ready(self, timeout_sec: float = 120.0) -> Dict:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            st = self._get_state()
            screen = str(st.get("screen", "")).upper()
            legal = [str(a) for a in (st.get("legal_actions") or [])]

            if self.startup_debug:
                print(f"[startup] screen={screen} legal={legal}")

            # Already in a playable run-state
            if screen in (
                "COMBAT", "MAP", "EVENT", "SHOP", "REST", "REWARD",
                "CARD_REWARD", "CARD_SELECTION", "CARD_BUNDLE", "CHEST",
            ):
                self._startup_character_selected = False
                return st

            # Menu bootstrap to new run
            if "open_character_select" in legal:
                self._execute_action_with_recovery({"action": "open_character_select"})
                self._startup_character_selected = False
                continue

            if "select_character" in legal and not self._startup_character_selected:
                self._execute_action_with_recovery({"action": "select_character", "option_index": self.character_index})
                self._startup_character_selected = True
                continue

            if "embark" in legal:
                self._execute_action_with_recovery({"action": "embark"})
                continue

            if "return_to_main_menu" in legal and screen != "MAIN_MENU":
                self._execute_action_with_recovery({"action": "return_to_main_menu"})
                self._startup_character_selected = False
                continue

            time.sleep(self.action_poll_interval)

        raise TimeoutError("timed out waiting for run-ready state")

    @staticmethod
    def _state_is_actionable(state: Dict) -> bool:
        screen = str(state.get("screen", "") or "").upper()
        return bool(screen) and screen != "MAIN_MENU"

    @staticmethod
    def _can_act_now(state: Dict) -> bool:
        return bool(state.get("can_act", True))

    def _wait_for_manual_state_change(
        self,
        prev_state: Dict,
        max_wait: float = 180.0,
        poll: Optional[float] = None,
    ) -> Tuple[Dict, bool]:
        poll = self.action_poll_interval if poll is None else max(0.1, poll)
        start = time.time()
        baseline_sig = self._state_signature(prev_state)
        last_state = prev_state
        while time.time() - start < max_wait:
            st = self._get_state()
            last_state = st
            if self._state_signature(st) != baseline_sig:
                return st, True
            time.sleep(poll)
        return last_state, False

    def _wait_until_actionable_or_terminal(
        self,
        start_state: Dict,
        max_wait: float = 30.0,
        poll: Optional[float] = None,
    ) -> Dict:
        """
        Wait until state is actionable again or terminal.
        `can_act` is the authoritative readiness signal from the Mod API.
        """
        if self._can_act_now(start_state) or self._is_terminal_state(start_state):
            return start_state

        poll = max(0.1, poll if poll is not None else self.action_poll_interval)
        deadline = time.time() + max(1.0, max_wait)
        last_state = start_state
        while time.time() < deadline:
            time.sleep(poll)
            st = self._get_state()
            last_state = st
            if self._can_act_now(st) or self._is_terminal_state(st):
                return st
        return last_state

    def _wait_for_action_stable(
        self,
        start_state: Dict,
        max_wait: float = 20.0,
        poll: Optional[float] = None,
    ) -> Dict:
        """
        Waits for action to stabilize (API processing complete).
        This avoids the multiple-POST problem by ensuring the API has processed 
        the action before the RL environment polls for the next state.
        
        Stability is indicated by the "stable" flag in the response.
        If not present, returns immediately (assumes stable).
        """
        if start_state.get("stable", True):
            return start_state

        poll = self.action_poll_interval if poll is None else max(0.1, poll)
        deadline = time.time() + max(0.1, max_wait)
        last_state = start_state
        
        while time.time() < deadline:
            st = self._get_state()
            last_state = st
            if st.get("stable", True):
                return st
            time.sleep(poll)
        
        return last_state

    @staticmethod
    def _state_signature(state: Dict) -> Tuple:
        combat_player = (state.get("combat") or {}).get("player") or {}
        return (
            state.get("screen", ""),
            state.get("state_type", ""),
            bool(state.get("can_act", False)),
            tuple(state.get("legal_actions") or []),
            int(state.get("floor", 0) or 0),
            int(state.get("gold", 0) or 0),
            int(combat_player.get("hp", 0) or 0),
            int(combat_player.get("block", 0) or 0),
            len((state.get("deck") or [])),
            len((state.get("relics") or [])),
        )

    @staticmethod
    def _is_terminal_state(state: Dict) -> bool:
        screen = str(state.get("screen", "")).upper()
        if screen == "GAME_OVER":
            return True
        if screen in ("MAIN_MENU", "CHARACTER_SELECT"):
            return True
        game_over = state.get("game_over") or {}
        return bool(game_over.get("victory", False) or game_over.get("defeat", False))

    @staticmethod
    def _build_state_delta(prev_state: Dict, new_state: Dict) -> Dict:
        prev_player = (prev_state.get("combat") or {}).get("player") or {}
        new_player = (new_state.get("combat") or {}).get("player") or {}
        return {
            "screen": [prev_state.get("screen", ""), new_state.get("screen", "")],
            "state_type": [prev_state.get("state_type", ""), new_state.get("state_type", "")],
            "can_act": [bool(prev_state.get("can_act", False)), bool(new_state.get("can_act", False))],
            "legal_actions_count": [len(prev_state.get("legal_actions") or []), len(new_state.get("legal_actions") or [])],
            "floor": [int(prev_state.get("floor", 0) or 0), int(new_state.get("floor", 0) or 0)],
            "gold": [int(prev_state.get("gold", 0) or 0), int(new_state.get("gold", 0) or 0)],
            "hp": [int(prev_player.get("hp", 0) or 0), int(new_player.get("hp", 0) or 0)],
            "deck_size": [len(prev_state.get("deck") or []), len(new_state.get("deck") or [])],
            "relic_count": [len(prev_state.get("relics") or []), len(new_state.get("relics") or [])],
        }

    def _throttle_action_if_needed(self):
        if self.action_min_interval <= 0:
            return
        elapsed = time.time() - self._last_action_at
        remain = self.action_min_interval - elapsed
        if remain > 0:
            time.sleep(remain)

    def _compute_reward(self, prev_state: Dict, new_state: Dict) -> Tuple[float, bool]:
        reward = 0.0
        done = False

        if str(prev_state.get("screen", "")).upper() == "GAME_OVER":
            done = True
            return reward, done

        if str(new_state.get("screen", "")).upper() == "GAME_OVER":
            done = True
            return reward, done

        # Terminal transitions may return to menu-like states directly.
        prev_screen = str(prev_state.get("screen", "")).upper()
        new_screen = str(new_state.get("screen", "")).upper()
        if prev_screen not in ("MAIN_MENU", "CHARACTER_SELECT") and new_screen in ("MAIN_MENU", "CHARACTER_SELECT"):
            done = True
            return reward, done

        game_over = new_state.get("game_over") or {}
        if bool(game_over.get("victory", False)) or bool(game_over.get("defeat", False)):
            done = True
            return reward, done

        return reward, done

    def _build_info(self, state: Dict) -> Dict:
        return {
            "screen": state.get("screen", ""),
            "state_type": state.get("state_type", ""),
            "floor": state.get("floor", 0),
            "hp": state.get("combat", {}).get("player", {}).get("hp", 0),
            "max_hp": state.get("combat", {}).get("player", {}).get("max_hp", 0),
            "gold": state.get("gold", 0),
            "deck_size": len(state.get("deck", [])),
            "relics": [r.get("name") for r in state.get("relics", []) if isinstance(r, dict)],
            "legal_actions": state.get("legal_actions", []),
            "available_actions": state.get("available_actions", []),
            "raw_state": state,
        }

    def _print_state(self, state: Dict):
        screen = state.get("screen", "?")
        floor = state.get("floor", 0)
        combat = state.get("combat", {})
        player = combat.get("player", {})
        hp = player.get("hp", "?")
        max_hp = player.get("max_hp", "?")
        gold = state.get("gold", 0)
        print(f"\\n[Floor {floor}] Screen: {screen} | HP: {hp}/{max_hp} | Gold: {gold}")
