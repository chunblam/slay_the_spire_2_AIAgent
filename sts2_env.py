"""
src/env/sts2_env.py

STS2 Gymnasium-compatible Environment
仅使用 STS2AIAgent Session API：
- GET  /api/v1/session/state
- GET  /api/v1/session/legal_actions
- POST /api/v1/session/action
"""

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

        self._session_state_path = "/api/v1/session/state"
        self._session_actions_path = "/api/v1/session/action"
        self._session_legal_actions_path = "/api/v1/session/legal_actions"
        self._legacy_state_path = "/state"

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
        assert self._current_state is not None, "请先调用 reset()"

        prev_state = self._get_state()
        self._current_state = prev_state

        api_call = self.action_handler.decode(action, prev_state)
        new_state = self._execute_action_with_recovery(api_call)
        self._current_state = new_state
        self._step_count += 1

        reward, done = self._compute_reward(prev_state, new_state)
        self._episode_reward += reward

        obs = self.encoder.encode(new_state)
        info = self._build_info(new_state)
        info["action_executed"] = api_call

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
            raise RuntimeError(str(payload.get("error", payload)))
        return payload

    @staticmethod
    def _map_screen(raw_state: Dict) -> str:
        screen = str(raw_state.get("screen", "")).upper()
        reward = raw_state.get("reward") or {}
        if screen == "REWARD" and isinstance(reward, dict) and reward.get("pending_card_choice"):
            return "CARD_REWARD"
        mapping = {
            "COMBAT": "COMBAT",
            "MAP": "MAP",
            "REST": "REST",
            "SHOP": "SHOP",
            "EVENT": "EVENT",
            "CHEST": "CHEST",
            "CARD_SELECTION": "CARD_SELECT",
            "REWARD": "REWARD",
            "GAME_OVER": "GAME_OVER",
            "MAIN_MENU": "NONE",
            "CHARACTER_SELECT": "OTHER",
            "MODAL": "OTHER",
        }
        return mapping.get(screen, "OTHER")

    @staticmethod
    def _normalize_state(raw_state: Dict, session_state: Optional[Dict], legal_actions: List[str]) -> Dict:
        out: Dict[str, Any] = {}
        run = raw_state.get("run") or {}
        combat = raw_state.get("combat") or {}
        player = combat.get("player") or {}

        out["screen_type"] = STS2Env._map_screen(raw_state)
        out["phase"] = (session_state or {}).get("phase", "")
        out["can_act"] = bool((session_state or {}).get("can_act", False))
        out["block_reason"] = (session_state or {}).get("block_reason")
        out["legal_actions"] = legal_actions

        out["floor"] = run.get("floor", 0)
        out["gold"] = run.get("gold", 0)
        out["deck"] = run.get("deck", []) if isinstance(run.get("deck"), list) else []
        out["relics"] = run.get("relics", []) if isinstance(run.get("relics"), list) else []
        out["potions"] = run.get("potions", []) if isinstance(run.get("potions"), list) else []
        out["game_over"] = raw_state.get("game_over") or {}

        combat_payload: Dict[str, Any] = {}
        combat_payload["energy"] = player.get("energy", 0)
        combat_payload["player"] = {
            "hp": player.get("current_hp", 0),
            "max_hp": player.get("max_hp", 1),
            "block": player.get("block", 0),
            "energy": player.get("energy", 0),
            "buffs": player.get("powers", []) if isinstance(player.get("powers"), list) else [],
        }

        hand_payload = []
        for idx, card in enumerate(combat.get("hand", []) or []):
            hand_payload.append({
                "index": idx,
                "name": card.get("name"),
                "cost": card.get("energy_cost", 0),
                "damage": card.get("damage", 0) or 0,
                "block": card.get("block", 0) or 0,
                "type": str(card.get("card_type", "ATTACK")).upper(),
                "playable": bool(card.get("playable", True)),
                "requires_target": bool(card.get("requires_target", False)),
            })
        combat_payload["hand"] = hand_payload

        monsters_payload = []
        for idx, enemy in enumerate(combat.get("enemies", []) or []):
            intents = enemy.get("intents", []) or []
            first_intent = intents[0] if intents else {}
            monsters_payload.append({
                "index": idx,
                "name": enemy.get("name"),
                "hp": enemy.get("current_hp", 0),
                "max_hp": enemy.get("max_hp", 1),
                "block": enemy.get("block", 0),
                "is_alive": enemy.get("is_alive", True),
                "intent": {
                    "type": str(first_intent.get("intent_type", "UNKNOWN")).upper(),
                    "damage": first_intent.get("damage", 0) or 0,
                },
            })
        combat_payload["monsters"] = monsters_payload
        out["combat"] = combat_payload

        reward = raw_state.get("reward") or {}
        card_options = reward.get("card_options", []) if isinstance(reward.get("card_options"), list) else []
        out["card_reward"] = {"cards": card_options}
        out["reward"] = reward if isinstance(reward, dict) else {}

        map_payload = raw_state.get("map") or {}
        out["map"] = {
            "next_options": map_payload.get("available_nodes", []) if isinstance(map_payload.get("available_nodes"), list) else [],
            "nodes": map_payload.get("nodes", []) if isinstance(map_payload.get("nodes"), list) else [],
        }

        out["rest"] = raw_state.get("rest") or {}
        out["shop"] = raw_state.get("shop") or {}
        out["event"] = raw_state.get("event") or {}
        out["chest"] = raw_state.get("chest") or {}
        out["selection"] = raw_state.get("selection") or {}

        return out

    def _get_session_state(self) -> Dict:
        resp = requests.get(f"{self.base_url}{self._session_state_path}", timeout=self.timeout)
        resp.raise_for_status()
        return self._unwrap_envelope(resp.json())

    def _get_legacy_state(self) -> Dict:
        resp = requests.get(f"{self.base_url}{self._legacy_state_path}", timeout=self.timeout)
        resp.raise_for_status()
        return self._unwrap_envelope(resp.json())

    def _get_state(self) -> Dict:
        session_state = self._get_session_state()
        legacy_state = self._get_legacy_state()
        legal_actions = [str(a) for a in session_state.get("legal_actions", []) or []]
        return self._normalize_state(legacy_state, session_state, legal_actions)

    def _post_session_action(self, body: Dict) -> Dict:
        if "action" not in body:
            raise ValueError(f"POST body 缺少 action 字段: {body!r}")
        self._throttle_action_if_needed()
        resp = requests.post(f"{self.base_url}{self._session_actions_path}", json=body, timeout=self.timeout)
        if resp.status_code >= 400:
            text = resp.text
            raise RuntimeError(
                f"session action failed: status={resp.status_code}, body={body}, response={text}"
            )
        payload = self._unwrap_envelope(resp.json())
        if isinstance(payload, dict):
            _ = payload.get("status", "completed")
        self._last_action_at = time.time()
        if self.post_action_settle > 0:
            time.sleep(self.post_action_settle)
        return self._get_state()

    def _execute_action_with_recovery(self, body: Dict, max_retries: Optional[int] = None) -> Dict:
        max_retries = self.action_retry_count if max_retries is None else max(0, max_retries)
        last_err = ""
        for _ in range(max_retries + 1):
            try:
                return self._post_session_action(body)
            except Exception as ex:
                last_err = str(ex)
                time.sleep(self.action_poll_interval)
        raise RuntimeError(f"动作执行失败: {body} | {last_err}")

    def _ensure_run_ready(self, timeout_sec: float = 120.0) -> Dict:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            session_state = self._get_session_state()
            phase = str(session_state.get("phase", "")).lower()
            legal = [str(a) for a in session_state.get("legal_actions", []) or []]
            can_act = bool(session_state.get("can_act", False))
            if self.startup_debug:
                print(f"[startup] phase={phase} can_act={can_act} legal={legal}")

            if phase == "run":
                st = self._get_state()
                if st.get("screen_type") != "GAME_OVER":
                    self._startup_character_selected = False
                    return st

            if phase != "character_select":
                self._startup_character_selected = False

            if not can_act:
                time.sleep(self.action_poll_interval)
                continue

            if "menu_new_run" in legal:
                if self.startup_debug:
                    print('[startup] action={"action":"menu_new_run"}')
                self._post_session_action({"action": "menu_new_run"})
                self._startup_character_selected = False
                continue

            if "embark" in legal:
                if self.startup_debug:
                    print('[startup] action={"action":"embark"}')
                self._post_session_action({"action": "embark"})
                continue

            if "menu_confirm" in legal:
                if self.startup_debug:
                    print('[startup] action={"action":"menu_confirm"}')
                self._post_session_action({"action": "menu_confirm"})
                continue

            if "menu_choose_character" in legal and not self._startup_character_selected:
                body = {"action": "menu_choose_character", "option_index": self.character_index}
                if self.startup_debug:
                    print(f"[startup] action={body}")
                self._post_session_action(body)
                self._startup_character_selected = True
                continue

            if "menu_return" in legal:
                if self.startup_debug:
                    print('[startup] action={"action":"menu_return"}')
                self._post_session_action({"action": "menu_return"})
                self._startup_character_selected = False
                continue

            if "return_to_main_menu" in legal:
                if self.startup_debug:
                    print('[startup] action={"action":"return_to_main_menu"}')
                self._post_session_action({"action": "return_to_main_menu"})
                self._startup_character_selected = False
                continue

            if "confirm_modal" in legal:
                if self.startup_debug:
                    print('[startup] action={"action":"confirm_modal"}')
                self._post_session_action({"action": "confirm_modal"})
                continue

            if "dismiss_modal" in legal:
                if self.startup_debug:
                    print('[startup] action={"action":"dismiss_modal"}')
                self._post_session_action({"action": "dismiss_modal"})
                continue

            time.sleep(self.action_poll_interval)

        raise TimeoutError("自动开局超时：未能进入 run 阶段。")

    @staticmethod
    def _state_is_actionable(state: Dict) -> bool:
        return bool(state.get("screen_type", "")) and state.get("screen_type") != "NONE"

    @staticmethod
    def _can_act_now(state: Dict) -> bool:
        return bool(state.get("can_act", True))

    def _wait_until_actionable(self, max_wait: float = 20.0, poll: Optional[float] = None) -> Dict:
        poll = self.action_poll_interval if poll is None else max(0.1, poll)
        start = time.time()
        last_state = self._current_state or {}
        while time.time() - start < max_wait:
            session_state = self._get_session_state()
            can_act = bool(session_state.get("can_act", False))
            phase = str(session_state.get("phase", "")).lower()
            if can_act and phase == "run":
                st = self._get_state()
                last_state = st
                if self._state_is_actionable(st) and self._can_act_now(st):
                    return st
            time.sleep(poll)
        return last_state

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

        if new_state.get("screen_type") == "GAME_OVER":
            victory = bool((new_state.get("game_over") or {}).get("victory", False))
            reward += 50.0 if victory else -20.0
            done = True
            return reward, done

        prev_combat = prev_state.get("combat", {})
        new_combat = new_state.get("combat", {})
        if prev_combat and new_combat:
            prev_hp = prev_combat.get("player", {}).get("hp", 0)
            new_hp = new_combat.get("player", {}).get("hp", 0)
            reward -= max(prev_hp - new_hp, 0) * 0.1

            prev_alive = len([m for m in prev_combat.get("monsters", []) if m.get("hp", 0) > 0])
            new_alive = len([m for m in new_combat.get("monsters", []) if m.get("hp", 0) > 0])
            reward += max(prev_alive - new_alive, 0) * 5.0

        prev_floor = prev_state.get("floor", 0)
        new_floor = new_state.get("floor", 0)
        if new_floor > prev_floor:
            reward += 1.0

        prev_gold = prev_state.get("gold", 0)
        new_gold = new_state.get("gold", 0)
        if new_gold > prev_gold:
            reward += (new_gold - prev_gold) * 0.01

        return reward, done

    def _build_info(self, state: Dict) -> Dict:
        return {
            "screen_type": state.get("screen_type", ""),
            "floor": state.get("floor", 0),
            "hp": state.get("combat", {}).get("player", {}).get("hp", 0),
            "max_hp": state.get("combat", {}).get("player", {}).get("max_hp", 0),
            "gold": state.get("gold", 0),
            "deck_size": len(state.get("deck", [])),
            "relics": [r.get("name") for r in state.get("relics", []) if isinstance(r, dict)],
            "legal_actions": state.get("legal_actions", []),
            "raw_state": state,
        }

    def _print_state(self, state: Dict):
        screen = state.get("screen_type", "?")
        floor = state.get("floor", 0)
        combat = state.get("combat", {})
        player = combat.get("player", {})
        hp = player.get("hp", "?")
        max_hp = player.get("max_hp", "?")
        gold = state.get("gold", 0)
        print(f"\n[Floor {floor}] Screen: {screen} | HP: {hp}/{max_hp} | Gold: {gold}")