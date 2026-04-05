from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from llm_advisor import LLMAdvisor
except ImportError:
    try:
        from llm_advisor import LLMAdvisor
    except ImportError:
        LLMAdvisor = None  # type: ignore


def _sum_enemy_hp(state: Dict) -> float:
    combat = state.get("combat") or {}
    enemies = combat.get("enemies") or combat.get("monsters") or []
    return float(sum(
        (e.get("current_hp", e.get("hp", 0)) or 0)
        for e in enemies
        if isinstance(e, dict) and bool(e.get("is_alive", True))
    ))


def _alive_enemy_count(state: Dict) -> int:
    combat = state.get("combat") or {}
    enemies = combat.get("enemies") or combat.get("monsters") or []
    return len([
        e for e in enemies
        if isinstance(e, dict)
        and bool(e.get("is_alive", True))
        and float(e.get("current_hp", e.get("hp", 0)) or 0) > 0
    ])


def _player(state: Dict) -> Dict:
    return ((state.get("combat") or {}).get("player") or {})


def _extract_powers(entity: Dict) -> List[Dict]:
    """Read powers with backward-compatible fallback to buffs."""
    powers = entity.get("powers")
    if not isinstance(powers, list):
        powers = entity.get("buffs")
    if not isinstance(powers, list):
        return []
    return [p for p in powers if isinstance(p, dict)]


def _sum_positive_player_buff_amount(state: Dict) -> float:
    player = _player(state)
    total = 0.0
    for p in _extract_powers(player):
        if bool(p.get("is_debuff", False)):
            continue
        total += max(float(p.get("amount", 0) or 0), 0.0)
    return total


def _sum_enemy_debuff_amount(state: Dict) -> float:
    enemies = ((state.get("combat") or {}).get("enemies") or [])
    total = 0.0
    for e in enemies:
        if not isinstance(e, dict):
            continue
        for p in _extract_powers(e):
            if not bool(p.get("is_debuff", False)):
                continue
            total += max(float(p.get("amount", 0) or 0), 0.0)
    return total


def _player_hp(state: Dict) -> float:
    p = _player(state)
    return float(p.get("current_hp", p.get("hp", 0)) or 0)


def _player_max_hp(state: Dict) -> float:
    return max(float(_player(state).get("max_hp", 1) or 1), 1.0)


def _player_hp_ratio(state: Dict) -> float:
    return max(0.0, min(1.0, _player_hp(state) / _player_max_hp(state)))


def _sum_enemy_intent_damage(state: Dict) -> float:
    dmg = 0.0
    enemies = ((state.get("combat") or {}).get("enemies") or [])
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        intents = enemy.get("intents")
        if not isinstance(intents, list):
            intents = []
        for intent in intents:
            if not isinstance(intent, dict):
                continue
            intent_type = str(intent.get("intent_type", "") or "").strip().upper().replace("-", "_").replace(" ", "_")
            if intent_type not in ("ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF"):
                continue
            total_damage = intent.get("total_damage")
            if total_damage is not None:
                dmg += float(total_damage or 0)
            else:
                base_dmg = float(intent.get("damage", 0) or 0)
                hits = int(intent.get("hits", 1) or 1)
                dmg += base_dmg * hits
    return dmg


def _enemy_total_damage(state: Dict) -> float:
    """
    Sum enemy intents total_damage from intents[]; fallback to damage * hits.
    """
    total = 0.0
    enemies = ((state.get("combat") or {}).get("enemies") or [])
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        intents = enemy.get("intents")
        if not isinstance(intents, list):
            intents = []
        for intent in intents:
            if not isinstance(intent, dict):
                continue
            total_damage = intent.get("total_damage")
            if total_damage is not None:
                total += float(total_damage or 0)
            else:
                total += float(intent.get("damage", 0) or 0) * int(intent.get("hits", 1) or 1)
    return total


def _state_phase(state: Dict) -> str:
    return str(state.get("phase", "") or "").lower()


def _state_turn(state: Dict) -> int:
    turn = state.get("turn")
    if turn is not None:
        return int(turn or 0)
    return int(((state.get("combat") or {}).get("round", 0)) or 0)


def _is_in_combat(state: Dict) -> bool:
    in_combat = state.get("in_combat")
    if in_combat is not None:
        return bool(in_combat)
    return str(state.get("screen", "")).upper() == "COMBAT"


@dataclass
class TurnTracker:
    in_combat: bool = False
    turn_number: int = 0
    turn_start_snapshot: Optional[Dict] = None
    expected_enemy_damage: float = 0.0
    acc_kills: int = 0

    def on_combat_start(self, state: Dict):
        self.in_combat = True
        self.turn_number = 0
        self.on_turn_start(state)

    def on_turn_start(self, state: Dict):
        self.turn_number += 1
        self.turn_start_snapshot = state
        self.expected_enemy_damage = _sum_enemy_intent_damage(state)
        self.acc_kills = 0

    def accumulate_kills(self, prev_state: Dict, new_state: Dict):
        self.acc_kills += max(_alive_enemy_count(prev_state) - _alive_enemy_count(new_state), 0)


@dataclass
class CombatTracker:
    in_combat: bool = False
    combat_start_state: Optional[Dict] = None
    combat_start_floor: int = 0
    combat_enemy_type: str = "Monster"
    combat_turns: int = 0

    def on_combat_start(self, state: Dict):
        self.in_combat = True
        self.combat_start_state = state
        self.combat_start_floor = int(state.get("floor", 0) or 0)
        self.combat_enemy_type = self._detect_enemy_type(state)
        self.combat_turns = 0

    @staticmethod
    def _detect_enemy_type(state: Dict) -> str:
        # 优先使用新的 enemy_type 字段（来自 /api/v1/session/state）
        combat = state.get("combat") or {}
        if isinstance(combat, dict):
            enemy_type = str(combat.get("enemy_type", "")).lower()
            if enemy_type == "boss":
                return "BOSS"
            if enemy_type == "elite":
                return "ELITE"
            if enemy_type == "monster":
                return "MONSTER"
        
        enemies = ((state.get("combat") or {}).get("enemies") or [])
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            enemy_id = str(enemy.get("enemy_id", "") or "").upper()
            if "BOSS" in enemy_id or bool(enemy.get("is_boss", False)):
                return "BOSS"
            if "ELITE" in enemy_id or bool(enemy.get("is_elite", False)):
                return "ELITE"
        return "MONSTER"


@dataclass
class PendingTurnPenalty:
    armed: bool = False
    energy_before_end: float = 0.0
    play_card_legal: bool = False
    block_before_end: float = 0.0
    enemy_total_damage_before_end: float = 0.0
    turn_at_end: int = 0
    phase_at_end: str = ""


class RewardShaper:
    def __init__(
        self,
        llm_advisor=None,
        llm_weight: float = 0.3,
        layer_a_weight: float = 1.0,
        layer_b_weight: float = 1.0,
        layer_c_weight: float = 1.0,
        layer_d_weight: float = 0.3,
        layer_e_weight: float = 1.0,
        action_damage_coef: float = 0.004,
        action_block_coef: float = 0.002,
        action_kill_bonus: float = 0.3,
        action_card_pick_bonus: float = 0.05,
        action_potion_bonus: float = 0.05,
        dmg_reward_cap: float = 1.5,
        kill_reward_per_enemy: float = 2.0,
        block_coverage_reward: float = 1.0,
        excess_block_penalty_cap: float = 0.2,
        overflow_block_penalty_coef: float = 0.03,
        energy_waste_penalty: float = 0.5,
        hp_loss_penalty: float = 1.5,
        hp_loss_urgency_max_mul: float = 2.0,
        action_player_buff_gain_bonus: float = 0.05,
        action_enemy_debuff_gain_bonus: float = 0.08,
        player_buff_gain_bonus: float = 0.15,
        enemy_debuff_gain_bonus: float = 0.1,
        normal_combat_bonus: float = 3.0,
        elite_combat_bonus: float = 8.0,
        boss_combat_bonus: float = 20.0,
        boss_extra_bonus: float = 10.0,
        hp_efficiency_max: float = 4.0,
        elite_clean_bonus: float = 3.0,
        elite_clean_threshold: float = 0.3,
        high_hp_loss_threshold: float = 0.5,
        high_hp_loss_penalty_scale: float = 3.0,
        rest_low_hp_bonus: float = 1.0,
        rest_mid_hp_bonus: float = 0.3,
        rest_high_hp_penalty: float = 0.5,
        rest_low_threshold: float = 0.35,
        rest_mid_threshold: float = 0.6,
        rest_high_threshold: float = 0.8,
        smith_bonus: float = 0.5,
        remove_card_bonus: float = 0.4,
        choose_card_meta_bonus: float = 0.3,
        claim_gold_bonus: float = 0.12,
        claim_card_bonus: float = 0.18,
        claim_potion_bonus: float = 0.15,
        claim_relic_bonus: float = 0.25,
        claim_remove_card_bonus: float = 0.2,
        claim_special_card_bonus: float = 0.2,
        claim_linked_set_bonus: float = 0.18,
        claim_unknown_bonus: float = 0.1,
        event_option_bonus: float = 0.1,
        map_node_bonus: float = 0.0,
        deck_optimal_min: int = 10,
        deck_optimal_max: int = 20,
        deck_optimal_bonus: float = 0.05,
        deck_large_threshold: int = 25,
        deck_large_penalty_per_card: float = 0.1,
        deck_too_small_threshold: int = 6,
        deck_too_small_penalty: float = 0.1,
        remove_curse_bonus: float = 0.6,
        buy_card_common_bonus: float = 0.1,
        buy_card_uncommon_bonus: float = 0.25,
        buy_card_rare_bonus: float = 0.4,
        smith_low_upgrade_ratio_threshold: float = 0.3,
        smith_low_upgrade_ratio_mul: float = 1.5,
        buy_bonus: float = 0.2,
        terminal_victory_bonus: float = 100.0,
        terminal_defeat_penalty: float = 30.0,
        terminal_floor_weight: float = 1.5,
        terminal_hp_quality_weight: float = 10.0,
        confidence_threshold: float = 0.55,
        card_weight: float = 0.4,
        event_choice_weight: float = 0.15,
        card_match_bonus: float = 1.0,
        card_mismatch_penalty: float = 0.5,
        event_match_bonus: float = 0.8,
        event_mismatch_penalty: float = 0.4,
        relic_choice_weight: float = 0.25,
        map_route_weight: float = 0.25,
        combat_opening_weight: float = 0.2,
        remove_choice_weight: float = 0.15,
        remove_match_bonus: float = 0.8,
        remove_mismatch_penalty: float = 0.4,
        shop_choice_weight: float = 0.15,
        shop_match_bonus: float = 0.8,
        shop_mismatch_penalty: float = 0.4,
        combo_order_bonus: float = 0.25,
        combo_order_penalty: float = 0.10,
        avoid_card_penalty: float = 0.12,
        turn_goal_bonus: float = 0.40,
        combat_bias_steps: int = 3,
        reward_clip: float = 50.0,
    ):
        self.llm_advisor = llm_advisor
        self.llm_weight = llm_weight

        self.layer_a_weight = layer_a_weight
        self.layer_b_weight = layer_b_weight
        self.layer_c_weight = layer_c_weight
        self.layer_d_weight = layer_d_weight
        self.layer_e_weight = layer_e_weight

        self.action_damage_coef = action_damage_coef
        self.action_block_coef = action_block_coef
        self.action_kill_bonus = action_kill_bonus
        self.action_card_pick_bonus = action_card_pick_bonus
        self.action_potion_bonus = action_potion_bonus

        self.dmg_reward_cap = dmg_reward_cap
        self.kill_reward_per_enemy = kill_reward_per_enemy
        self.block_coverage_reward = block_coverage_reward
        self.excess_block_penalty_cap = excess_block_penalty_cap
        self.overflow_block_penalty_coef = overflow_block_penalty_coef
        self.energy_waste_penalty = energy_waste_penalty
        self.hp_loss_penalty = hp_loss_penalty
        self.hp_loss_urgency_max_mul = hp_loss_urgency_max_mul
        self.action_player_buff_gain_bonus = action_player_buff_gain_bonus
        self.action_enemy_debuff_gain_bonus = action_enemy_debuff_gain_bonus
        self.player_buff_gain_bonus = player_buff_gain_bonus
        self.enemy_debuff_gain_bonus = enemy_debuff_gain_bonus

        self.normal_combat_bonus = normal_combat_bonus
        self.elite_combat_bonus = elite_combat_bonus
        self.boss_combat_bonus = boss_combat_bonus
        self.boss_extra_bonus = boss_extra_bonus
        self.hp_efficiency_max = hp_efficiency_max
        self.elite_clean_bonus = elite_clean_bonus
        self.elite_clean_threshold = elite_clean_threshold
        self.high_hp_loss_threshold = high_hp_loss_threshold
        self.high_hp_loss_penalty_scale = high_hp_loss_penalty_scale

        self.rest_low_hp_bonus = rest_low_hp_bonus
        self.rest_mid_hp_bonus = rest_mid_hp_bonus
        self.rest_high_hp_penalty = rest_high_hp_penalty
        self.rest_low_threshold = rest_low_threshold
        self.rest_mid_threshold = rest_mid_threshold
        self.rest_high_threshold = rest_high_threshold
        self.smith_bonus = smith_bonus
        self.remove_card_bonus = remove_card_bonus
        self.choose_card_meta_bonus = choose_card_meta_bonus
        self.claim_gold_bonus = claim_gold_bonus
        self.claim_card_bonus = claim_card_bonus
        self.claim_potion_bonus = claim_potion_bonus
        self.claim_relic_bonus = claim_relic_bonus
        self.claim_remove_card_bonus = claim_remove_card_bonus
        self.claim_special_card_bonus = claim_special_card_bonus
        self.claim_linked_set_bonus = claim_linked_set_bonus
        self.claim_unknown_bonus = claim_unknown_bonus
        self.event_option_bonus = event_option_bonus
        self.map_node_bonus = map_node_bonus
        self.deck_optimal_min = max(1, int(deck_optimal_min))
        self.deck_optimal_max = max(self.deck_optimal_min, int(deck_optimal_max))
        self.deck_optimal_bonus = deck_optimal_bonus
        self.deck_large_threshold = max(self.deck_optimal_max, int(deck_large_threshold))
        self.deck_large_penalty_per_card = deck_large_penalty_per_card
        self.deck_too_small_threshold = max(1, int(deck_too_small_threshold))
        self.deck_too_small_penalty = deck_too_small_penalty
        self.remove_curse_bonus = remove_curse_bonus
        self.buy_card_common_bonus = buy_card_common_bonus
        self.buy_card_uncommon_bonus = buy_card_uncommon_bonus
        self.buy_card_rare_bonus = buy_card_rare_bonus
        self.smith_low_upgrade_ratio_threshold = smith_low_upgrade_ratio_threshold
        self.smith_low_upgrade_ratio_mul = smith_low_upgrade_ratio_mul
        self.buy_bonus = buy_bonus

        self.terminal_victory_bonus = terminal_victory_bonus
        self.terminal_defeat_penalty = terminal_defeat_penalty
        self.terminal_floor_weight = terminal_floor_weight
        self.terminal_hp_quality_weight = terminal_hp_quality_weight

        self.confidence_threshold = confidence_threshold
        self.card_weight = card_weight
        self.event_choice_weight = event_choice_weight
        self.card_match_bonus = card_match_bonus
        self.card_mismatch_penalty = card_mismatch_penalty
        self.event_match_bonus = event_match_bonus
        self.event_mismatch_penalty = event_mismatch_penalty
        self.relic_choice_weight = relic_choice_weight
        self.map_route_weight = map_route_weight
        self.combat_opening_weight = combat_opening_weight
        self.remove_choice_weight = remove_choice_weight
        self.remove_match_bonus = remove_match_bonus
        self.remove_mismatch_penalty = remove_mismatch_penalty
        self.shop_choice_weight = shop_choice_weight
        self.shop_match_bonus = shop_match_bonus
        self.shop_mismatch_penalty = shop_mismatch_penalty
        self.combo_order_bonus = combo_order_bonus
        self.combo_order_penalty = combo_order_penalty
        self.avoid_card_penalty = avoid_card_penalty
        self.turn_goal_bonus = turn_goal_bonus
        self.combat_bias_steps = max(1, int(combat_bias_steps))
        self.reward_clip = abs(float(reward_clip))

        self.turn_tracker = TurnTracker()
        self.combat_tracker = CombatTracker()
        self.pending_turn_penalty = PendingTurnPenalty()
        self._b_energy_waste = 0.0
        self._b_overflow_block = 0.0
        self._b_hp_loss = 0.0
        self._b_trigger = 0.0
        self._last_combo_order_bonus = 0.0
        self._last_turn_goal_bonus = 0.0
        self._current_combat_turn = 0
        self._current_turn_step = 0
        self._turn_play_log: List[Dict[str, object]] = []
        self.last_breakdown: Dict[str, float] = {}

    def update_layer_weights(self, layer_a: float, layer_b: float, layer_c: float, layer_d: float, layer_e: float):
        """
        动态更新 Layer 权重（用于阶段性权重调整）。
        
        Args:
            layer_a: Layer A（动作级即时奖励）权重
            layer_b: Layer B（回合级结算奖励）权重
            layer_c: Layer C（战斗级结算奖励）权重
            layer_d: Layer D（局外阶段奖励）权重
            layer_e: Layer E（终局奖励）权重
        """
        self.layer_a_weight = float(max(0.0, layer_a))
        self.layer_b_weight = float(max(0.0, layer_b))
        self.layer_c_weight = float(max(0.0, layer_c))
        self.layer_d_weight = float(max(0.0, layer_d))
        self.layer_e_weight = float(max(0.0, layer_e))

    def on_new_combat_turn(self, turn_number: int):
        self._current_combat_turn = max(0, int(turn_number or 0))
        self._current_turn_step = 0

    def shape(
        self,
        base_reward: float,
        prev_state: Dict,
        new_state: Dict,
        action: Dict,
        done: bool,
        agent_card_index: Optional[int] = None,
        agent_relic_index: Optional[int] = None,
        agent_map_index: Optional[int] = None,
        agent_remove_index: Optional[int] = None,
        agent_event_index: Optional[int] = None,
        agent_shop_action: Optional[str] = None,
        agent_shop_index: Optional[int] = None,
        combat_step: Optional[int] = None,
        agent_card_played: Optional[int] = None,
    ) -> float:
        _ = base_reward
        kind = str(action.get("action") or action.get("type") or "")
        prev_in_combat = _is_in_combat(prev_state)
        new_in_combat = _is_in_combat(new_state)
        prev_screen = str(prev_state.get("screen", "")).upper()
        new_screen = str(new_state.get("screen", "")).upper()

        combat_start = (not prev_in_combat and new_in_combat)
        combat_end = (prev_in_combat and not new_in_combat)

        if combat_start:
            self.combat_tracker.on_combat_start(new_state)
            self.turn_tracker.on_combat_start(new_state)
            self.on_new_combat_turn(_state_turn(new_state))

        if self.turn_tracker.in_combat and new_in_combat:
            self.turn_tracker.accumulate_kills(prev_state, new_state)

        a = self.layer_a_action_reward(prev_state, new_state, kind, action)

        self._b_energy_waste = 0.0
        self._b_overflow_block = 0.0
        self._b_hp_loss = 0.0
        self._b_trigger = 0.0
        self._last_combo_order_bonus = 0.0
        self._last_turn_goal_bonus = 0.0

        b = 0.0
        if kind == "end_turn" and prev_in_combat:
            self._arm_pending_end_turn(prev_state)

        if self.turn_tracker.in_combat and prev_in_combat and new_in_combat:
            prev_round = _state_turn(prev_state)
            new_round = _state_turn(new_state)
            if new_round >= prev_round + 1:
                b += self._layer_b_progress(prev_state)
                self._b_hp_loss = self._layer_b4_hp_loss_from_round_transition(prev_state, new_state)
                b += self._b_hp_loss
                self.combat_tracker.combat_turns += 1
                self.turn_tracker.on_turn_start(new_state)
                self.on_new_combat_turn(new_round)

        resolved = self._resolve_pending_end_turn_penalty(new_state)
        self._b_energy_waste = resolved["energy"]
        self._b_overflow_block = resolved["overflow"]
        self._b_trigger = resolved["trigger"]
        b += resolved["total"]

        c = 0.0
        if combat_end and prev_screen == "COMBAT" and new_screen == "REWARD":
            game_over = new_state.get("game_over") or {}
            if isinstance(game_over, dict) and ("is_victory" in game_over):
                is_victory = bool(game_over.get("is_victory", False))
            else:
                # Layer C: combat victory is defined by COMBAT -> REWARD transition.
                is_victory = True
            c = self.layer_c_combat_reward(new_state, is_victory)
            self.turn_tracker.in_combat = False
            self.combat_tracker.in_combat = False
            self._current_combat_turn = 0
            self._current_turn_step = 0
        elif combat_end:
            self.turn_tracker.in_combat = False
            self.combat_tracker.in_combat = False
            self._current_combat_turn = 0
            self._current_turn_step = 0

        d = self.layer_d_meta_reward(prev_state, new_state, action)
        e = self.layer_e_terminal_reward(new_state, done)

        total = (
            self.layer_a_weight * a
            + self.layer_b_weight * b
            + self.layer_c_weight * c
            + self.layer_d_weight * d
            + self.layer_e_weight * e
        )

        llm_route = 0.0
        if (self.llm_advisor is not None
                and self._should_query_llm(new_screen)
                and new_screen != "CARD_REWARD"):
            llm_route = self.llm_advisor.get_reward_shaping_bonus(new_state)
            total += self.llm_weight * llm_route

        llm_card = self._compute_card_match_bonus(agent_card_index) if agent_card_index is not None else 0.0
        llm_event = self._compute_event_match_bonus(agent_event_index) if agent_event_index is not None else 0.0
        llm_relic = self._compute_relic_match_bonus(agent_relic_index) if agent_relic_index is not None else 0.0
        llm_map = self._compute_map_match_bonus(agent_map_index) if agent_map_index is not None else 0.0
        llm_remove = self._compute_remove_match_bonus(prev_state, new_state, action, agent_remove_index) if agent_remove_index is not None else 0.0
        llm_shop = self._compute_shop_match_bonus(agent_shop_action, agent_shop_index) if agent_shop_action is not None else 0.0
        llm_open = 0.0

        total += self.card_weight * llm_card
        total += self.event_choice_weight * llm_event
        total += self.relic_choice_weight * llm_relic
        total += self.map_route_weight * llm_map
        total += self.remove_choice_weight * llm_remove
        total += self.shop_choice_weight * llm_shop
        total += self.combat_opening_weight * llm_open

        if self.llm_advisor is not None:
            if agent_card_index is not None:
                self.llm_advisor.invalidate_card_recommendation()
            if agent_event_index is not None and hasattr(self.llm_advisor, "invalidate_event_recommendation"):
                self.llm_advisor.invalidate_event_recommendation()
            if agent_relic_index is not None:
                self.llm_advisor.invalidate_relic_recommendation()
            if agent_map_index is not None:
                self.llm_advisor.invalidate_map_recommendation()
            if agent_remove_index is not None and hasattr(self.llm_advisor, "invalidate_remove_recommendation"):
                self.llm_advisor.invalidate_remove_recommendation()
            if agent_shop_action is not None and hasattr(self.llm_advisor, "invalidate_shop_recommendation"):
                self.llm_advisor.invalidate_shop_recommendation()

        if self.reward_clip > 0:
            total = max(min(total, self.reward_clip), -self.reward_clip)

        self.last_breakdown = {
            "A_action": float(a),
            "B_turn": float(b),
            "B_energy_waste": float(self._b_energy_waste),
            "B_overflow_block": float(self._b_overflow_block),
            "B_hp_loss": float(self._b_hp_loss),
            "B_trigger": float(self._b_trigger),
            "C_combat": float(c),
            "D_meta": float(d),
            "E_terminal": float(e),
            "LLM_route": float(llm_route),
            "LLM_card": float(llm_card),
            "LLM_event": float(llm_event),
            "LLM_relic": float(llm_relic),
            "LLM_map": float(llm_map),
            "LLM_remove": float(llm_remove),
            "LLM_shop": float(llm_shop),
            "LLM_opening": float(llm_open),
            "LLM_combo_order": float(self._last_combo_order_bonus),
            "LLM_turn_goal": float(self._last_turn_goal_bonus),
            "total": float(total),
        }
        return total

    def layer_a_action_reward(self, prev_state: Dict, new_state: Dict, action_kind: str, executed_action: Optional[Dict] = None) -> float:
        reward = 0.0
        if action_kind == "play_card":
            dmg = max(_sum_enemy_hp(prev_state) - _sum_enemy_hp(new_state), 0.0)
            reward += dmg * self.action_damage_coef
            kills = max(_alive_enemy_count(prev_state) - _alive_enemy_count(new_state), 0)
            if kills > 0:
                reward += kills * self.action_kill_bonus
            prev_blk = float((_player(prev_state)).get("block", 0) or 0)
            new_blk = float((_player(new_state)).get("block", 0) or 0)
            reward += max(new_blk - prev_blk, 0.0) * self.action_block_coef

            # Immediate shaping for self buffs and enemy debuffs applied by us.
            player_buff_gain = max(_sum_positive_player_buff_amount(new_state) - _sum_positive_player_buff_amount(prev_state), 0.0)
            enemy_debuff_gain = max(_sum_enemy_debuff_amount(new_state) - _sum_enemy_debuff_amount(prev_state), 0.0)
            reward += player_buff_gain * self.action_player_buff_gain_bonus
            reward += enemy_debuff_gain * self.action_enemy_debuff_gain_bonus

            card_name = ""
            if isinstance(executed_action, dict):
                card_name = str(executed_action.get("card_name", executed_action.get("name", "")) or "").strip()
            if card_name:
                self._turn_play_log.append({
                    "card_name": card_name,
                    "turn": self._current_combat_turn,
                    "step_in_turn": self._current_turn_step,
                })
                self._last_combo_order_bonus = self._compute_combo_order_bonus(card_name, self._current_combat_turn)
                reward += self._last_combo_order_bonus
                self._current_turn_step += 1
        elif action_kind == "use_potion":
            reward += self.action_potion_bonus
        elif action_kind == "choose_reward_card":
            reward += self.action_card_pick_bonus
        return reward

    def _layer_b_progress(self, end_state: Dict) -> float:
        reward = 0.0
        snap = self.turn_tracker.turn_start_snapshot or end_state
        snap_combat = snap.get("combat") or {}
        end_combat = end_state.get("combat") or {}
        snap_player = snap_combat.get("player") or {}
        end_player = end_combat.get("player") or {}

        max_hp = max(float(snap_player.get("max_hp", 80) or 80), 1.0)

        snap_enemy_hp = float(sum((e.get("current_hp", 0) or 0) for e in (snap_combat.get("enemies") or []) if isinstance(e, dict)))
        end_enemy_hp = float(sum((e.get("current_hp", 0) or 0) for e in (end_combat.get("enemies") or []) if isinstance(e, dict)))
        total_dmg_this_turn = max(snap_enemy_hp - end_enemy_hp, 0.0)
        dmg_efficiency = total_dmg_this_turn / (max_hp * 0.1)
        reward += min(dmg_efficiency * 0.8, self.dmg_reward_cap)

        reward += self.turn_tracker.acc_kills * self.kill_reward_per_enemy

        expected_dmg = _sum_enemy_intent_damage(end_state)
        block_at_end_turn = float(end_player.get("block", 0) or 0)
        if expected_dmg > 0:
            effective_block = min(block_at_end_turn, expected_dmg)
            block_coverage = effective_block / expected_dmg
            reward += block_coverage * self.block_coverage_reward

        # Buff/debuff progression signal (strictly our buffs + enemy debuffs).
        prev_player_buff_amount = _sum_positive_player_buff_amount(snap)
        new_player_buff_amount = _sum_positive_player_buff_amount(end_state)
        reward += max(new_player_buff_amount - prev_player_buff_amount, 0.0) * self.player_buff_gain_bonus

        prev_enemy_debuff_amount = _sum_enemy_debuff_amount(snap)
        new_enemy_debuff_amount = _sum_enemy_debuff_amount(end_state)
        reward += max(new_enemy_debuff_amount - prev_enemy_debuff_amount, 0.0) * self.enemy_debuff_gain_bonus

        self._last_turn_goal_bonus = self._compute_turn_goal_bonus(self._current_combat_turn, snap, end_state)
        reward += self._last_turn_goal_bonus
        return reward

    def _arm_pending_end_turn(self, state: Dict) -> None:
        combat = state.get("combat") or {}
        player = combat.get("player") or {}
        legal = [str(a) for a in (state.get("legal_actions") or [])]
        hand = combat.get("hand") or []
        energy = float(combat.get("energy", player.get("energy", 0)) or 0)

        playable_from_hand = False
        for card in hand:
            if not isinstance(card, dict):
                continue
            if bool(card.get("playable", False)):
                playable_from_hand = True
                break
            if bool(card.get("costs_x", False)) and energy > 0:
                playable_from_hand = True
                break
            cost = int(card.get("energy_cost", 99) or 99)
            if cost <= energy and not str(card.get("unplayable_reason", "") or "").strip():
                playable_from_hand = True
                break

        self.pending_turn_penalty.armed = True
        self.pending_turn_penalty.energy_before_end = energy
        self.pending_turn_penalty.play_card_legal = ("play_card" in legal) and playable_from_hand
        self.pending_turn_penalty.block_before_end = float(player.get("block", 0) or 0)
        self.pending_turn_penalty.enemy_total_damage_before_end = float(_enemy_total_damage(state))
        self.pending_turn_penalty.turn_at_end = _state_turn(state)
        self.pending_turn_penalty.phase_at_end = _state_phase(state)

    def _resolve_pending_end_turn_penalty(self, state: Dict) -> Dict[str, float]:
        if not self.pending_turn_penalty.armed:
            return {"total": 0.0, "energy": 0.0, "overflow": 0.0, "trigger": 0.0}

        phase = _state_phase(state)
        now_turn = _state_turn(state)
        hit_transition = phase == "transition"
        hit_turn_increment = now_turn >= (self.pending_turn_penalty.turn_at_end + 1)
        if not hit_transition and not hit_turn_increment:
            return {"total": 0.0, "energy": 0.0, "overflow": 0.0, "trigger": 0.0}

        energy_penalty = 0.0
        if self.pending_turn_penalty.energy_before_end > 0 and self.pending_turn_penalty.play_card_legal:
            energy_penalty = -self.energy_waste_penalty * self.pending_turn_penalty.energy_before_end

        overflow = max(
            self.pending_turn_penalty.block_before_end - self.pending_turn_penalty.enemy_total_damage_before_end,
            0.0,
        )
        overflow_penalty = -min(overflow * self.overflow_block_penalty_coef, self.excess_block_penalty_cap)

        self.pending_turn_penalty.armed = False
        trigger_code = 1.0 if hit_transition else 2.0
        return {
            "total": energy_penalty + overflow_penalty,
            "energy": energy_penalty,
            "overflow": overflow_penalty,
            "trigger": trigger_code,
        }

    def _layer_b4_hp_loss_from_round_transition(self, prev_state: Dict, new_state: Dict) -> float:
        prev_hp = _player_hp(prev_state)
        new_hp = _player_hp(new_state)
        hp_lost = max(prev_hp - new_hp, 0.0)
        if hp_lost <= 0:
            return 0.0

        max_hp = _player_max_hp(prev_state)
        hp_lost_ratio = hp_lost / max_hp
        current_hp_ratio = new_hp / max_hp
        urgency_mul = 1.0 + max(0.5 - current_hp_ratio, 0.0) * 2.0
        urgency_mul = min(urgency_mul, self.hp_loss_urgency_max_mul)
        penalty = hp_lost_ratio * self.hp_loss_penalty * urgency_mul
        return -penalty

    def layer_b_turn_reward(self, prev_state: Dict, new_state: Dict, action_kind: str) -> float:
        _ = action_kind
        prev_round = _state_turn(prev_state)
        new_round = _state_turn(new_state)
        if new_round < prev_round + 1:
            return 0.0
        return self._layer_b_progress(prev_state) + self._layer_b4_hp_loss_from_round_transition(prev_state, new_state)

    def layer_c_combat_reward(self, new_state: Dict, is_victory: bool) -> float:
        if not is_victory:
            return 0.0

        enemy_type = self.combat_tracker.combat_enemy_type
        type_bonus_map = {
            "MONSTER": self.normal_combat_bonus,
            "ELITE": self.elite_combat_bonus,
            "BOSS": self.boss_combat_bonus,
        }
        reward = type_bonus_map.get(enemy_type, self.normal_combat_bonus)

        start_state = self.combat_tracker.combat_start_state or new_state
        max_hp = _player_max_hp(start_state)
        start_hp = _player_hp(start_state)
        end_hp = _player_hp(new_state)
        hp_lost = max(start_hp - end_hp, 0.0)
        hp_lost_ratio = max(0.0, min(1.0, hp_lost / max_hp))
        reward += (1.0 - hp_lost_ratio) * self.hp_efficiency_max

        if enemy_type == "ELITE" and hp_lost_ratio < self.elite_clean_threshold:
            reward += self.elite_clean_bonus
        if enemy_type == "BOSS":
            reward += self.boss_extra_bonus

        if hp_lost_ratio > self.high_hp_loss_threshold:
            reward -= (hp_lost_ratio - self.high_hp_loss_threshold) * self.high_hp_loss_penalty_scale

        return reward

    def layer_d_meta_reward(self, prev_state: Dict, new_state: Dict, action: Dict) -> float:
        action_kind = str(action.get("action") or action.get("type") or "")
        prev_screen = str(prev_state.get("screen", "") or "").upper()
        new_screen = str(new_state.get("screen", "") or "").upper()
        prev_legal = {str(a) for a in (prev_state.get("legal_actions") or [])}
        prev_reward = prev_state.get("reward") or {}
        reward = 0.0

        def _selection_card_by_option_index() -> Dict:
            selection = prev_state.get("selection") or {}
            cards = selection.get("cards") if isinstance(selection.get("cards"), list) else []
            option_index = int(action.get("option_index", action.get("index", -1)) or -1)
            if option_index < 0 or not cards:
                return {}
            for c in cards:
                if not isinstance(c, dict):
                    continue
                if int(c.get("index", -9999) or -9999) == option_index:
                    return c
            if 0 <= option_index < len(cards) and isinstance(cards[option_index], dict):
                return cards[option_index]
            return {}

        def _remove_card_bonus_from_selection() -> float:
            c = _selection_card_by_option_index()
            card_type = str(c.get("card_type", c.get("type", "")) or "").strip().upper()
            if card_type in ("CURSE", "STATUS"):
                return self.remove_curse_bonus
            return self.remove_card_bonus

        def _shop_item_by_action() -> Dict:
            shop = prev_state.get("shop") or {}
            option_index = int(action.get("option_index", action.get("index", -1)) or -1)
            if option_index < 0:
                return {}
            key = ""
            if action_kind == "buy_card":
                key = "cards"
            elif action_kind == "buy_relic":
                key = "relics"
            elif action_kind == "buy_potion":
                key = "potions"
            if not key:
                return {}
            items = shop.get(key) if isinstance(shop.get(key), list) else []
            for it in items:
                if not isinstance(it, dict):
                    continue
                if int(it.get("index", -9999) or -9999) == option_index:
                    return it
            if 0 <= option_index < len(items) and isinstance(items[option_index], dict):
                return items[option_index]
            return {}

        def _claim_reward_type_bonus() -> float:
            if not isinstance(prev_reward, dict):
                return 0.0

            rewards = prev_reward.get("rewards") or []
            if not isinstance(rewards, list) or not rewards:
                return 0.0

            option_index = int(action.get("option_index", action.get("index", -1)) or -1)
            target = None

            for item in rewards:
                if not isinstance(item, dict):
                    continue
                if int(item.get("index", -9999) or -9999) == option_index:
                    target = item
                    break

            if target is None and 0 <= option_index < len(rewards):
                maybe_item = rewards[option_index]
                if isinstance(maybe_item, dict):
                    target = maybe_item

            if not isinstance(target, dict):
                return 0.0

            reward_type = str(target.get("reward_type", "") or "").strip().upper()
            if reward_type == "GOLD":
                return self.claim_gold_bonus
            if reward_type == "CARD":
                return self.claim_card_bonus
            if reward_type == "POTION":
                return self.claim_potion_bonus
            if reward_type == "RELIC":
                return self.claim_relic_bonus
            if reward_type == "REMOVECARD":
                return self.claim_remove_card_bonus
            if reward_type == "SPECIALCARD":
                return self.claim_special_card_bonus
            if reward_type == "LINKEDREWARDSET":
                return self.claim_linked_set_bonus
            return self.claim_unknown_bonus

        if action_kind == "choose_reward_card":
            reward += self.choose_card_meta_bonus
        elif action_kind == "claim_reward":
            # Use reward.rewards[].reward_type + option_index to give differentiated meta rewards.
            reward += _claim_reward_type_bonus()
        elif action_kind == "select_deck_card":
            # Some reward-card picks are surfaced as select_deck_card by backend/action decoder.
            pending_card_choice = bool(prev_reward.get("pending_card_choice", False)) if isinstance(prev_reward, dict) else False
            reward_card_flow = (
                pending_card_choice
                or "choose_reward_card" in prev_legal
                or prev_screen == "CARD_REWARD"
                or (prev_screen == "REWARD" and new_screen == "REWARD")
            )
            if reward_card_flow:
                reward += self.choose_card_meta_bonus
        elif action_kind == "skip_reward_cards":
            reward += 0.0
        elif action_kind == "choose_rest_option":
            idx = int(action.get("index", action.get("option_index", 0)) or 0)
            if idx == 1:
                deck_cards = prev_state.get("deck") or []
                if isinstance(deck_cards, list) and deck_cards:
                    upgraded_count = sum(1 for c in deck_cards if isinstance(c, dict) and bool(c.get("upgraded", False)))
                    upgrade_ratio = upgraded_count / max(len(deck_cards), 1)
                    if upgrade_ratio < self.smith_low_upgrade_ratio_threshold:
                        reward += self.smith_bonus * self.smith_low_upgrade_ratio_mul
                    else:
                        reward += self.smith_bonus
                else:
                    reward += self.smith_bonus
            elif idx == 0:
                prev_hp_ratio = _player_hp_ratio(prev_state)
                if prev_hp_ratio < self.rest_low_threshold:
                    reward += self.rest_low_hp_bonus
                elif prev_hp_ratio < self.rest_mid_threshold:
                    reward += self.rest_mid_hp_bonus
                elif prev_hp_ratio > self.rest_high_threshold:
                    reward -= self.rest_high_hp_penalty
        elif action_kind == "buy_card":
            item = _shop_item_by_action()
            rarity = str(item.get("rarity", "") or "").strip().upper()
            if rarity == "RARE":
                reward += self.buy_card_rare_bonus
            elif rarity == "UNCOMMON":
                reward += self.buy_card_uncommon_bonus
            else:
                reward += self.buy_card_common_bonus
        elif action_kind in ("buy_relic", "buy_potion"):
            reward += self.buy_bonus
        elif action_kind == "choose_map_node":
            reward += self.map_node_bonus
        elif action_kind == "choose_event_option":
            reward += self.event_option_bonus
        elif action_kind in ("remove_card", "remove_card_at_shop", "scrap"):
            reward += _remove_card_bonus_from_selection()

        prev_deck = len(prev_state.get("deck") or [])
        new_deck = len(new_state.get("deck") or [])
        is_reward_card_select = False
        if action_kind == "select_deck_card" and new_deck < prev_deck:
            # Covers remove-card event/shop flows that execute as select_deck_card.
            reward += _remove_card_bonus_from_selection()
        if action_kind == "select_deck_card":
            pending_card_choice = bool(prev_reward.get("pending_card_choice", False)) if isinstance(prev_reward, dict) else False
            is_reward_card_select = (
                pending_card_choice
                or "choose_reward_card" in prev_legal
                or prev_screen == "CARD_REWARD"
                or (prev_screen == "REWARD" and new_screen in ("REWARD", "CARD_SELECTION"))
            )

        # Deck-size shaping on meta decisions (non-combat strategy quality signal).
        if action_kind in {
            "claim_reward", "choose_reward_card", "skip_reward_cards",
            "choose_rest_option", "buy_card", "buy_relic", "buy_potion",
            "remove_card", "remove_card_at_shop", "scrap", "choose_event_option", "choose_map_node",
        } or (action_kind == "select_deck_card" and (new_deck < prev_deck or is_reward_card_select)):
            deck_size = len(new_state.get("deck") or [])
            if self.deck_optimal_min <= deck_size <= self.deck_optimal_max:
                reward += self.deck_optimal_bonus
            elif deck_size > self.deck_large_threshold:
                reward -= (deck_size - self.deck_large_threshold) * self.deck_large_penalty_per_card
            elif deck_size < self.deck_too_small_threshold:
                reward -= self.deck_too_small_penalty

        return reward

    def layer_e_terminal_reward(self, final_state: Dict, done: bool) -> float:
        if not done:
            return 0.0

        if bool((final_state.get("game_over") or {}).get("is_victory", False)):
            return self.terminal_victory_bonus

        floor = float(final_state.get("floor", 0) or 0)
        hp = _player_hp(final_state)
        hp_quality = (hp / _player_max_hp(final_state)) * self.terminal_hp_quality_weight
        return (
            -self.terminal_defeat_penalty
            + floor * self.terminal_floor_weight
            + hp_quality
        )

    def _compute_card_match_bonus(self, agent_card_index: Optional[int]) -> float:
        if self.llm_advisor is None or agent_card_index is None:
            return 0.0
        rec_idx, conf = self.llm_advisor.get_last_card_recommendation()
        if rec_idx == -99 or conf < self.confidence_threshold:
            return 0.0
        if agent_card_index == rec_idx:
            return self.card_match_bonus
        if rec_idx == -1 and agent_card_index == -1:
            return self.card_match_bonus * 0.5
        if conf >= self.confidence_threshold + 0.1:
            return -self.card_mismatch_penalty
        return 0.0

    def _compute_event_match_bonus(self, agent_event_index: Optional[int]) -> float:
        if self.llm_advisor is None or agent_event_index is None:
            return 0.0
        getter = getattr(self.llm_advisor, "get_last_event_recommendation", None)
        if getter is None:
            return 0.0
        rec_idx, conf = getter()
        if rec_idx == -99 or conf < self.confidence_threshold:
            return 0.0
        if int(agent_event_index) == int(rec_idx):
            return self.event_match_bonus
        if conf >= self.confidence_threshold + 0.1:
            return -self.event_mismatch_penalty
        return 0.0

    def _compute_relic_match_bonus(self, agent_relic_index: Optional[int]) -> float:
        if self.llm_advisor is None or agent_relic_index is None:
            return 0.0
        rec_idx, conf = self.llm_advisor.get_last_relic_recommendation()
        if rec_idx == -99 or conf < self.confidence_threshold:
            return 0.0
        if agent_relic_index == rec_idx:
            return 1.0
        if conf >= self.confidence_threshold + 0.1:
            return -0.5
        return 0.0

    def _compute_map_match_bonus(self, agent_map_index: Optional[int]) -> float:
        if self.llm_advisor is None or agent_map_index is None:
            return 0.0
        rec_idx, conf, route_scores = self.llm_advisor.get_last_map_recommendation()
        if rec_idx == -99 or conf < self.confidence_threshold:
            return 0.0
        if route_scores and 0 <= agent_map_index < len(route_scores):
            agent_score = float(route_scores[agent_map_index])
            best_score = max(float(x) for x in route_scores)
            if best_score <= 1e-6:
                return 0.0
            return max(min((agent_score / best_score) - 0.5, 1.0), -0.5)
        if agent_map_index == rec_idx:
            return 1.0
        if conf >= self.confidence_threshold + 0.1:
            return -0.4
        return 0.0

    def _compute_combo_order_bonus(self, card_name: str, current_turn: int) -> float:
        if self.llm_advisor is None or not card_name:
            return 0.0
        advice = self.llm_advisor.get_last_combat_opening()
        if not advice or int(advice.get("turn", -1) or -1) != int(current_turn):
            return 0.0
        conf = float(advice.get("confidence", 0.0) or 0.0)
        if conf < self.confidence_threshold:
            return 0.0
        avoid_cards = advice.get("avoid_cards", [])
        if isinstance(avoid_cards, list) and card_name in avoid_cards:
            return -self.avoid_card_penalty

        play_order = advice.get("play_order", [])
        if not isinstance(play_order, list):
            return 0.0
        planned = next((item for item in play_order if isinstance(item, dict) and str(item.get("card_name", "")) == card_name), None)
        if planned is None:
            return 0.0
        my_priority = int(planned.get("priority", 99) or 99)
        should_be_before = {
            str(item.get("card_name", ""))
            for item in play_order
            if isinstance(item, dict) and int(item.get("priority", 99) or 99) < my_priority
        }
        already_played = {
            str(entry.get("card_name", ""))
            for entry in self._turn_play_log
            if int(entry.get("turn", -1) or -1) == int(current_turn)
            and str(entry.get("card_name", "")) != card_name
        }
        key_combo = advice.get("key_combo_sequence", [])
        is_key_combo = isinstance(key_combo, list) and card_name in key_combo
        if not should_be_before:
            return self.combo_order_bonus * (1.5 if is_key_combo else 1.0)
        hit_ratio = len(should_be_before & already_played) / max(len(should_be_before), 1)
        if hit_ratio >= 1.0:
            return self.combo_order_bonus * (1.5 if is_key_combo else 1.0)
        if hit_ratio >= 0.5:
            return self.combo_order_bonus * 0.5
        return -self.combo_order_penalty

    def _compute_turn_goal_bonus(self, turn: int, snap_state: Dict, end_state: Dict) -> float:
        if self.llm_advisor is None or turn <= 0:
            return 0.0
        advice = self.llm_advisor.get_last_combat_opening()
        if not advice or int(advice.get("turn", -1) or -1) != int(turn):
            return 0.0
        conf = float(advice.get("confidence", 0.0) or 0.0)
        if conf < self.confidence_threshold:
            return 0.0
        goal = str(advice.get("goal", "") or "").strip().lower()
        if not goal:
            return 0.0

        snap_combat = snap_state.get("combat") or {}
        end_combat = end_state.get("combat") or {}
        snap_enemies = snap_combat.get("enemies") or snap_combat.get("monsters") or []
        end_enemies = end_combat.get("enemies") or end_combat.get("monsters") or []
        snap_enemy_hp = sum(float(e.get("current_hp", e.get("hp", 0)) or 0) for e in snap_enemies if isinstance(e, dict) and bool(e.get("is_alive", True)))
        end_enemy_hp = sum(float(e.get("current_hp", e.get("hp", 0)) or 0) for e in end_enemies if isinstance(e, dict) and bool(e.get("is_alive", True)))
        damage_dealt = max(snap_enemy_hp - end_enemy_hp, 0.0)
        end_player = end_combat.get("player") or {}
        snap_player = snap_combat.get("player") or {}
        player_block = float(end_player.get("block", 0) or 0)
        enemy_intent_dmg = _sum_enemy_intent_damage(snap_state)

        if goal == "aggressive":
            return self.turn_goal_bonus if damage_dealt >= 15 else 0.0
        if goal == "defensive":
            threshold = max(enemy_intent_dmg * 0.8, 8.0)
            return self.turn_goal_bonus if player_block >= threshold else 0.0
        if goal == "buff":
            return self.turn_goal_bonus if _sum_positive_player_buff_amount(end_state) > _sum_positive_player_buff_amount(snap_state) else 0.0
        if goal == "debuff":
            return self.turn_goal_bonus if _sum_enemy_debuff_amount(end_state) > _sum_enemy_debuff_amount(snap_state) else 0.0
        if goal == "mixed":
            dmg_ok = damage_dealt >= 8.0
            block_ok = enemy_intent_dmg > 0 and player_block >= enemy_intent_dmg * 0.5
            if dmg_ok and block_ok:
                return self.turn_goal_bonus
            if dmg_ok or block_ok:
                return self.turn_goal_bonus * 0.5
        return 0.0

    def _compute_remove_match_bonus(
        self,
        prev_state: Dict,
        new_state: Dict,
        action: Dict,
        agent_remove_index: Optional[int],
    ) -> float:
        if self.llm_advisor is None or agent_remove_index is None:
            return 0.0
        if str(action.get("action", "")) != "select_deck_card":
            return 0.0
        prev_deck = len(prev_state.get("deck") or [])
        new_deck = len(new_state.get("deck") or [])
        if new_deck >= prev_deck:
            return 0.0
        getter = getattr(self.llm_advisor, "get_last_remove_recommendation", None)
        if getter is None:
            return 0.0
        rec_idx, conf = getter()
        if rec_idx == -99 or conf < self.confidence_threshold:
            return 0.0
        if int(agent_remove_index) == int(rec_idx):
            return self.remove_match_bonus
        if conf >= self.confidence_threshold + 0.1:
            return -self.remove_mismatch_penalty
        return 0.0

    def _compute_shop_match_bonus(
        self,
        agent_shop_action: Optional[str],
        agent_shop_index: Optional[int],
    ) -> float:
        if self.llm_advisor is None or not agent_shop_action:
            return 0.0
        getter = getattr(self.llm_advisor, "get_last_shop_recommendation", None)
        if getter is None:
            return 0.0
        rec_action, rec_idx, conf = getter()
        if not rec_action or rec_action == "NONE" or conf < self.confidence_threshold:
            return 0.0
        if str(agent_shop_action) == str(rec_action):
            if rec_idx is None or agent_shop_index is None or int(rec_idx) == int(agent_shop_index):
                return self.shop_match_bonus
        if conf >= self.confidence_threshold + 0.1:
            return -self.shop_mismatch_penalty
        return 0.0

    @staticmethod
    def _should_query_llm(screen: str) -> bool:
        return screen not in ("COMBAT", "NONE", "LOADING", "")

    @staticmethod
    def _is_new_player_turn(old_state: Optional[Dict], new_state: Dict) -> bool:
        if old_state is None:
            return False
        old_combat = old_state.get("combat") or {}
        new_combat = new_state.get("combat") or {}
        old_round = int(old_combat.get("round", 0) or 0)
        new_round = int(new_combat.get("round", 0) or 0)
        return new_round >= old_round + 1
