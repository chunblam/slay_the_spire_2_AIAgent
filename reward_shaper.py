"""
src/reward/reward_shaper.py  (更新版)

核心改动：
1. CARD_REWARD 分支：从 LLMAdvisor 读取上次的 card_recommendation，
   与 agent 实际选的 card_index 比对，给出 match_bonus / mismatch_penalty
2. 权重控制：只在 confidence >= threshold 时应用 match 塑形，避免 LLM
   不确定时误导策略
3. 修复 import：改为相对导入友好的形式，兼容扁平结构
"""

from typing import Dict, Optional
# 兼容两种项目结构（扁平 / src/xxx 分包）
try:
    from src.llm.llm_advisor import LLMAdvisor
except ImportError:
    from llm_advisor import LLMAdvisor


class RewardShaper:
    """
    奖励塑形模块

    最终奖励 = base_reward
              + rule_weight  * rule_bonus
              + llm_weight   * llm_route_bonus   （非战斗、全局路线评估）
              + card_weight  * card_match_bonus   （CARD_REWARD 时，仅在高置信度下）

    card_match_bonus 规则（方案 A）：
      LLM 推荐 idx，agent 选了 agent_idx：
        • 完全一致                       : +card_match_bonus
        • LLM 建议跳过(-1)，agent 也跳  : +card_match_bonus * 0.5
        • 明确不一致且 conf >= threshold : -card_mismatch_penalty
        • conf < threshold               : 不施加任何 match 塑形（不确定时中性）
    """

    def __init__(
        self,
        llm_advisor: Optional[LLMAdvisor] = None,
        llm_weight: float = 0.3,
        rule_weight: float = 0.5,
        card_weight: float = 0.4,           # 选牌 match 奖励的权重（独立于 llm_weight）
        card_match_bonus: float = 1.0,      # 与 LLM 推荐一致时的原始奖励
        card_mismatch_penalty: float = 0.5, # 明确不一致时的惩罚（应小于 match_bonus）
        confidence_threshold: float = 0.55, # 低于此不施加 match 塑形
    ):
        self.llm_advisor = llm_advisor
        self.llm_weight = llm_weight
        self.rule_weight = rule_weight
        self.card_weight = card_weight
        self.card_match_bonus = card_match_bonus
        self.card_mismatch_penalty = card_mismatch_penalty
        self.confidence_threshold = confidence_threshold

    # ── 主入口 ──────────────────────────────────────────────────────────────

    def shape(
        self,
        base_reward: float,
        prev_state: Dict,
        new_state: Dict,
        action: Dict,
        done: bool,
        agent_card_index: Optional[int] = None,  # 新增：agent 实际选的卡牌索引
    ) -> float:
        """
        计算最终塑形奖励。

        新增参数 agent_card_index：
            train.py 在 CARD_REWARD 时把 agent 决策的卡牌索引传进来，
            用于和 LLM 推荐做比对。None 表示非选牌动作，跳过 match 计算。
        """
        total = base_reward

        # 1. 规则奖励（始终计算）
        rule_bonus = self._compute_rule_bonus(prev_state, new_state, action)
        total += self.rule_weight * rule_bonus

        # 2. 选牌 match 奖励（CARD_REWARD 专用）
        if agent_card_index is not None and self.llm_advisor is not None:
            card_bonus = self._compute_card_match_bonus(agent_card_index)
            total += self.card_weight * card_bonus
            # 用完即重置，避免同一推荐被多步重复消费
            self.llm_advisor.invalidate_card_recommendation()

        # 3. 全局路线奖励（非战斗屏幕，非 CARD_REWARD 时）
        screen = new_state.get("screen_type", "")
        if (
            self.llm_advisor is not None
            and self._should_query_llm(screen)
            and screen != "CARD_REWARD"   # CARD_REWARD 的 LLM 调用已在 train.py 里完成
        ):
            llm_bonus = self.llm_advisor.get_reward_shaping_bonus(new_state)
            total += self.llm_weight * llm_bonus

        return total

    # ── 选牌 match bonus ─────────────────────────────────────────────────────

    def _compute_card_match_bonus(self, agent_card_index: int) -> float:
        """
        根据 LLM 上次推荐 vs agent 实际选择计算 match bonus。

        agent_card_index:
          -1  = agent 选择跳过
          0~n = agent 选了第 n 张
        """
        if self.llm_advisor is None:
            return 0.0

        rec_idx, conf = self.llm_advisor.get_last_card_recommendation()

        # LLM 还没做推荐（-99）或置信度不够 → 中性
        if rec_idx == -99 or conf < self.confidence_threshold:
            return 0.0

        # 完全一致
        if agent_card_index == rec_idx:
            bonus = self.card_match_bonus
            print(f"[RewardShaper] ✅ 选牌与 LLM 一致 (idx={rec_idx}, conf={conf:.2f}) → +{bonus * self.card_weight:.3f}")
            return bonus

        # LLM 建议跳过，agent 也跳过
        if rec_idx == -1 and agent_card_index == -1:
            bonus = self.card_match_bonus * 0.5
            print(f"[RewardShaper] ✅ 双方均跳过 → +{bonus * self.card_weight:.3f}")
            return bonus

        # 高置信度时不一致 → 轻微惩罚
        if conf >= self.confidence_threshold + 0.1:   # 略高于阈值才惩罚
            penalty = -self.card_mismatch_penalty
            print(f"[RewardShaper] ⚠️  选牌与 LLM 不一致 "
                  f"(agent={agent_card_index}, llm={rec_idx}, conf={conf:.2f}) "
                  f"→ {penalty * self.card_weight:.3f}")
            return penalty

        return 0.0   # 置信度处于灰色区间，中性

    # ── 规则奖励（不变，补充注释）──────────────────────────────────────────

    def _compute_rule_bonus(
        self, prev_state: Dict, new_state: Dict, action: Dict
    ) -> float:
        bonus = 0.0
        # STS2AIAgent 使用根字段 "action"；旧式封装可能用 "type"
        kind = action.get("action") or action.get("type", "")

        # ── 选牌本身的规则奖励（不依赖 LLM，单纯动作类型）──────────────────
        # STS2AIAgent: choose_reward_card | skip_reward_cards
        if kind == "choose_reward_card":
            bonus += 0.05
        elif kind == "choose_reward":
            payload = action.get("payload", {})
            if not payload.get("skip", True):
                bonus += 0.05

        # ── 出牌奖励 ────────────────────────────────────────────────────────
        elif kind == "play_card":
            prev_combat = prev_state.get("combat", {})
            new_combat  = new_state.get("combat", {})
            if prev_combat and new_combat:
                # 伤害
                prev_hp = sum(m.get("hp", 0) for m in prev_combat.get("monsters", []))
                new_hp  = sum(m.get("hp", 0) for m in new_combat.get("monsters", []))
                dmg = prev_hp - new_hp
                if dmg > 0:
                    bonus += min(dmg / 50.0, 0.5)
                # 格挡（HP 越低价值越高）
                prev_blk = prev_combat.get("player", {}).get("block", 0)
                new_blk  = new_combat.get("player", {}).get("block", 0)
                blk_gain = new_blk - prev_blk
                if blk_gain > 0:
                    hp_ratio = (
                        new_combat.get("player", {}).get("hp", 1)
                        / max(new_combat.get("player", {}).get("max_hp", 1), 1)
                    )
                    bonus += min(blk_gain / 30.0, 0.3) * (1.5 - hp_ratio)

        # ── 结束回合（浪费能量惩罚）────────────────────────────────────────
        elif kind == "end_turn":
            combat = prev_state.get("combat", {})
            if combat:
                energy = combat.get("energy", 0)
                hand   = combat.get("hand", [])
                playable = [
                    c for c in hand
                    if isinstance(c.get("cost"), int) and c.get("cost", 0) <= energy
                ]
                if playable and energy > 0:
                    bonus -= 0.1 * len(playable)

        # ── 休息 / 锻造（Raw API: choose_rest_option + index）────────────────
        elif kind in ("rest", "choose_rest_option", "smith"):
            idx = int(action.get("option_index", action.get("index", 0))) if kind == "choose_rest_option" else 0
            if kind == "smith" or (kind == "choose_rest_option" and idx == 1):
                bonus += 0.3
            elif kind in ("rest", "choose_rest_option") and idx == 0:
                player   = new_state.get("combat", {}).get("player", {})
                hp       = player.get("hp", 0)
                max_hp   = max(player.get("max_hp", 1), 1)
                hp_ratio = hp / max_hp
                if hp_ratio < 0.4:
                    bonus += 0.5
                elif hp_ratio > 0.8:
                    bonus -= 0.2

        # ── 牌组质量持续项 ──────────────────────────────────────────────────
        deck = new_state.get("deck", [])
        if deck:
            quality = self._evaluate_deck_quality(deck)
            bonus += (quality - 0.5) * 0.05

        return bonus

    def _evaluate_deck_quality(self, deck: list) -> float:
        if not deck:
            return 0.5
        total      = len(deck)
        size_pen   = max(0, total - 20) * 0.02
        bad        = sum(1 for c in deck if c.get("type") in ("STATUS", "CURSE"))
        bad_ratio  = bad / total
        upgraded   = sum(1 for c in deck if c.get("upgraded", False))
        upg_ratio  = upgraded / total
        quality    = 0.5 - size_pen - bad_ratio * 0.3 + upg_ratio * 0.2
        return max(0.0, min(1.0, quality))

    @staticmethod
    def _should_query_llm(screen: str) -> bool:
        return screen not in ("COMBAT", "NONE", "LOADING", "")