"""
src/llm/llm_advisor.py  (更新版)

核心改动：
1. _retrieve_card_context()  —— 从 knowledge_base["cards"] 精确查找
   候选牌 + 当前牌组牌的数据，送入 prompt，替代单纯靠攻略段落
2. evaluate_card_reward()    —— prompt 里携带 Codex 级别的卡牌效果数据
3. _query_llm()              —— 用卡级检索上下文 + 流派段落双层 context
4. get_advice()              —— 增加 screen 感知，CARD_REWARD 时强制刷新

支持后端: openai / ollama / anthropic
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests


# ─── 数据类 ────────────────────────────────────────────────────────────────────

@dataclass
class LLMAdvice:
    deck_route: str              # 推荐路线: aggro / defense / poison / infinite / ...
    route_score: float           # 路线清晰度 [0, 1]
    key_synergies: List[str]     # 当前卡组中的关键协同
    reward_shaping: float        # 奖励塑形分 [-1, +1]
    reasoning: str               # 推理文字（调试用）
    # 新增：记录上次对选牌的具体推荐（供 RewardShaper 比对 agent 决策）
    card_recommendation: int = -99   # -99=未评估, -1=建议跳过, 0..n=推荐索引
    card_confidence: float = 0.0     # 推荐置信度 [0, 1]，低于阈值不加塑形
    card_scene_type: str = ""
    event_recommendation: int = -99
    event_confidence: float = 0.0
    relic_recommendation: int = -99
    relic_confidence: float = 0.0
    map_recommendation: int = -99
    map_confidence: float = 0.0
    map_route_scores: List[float] = field(default_factory=list)
    remove_recommendation: int = -99
    remove_confidence: float = 0.0
    shop_recommendation_action: str = "NONE"
    shop_recommendation_index: Optional[int] = None
    shop_confidence: float = 0.0
    combat_opening: Dict = field(default_factory=dict)


# ─── LLM 后端封装 ──────────────────────────────────────────────────────────────

class LLMBackend:
    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str = "",
    ):
        self.backend = backend
        self.model = model
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("DEEPSEEK_API_KEY", "")

    def call(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        if self.backend == "openai":
            return self._call_openai(system_prompt, user_prompt, max_tokens)
        elif self.backend == "ollama":
            return self._call_ollama(system_prompt, user_prompt, max_tokens)
        elif self.backend == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt, max_tokens)
        else:
            raise ValueError(f"不支持的后端: {self.backend}")

    def _call_openai(self, system: str, user: str, max_tokens: int) -> str:
        endpoint = f"{(self.base_url or 'https://api.openai.com/v1')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        resp = requests.post(
            endpoint,
            headers=headers, json=payload, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, system: str, user: str, max_tokens: int) -> str:
        endpoint = f"{(self.base_url or 'http://localhost:11434')}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        }
        resp = requests.post(endpoint, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _call_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        headers = {
            "x-api-key": self.api_key or os.getenv("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=payload, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


# ─── JSON 解析工具 ─────────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> Dict:
    """从 LLM 输出里健壮地提取 JSON，兼容 ```json 包裹和多余文字"""
    text = text.strip()
    # 去掉 markdown 代码块
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # 找第一个 { ... } 块
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)


# ─── 主模块 ────────────────────────────────────────────────────────────────────

class LLMAdvisor:
    """
    LLM 策略顾问

    调用时机：
    1. CARD_REWARD 屏幕前（evaluate_card_reward，强制刷新）
    2. 每 call_interval 步的全局评估（get_advice，有缓存）
    3. 地图选择前（get_advice）

    知识检索层次（新）：
    Layer-1: Codex 卡级数据 — 候选牌 + 牌组重叠牌的费用/效果/稀有度
    Layer-2: synergies 表  — 人工标注的 combo 条目
    Layer-3: strategies 段 — 角色流派攻略文字
    """

    # ── System prompts ──────────────────────────────────────────────────────

    SYSTEM_PROMPT_GLOBAL = """你是杀戮尖塔2的顶级策略专家。
分析玩家当前状态，评估牌组路线质量，必须只返回 JSON，不要有任何其他文字。
输出格式（所有字段必填）:
{
  "deck_route": "路线名称",
  "route_score": 0.0到1.0,
  "key_synergies": ["协同1", "协同2"],
  "reward_shaping": -1.0到1.0,
  "reasoning": "简短推理（50字以内）"
}
路线选项: aggro | defense | poison | infinite | strength | exhaust | relic_synergy | mixed
reward_shaping 规则:
  +1.0: 路线极清晰，关键牌全到位
  +0.5: 路线成形，有明显协同
   0.0: 一般，无明显路线
  -0.5: 混乱，路线冲突
  -1.0: 明显错误选择"""

    SYSTEM_PROMPT_CARD_EVAL = """你是杀戮尖塔2的顶级策略专家，专注于牌组构筑。
分析奖励卡选择，结合当前牌组和参考资料，给出最优选择。
必须只返回 JSON，不要有任何其他文字。
输出格式（所有字段必填）:
{
  "recommended_index": 整数（0开始，-1表示跳过），
  "confidence": 0.0到1.0（推荐置信度），
  "reasoning": "推荐理由（100字以内）",
  "deck_route_after": "选择后牌组路线名称",
  "key_combo": "该牌与已有牌/遗物形成的关键配合（若有，否则空字符串）"
}
注意:
- confidence < 0.5 表示不确定，不应用奖励塑形
- 若候选牌均与当前路线无关，建议跳过（-1）而非随便拿一张"""

    SYSTEM_PROMPT_RELIC_EVAL = """你是杀戮尖塔2的顶级策略专家。
你的任务：基于玩家【当前实际牌组、遗物、楼层、HP】，评估遗物选取价值。
重要原则：
- 评估遗物对现有牌组的实际加成，而非理论价值
- 考虑当前楼层（早期灵活性遗物更好，后期协同遗物更好）
- 考虑当前HP（低HP时防御性遗物优先）
- 若多个遗物均无明显协同，选择通用性最强的
必须只返回JSON，不要有任何其他文字：
{
  "recommended_index": 整数（0开始）,
  "confidence": 0.0到1.0,
  "reasoning": "推荐理由（100字以内）",
  "relic_effect_on_deck": "该遗物对当前牌组的具体加成描述",
  "synergy_cards": ["与该遗物有协同的现有牌组中的牌"]
}"""

    SYSTEM_PROMPT_MAP_EVAL = """你是杀戮尖塔2的顶级策略专家。
你的任务：基于玩家当前状态，评估地图各条路线的优先级。
评估维度：
- 当前HP vs 精英战能力（HP高时可提高精英权重）
- 金币 vs 商店价值（有明确购买目标时商店优先）
- 牌组是否需要升级 vs 营地价值
- 整体进度 vs 剩余楼层
必须只返回JSON，不要有任何其他文字：
{
  "recommended_option_index": 整数,
  "confidence": 0.0到1.0,
  "reasoning": "推荐理由（80字以内）",
  "route_value_scores": [与输入路线等长的0.0-1.0评分],
  "priority": "hp_recovery/card_upgrade/relic/shop/progress"
}"""

    SYSTEM_PROMPT_COMBAT_OPENING = """你是杀戮尖塔2的顶级策略专家。
你的任务：在战斗开始前，分析敌人信息和当前手牌，给出本场战斗的最优行动建议。
必须只返回JSON，不要有任何其他文字：
{
  "threat_level": "low/medium/high/critical",
  "priority_action": "attack/defend/mixed",
  "opening_card_sequence": [推荐出牌顺序，按当前手牌索引，例如[2,0,3]],
  "priority_target_index": 优先攻击敌人索引（0开始，可为null）,
  "key_warning": "本战最重要提示（50字以内）",
  "expected_rounds": "预计结束回合数: 1/2/3/4+"
}"""

    SYSTEM_PROMPT_COMBAT_TURN = """你是杀戮尖塔2的顶级策略专家。
请根据【当前回合的真实手牌】、敌人意图、当前增益减益、完整牌库和遗物，规划本回合最优出牌顺序。

分析重点：
1. 先判断本回合更偏进攻、防守、叠buff还是挂debuff
2. 强化/增益牌应尽量在伤害牌之前打出
3. 抽牌牌尽量在仍有能量打新牌时使用
4. X费牌通常应放在本回合末尾
5. 只规划当前真实手牌中存在且本回合合理打出的牌

必须只返回 JSON，不要有任何其他文字：
{
  "turn": 当前回合数,
  "goal": "aggressive|defensive|buff|debuff|mixed",
  "play_order": [
    {
      "card_name": "当前手牌中的中文牌名",
      "priority": 1,
      "role": "buff|debuff|damage|defense|draw|utility",
      "reason": "20字以内"
    }
  ],
  "key_combo_sequence": ["牌1", "牌2"],
  "avoid_cards": ["本回合不建议打出的牌"],
  "priority_target": 优先攻击的敌人索引或null,
  "confidence": 0.0到1.0,
  "reasoning": "本回合核心思路，50字以内"
}"""

    SYSTEM_PROMPT_REMOVE_EVAL = """你是杀戮尖塔2的顶级策略专家。
你的任务：在删牌界面中，基于当前牌组路线和卡组冗余，选择最该移除的一张牌。
必须只返回JSON，不要有任何其他文字：
{
    "recommended_index": 整数（0开始）,
    "confidence": 0.0到1.0,
    "reasoning": "推荐理由（80字以内）"
}"""

    SYSTEM_PROMPT_SHOP_EVAL = """你是杀戮尖塔2的顶级策略专家。
你的任务：在商店界面中，综合金币、当前路线、商店选项，给出最优购买动作。
必须只返回JSON，不要有任何其他文字：
{
    "recommended_action": "buy_card|buy_relic|buy_potion|remove_card_at_shop|proceed|NONE",
    "option_index": 整数或null,
    "confidence": 0.0到1.0,
    "reasoning": "推荐理由（100字以内）"
}
注意：
- 若建议 buy_card/buy_relic/buy_potion，option_index 必须为对应列表内索引。
- 若建议 remove_card_at_shop/proceed/NONE，option_index 可为 null。"""

    def __init__(
        self,
        llm_backend: LLMBackend,
        knowledge_base_path: str = "data/knowledge_base.json",
        call_interval_steps: int = 10,
        cache_ttl: float = 30.0,
        card_shaping_confidence_threshold: float = 0.55,
        combat_bias_steps: int = 3,
    ):
        self.llm = llm_backend
        self.call_interval = call_interval_steps
        self.cache_ttl = cache_ttl
        self.confidence_threshold = card_shaping_confidence_threshold
        self.combat_bias_steps = max(1, int(combat_bias_steps))
        self._last_call_time: float = 0.0
        self._last_advice: Optional[LLMAdvice] = None
        self._step_counter: int = 0

        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        _cards = self.knowledge_base.get("cards", {})
        print(f"[LLMAdvisor] 知识库已加载 — "
              f"cards:{len(_cards)}  "
              f"relics:{len(self.knowledge_base.get('relics', {}))}  "
              f"synergies:{len(self.knowledge_base.get('synergies', []))}  "
              f"strategies:{len(self.knowledge_base.get('strategies', []))}")

    # ── 公开接口 ────────────────────────────────────────────────────────────

    def get_advice(self, state: Dict, force: bool = False) -> Optional[LLMAdvice]:
        """
        获取全局路线评估（带缓存）。
        CARD_REWARD 屏幕时内部会自动 force=True，以便即时刷新。
        """
        self._step_counter += 1
        now = time.time()
        screen = str(state.get("screen", state.get("screen_type", "")) or "").upper()
        pending_card_choice = bool((state.get("reward") or {}).get("pending_card_choice", False))

        # REWARD 且存在选牌时强制刷新
        if screen in ("CARD_REWARD", "REWARD") and pending_card_choice:
            force = True

        if not force and (
            self._step_counter % self.call_interval != 0
            or now - self._last_call_time < self.cache_ttl
        ):
            return self._last_advice

        try:
            advice = self._query_llm_global(state)
            self._last_advice = advice
            self._last_call_time = now
            return advice
        except Exception as e:
            print(f"[LLMAdvisor] get_advice 调用失败: {e}")
            return self._last_advice

    def evaluate_card_reward(
        self,
        state: Dict,
        reward_cards: List[Dict],
    ) -> Tuple[int, float, str]:
        """
        评估奖励卡，返回 (recommended_index, confidence, reasoning)

        - recommended_index: -1=跳过, 0..n=推荐索引
        - confidence: [0,1]，低于阈值时 RewardShaper 不应用 match bonus
        - reasoning: 推理文字（供日志/调试）
        """
        if not reward_cards:
            return -1, 0.0, "无奖励卡"

        deck = state.get("deck", [])
        relics = state.get("relics", [])
        character = state.get("character", {}).get("name", "Unknown")
        floor = state.get("floor", 0)
        hp = state.get("combat", {}).get("player", {}).get("hp", "?")
        max_hp = state.get("combat", {}).get("player", {}).get("max_hp", "?")

        # ── Layer-1: Codex 卡级检索 ────────────────────────────────────────
        candidate_context = self._retrieve_card_context(reward_cards, deck)

        # ── Layer-2: synergies ──────────────────────────────────────────────
        synergy_context = self._retrieve_synergies(reward_cards, deck)

        # ── Layer-3: 流派策略 ───────────────────────────────────────────────
        strategy_context = self._retrieve_strategy(character)

        deck_summary = self._summarize_deck(deck)
        relic_names = [r.get("name", "?") for r in relics[:12]]

        user_prompt = (
            f"=== 游戏状态 ===\n"
            f"角色: {character} | 楼层: {floor} | HP: {hp}/{max_hp}\n"
            f"遗物: {relic_names}\n\n"
            f"=== 当前牌组 ===\n{deck_summary}\n\n"
            f"=== 奖励卡选项（含 Codex 数据）===\n{candidate_context}\n\n"
            f"=== 已知协同 ===\n{synergy_context}\n\n"
            f"=== 角色流派参考 ===\n{strategy_context}\n\n"
            f"请选出最优奖励卡（或跳过），返回 JSON。"
        )

        try:
            resp = self.llm.call(self.SYSTEM_PROMPT_CARD_EVAL, user_prompt, max_tokens=400)
            data = _parse_json_response(resp)

            idx = int(data.get("recommended_index", 0))
            conf = float(data.get("confidence", 0.5))
            reason = data.get("reasoning", "")

            # 边界检查
            if idx >= len(reward_cards):
                idx = 0
            if idx < -1:
                idx = -1

            # 同步更新 _last_advice 里的 card 字段
            if self._last_advice is not None:
                self._last_advice.card_recommendation = idx
                self._last_advice.card_confidence = conf
            else:
                self._last_advice = LLMAdvice(
                    deck_route="mixed", route_score=0.5, key_synergies=[],
                    reward_shaping=0.0, reasoning="card_eval_only",
                    card_recommendation=idx, card_confidence=conf,
                )

            print(f"[LLMAdvisor] 选牌推荐: idx={idx} conf={conf:.2f} | {reason[:60]}")
            return idx, conf, reason

        except Exception as e:
            print(f"[LLMAdvisor] evaluate_card_reward 失败: {e}")
            if self._last_advice is not None:
                self._last_advice.card_recommendation = -99
                self._last_advice.card_confidence = 0.0
            return -1, 0.0, f"LLM 调用失败: {e}"

    def evaluate_relic_choice(
        self,
        state: Dict,
        relic_options: List[Dict],
    ) -> Tuple[int, float, str]:
        """评估遗物选择，返回 (recommended_index, confidence, reasoning)"""
        if not relic_options:
            return 0, 0.0, "无遗物选项"

        deck = state.get("deck", [])
        relics = state.get("relics", [])
        floor = state.get("floor", 0)
        player = (state.get("combat") or {}).get("player", {})
        hp = player.get("hp", "?")
        max_hp = player.get("max_hp", "?")
        options_text = "\n".join(
            f"[{i}] {r.get('name', '?')} | {self._trim(str(r.get('description', '')), 80)}"
            for i, r in enumerate(relic_options)
        )
        user_prompt = (
            f"楼层: {floor} | HP: {hp}/{max_hp}\n"
            f"当前遗物: {[x.get('name', '?') for x in relics[:12]]}\n"
            f"当前牌组摘要:\n{self._summarize_deck(deck)}\n\n"
            f"候选遗物:\n{options_text}\n"
            "请给出最优遗物选择。"
        )
        try:
            resp = self.llm.call(self.SYSTEM_PROMPT_RELIC_EVAL, user_prompt, max_tokens=300)
            data = _parse_json_response(resp)
            idx = int(data.get("recommended_index", 0))
            conf = float(data.get("confidence", 0.5))
            reason = str(data.get("reasoning", ""))
            if idx < 0:
                idx = 0
            if idx >= len(relic_options):
                idx = len(relic_options) - 1

            if self._last_advice is None:
                self._last_advice = LLMAdvice(
                    deck_route="mixed",
                    route_score=0.5,
                    key_synergies=[],
                    reward_shaping=0.0,
                    reasoning="relic_eval_only",
                )
            self._last_advice.relic_recommendation = idx
            self._last_advice.relic_confidence = conf
            return idx, conf, reason
        except Exception as e:
            print(f"[LLMAdvisor] evaluate_relic_choice 失败: {e}")
            return 0, 0.0, f"LLM 调用失败: {e}"

    def evaluate_map_route(
        self,
        state: Dict,
        route_options: List[Dict],
    ) -> Tuple[int, float, str, List[float]]:
        """评估地图路线，返回 (recommended_index, confidence, reasoning, route_scores)"""
        if not route_options:
            return 0, 0.0, "无路线选项", []

        floor = state.get("floor", 0)
        gold = state.get("gold", 0)
        player = (state.get("combat") or {}).get("player", {})
        hp = player.get("hp", "?")
        max_hp = player.get("max_hp", "?")
        options_text = "\n".join(f"[{i}] {o}" for i, o in enumerate(route_options))
        user_prompt = (
            f"楼层: {floor} | HP: {hp}/{max_hp} | 金币: {gold}\n"
            f"当前牌组摘要:\n{self._summarize_deck(state.get('deck', []))}\n\n"
            f"地图可选路线:\n{options_text}\n"
            "请评估并给出最优路线索引。"
        )
        try:
            resp = self.llm.call(self.SYSTEM_PROMPT_MAP_EVAL, user_prompt, max_tokens=320)
            data = _parse_json_response(resp)
            idx = int(data.get("recommended_option_index", 0))
            conf = float(data.get("confidence", 0.5))
            reason = str(data.get("reasoning", ""))
            scores = data.get("route_value_scores", [])
            route_scores = [float(x) for x in scores] if isinstance(scores, list) else []
            if idx < 0:
                idx = 0
            if idx >= len(route_options):
                idx = len(route_options) - 1
            if len(route_scores) != len(route_options):
                route_scores = [0.5] * len(route_options)

            if self._last_advice is None:
                self._last_advice = LLMAdvice(
                    deck_route="mixed",
                    route_score=0.5,
                    key_synergies=[],
                    reward_shaping=0.0,
                    reasoning="map_eval_only",
                )
            self._last_advice.map_recommendation = idx
            self._last_advice.map_confidence = conf
            self._last_advice.map_route_scores = route_scores
            return idx, conf, reason, route_scores
        except Exception as e:
            print(f"[LLMAdvisor] evaluate_map_route 失败: {e}")
            return 0, 0.0, f"LLM 调用失败: {e}", [0.5] * len(route_options)

    def evaluate_combat_opening(self, state: Dict) -> Dict:
        """评估战斗开局建议，返回 opening advice dict。"""
        combat = state.get("combat") or {}
        player = combat.get("player") or {}
        monsters = combat.get("monsters") or []
        hand = combat.get("hand") or []
        monsters_text = "\n".join(
            f"[{i}] {m.get('name', '?')} hp={m.get('hp', '?')} block={m.get('block', 0)} intent={((m.get('intent') or {}).get('type', '?'))}"
            for i, m in enumerate(monsters)
        )
        hand_text = "\n".join(
            f"[{i}] {c.get('name', '?')} cost={c.get('cost', '?')} dmg={c.get('damage', 0)} blk={c.get('block', 0)}"
            for i, c in enumerate(hand)
        )
        user_prompt = (
            f"玩家HP: {player.get('hp', '?')}/{player.get('max_hp', '?')} 能量:{combat.get('energy', '?')}\n"
            f"敌人信息:\n{monsters_text}\n\n"
            f"当前手牌:\n{hand_text}\n"
            "请给出本战斗的开局行动建议。"
        )
        fallback = {
            "threat_level": "medium",
            "priority_action": "mixed",
            "opening_card_sequence": [],
            "priority_target_index": None,
            "key_warning": "",
            "expected_rounds": "3+",
        }
        try:
            resp = self.llm.call(self.SYSTEM_PROMPT_COMBAT_OPENING, user_prompt, max_tokens=280)
            data = _parse_json_response(resp)
            seq = data.get("opening_card_sequence", [])
            if not isinstance(seq, list):
                seq = []
            seq = [int(x) for x in seq if isinstance(x, (int, float))]
            data["opening_card_sequence"] = seq[: self.combat_bias_steps]
            if self._last_advice is None:
                self._last_advice = LLMAdvice(
                    deck_route="mixed",
                    route_score=0.5,
                    key_synergies=[],
                    reward_shaping=0.0,
                    reasoning="combat_opening_only",
                )
            self._last_advice.combat_opening = data
            return data
        except Exception as e:
            print(f"[LLMAdvisor] evaluate_combat_opening 失败: {e}")
            if self._last_advice is None:
                self._last_advice = LLMAdvice(
                    deck_route="mixed",
                    route_score=0.5,
                    key_synergies=[],
                    reward_shaping=0.0,
                    reasoning="combat_opening_failed",
                )
            self._last_advice.combat_opening = fallback
            return fallback

    def _trim(self, text: str, limit: int = 80) -> str:
        value = str(text or "").replace("\n", " ").strip()
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)] + "…"

    def _format_enemies(self, enemies: List[Dict]) -> str:
        lines: List[str] = []
        for enemy in enemies:
            if not isinstance(enemy, dict) or not bool(enemy.get("is_alive", True)):
                continue
            intents = enemy.get("intents")
            if not isinstance(intents, list):
                intents = []
            intent_parts = []
            for intent in intents[:3]:
                if not isinstance(intent, dict):
                    continue
                intent_type = str(intent.get("intent_type", "?") or "?")
                total_damage = intent.get("total_damage")
                hits = intent.get("hits")
                if total_damage is not None:
                    piece = f"{intent_type}:{total_damage}"
                    if hits and int(hits or 0) > 1:
                        piece += f"x{int(hits)}"
                else:
                    piece = intent_type
                intent_parts.append(piece)
            powers = enemy.get("powers")
            if not isinstance(powers, list):
                powers = []
            power_text = ", ".join(
                f"{p.get('name', '?')}×{p.get('amount', '')}"
                for p in powers[:3]
                if isinstance(p, dict)
            )
            suffix = f" | Powers:{power_text}" if power_text else ""
            lines.append(
                f"[{enemy.get('index', 0)}] {enemy.get('name', '?')} "
                f"HP:{enemy.get('current_hp', enemy.get('hp', '?'))}/{enemy.get('max_hp', '?')} "
                f"格挡:{enemy.get('block', 0)} "
                f"意图:{' + '.join(intent_parts) if intent_parts else '未知'}{suffix}"
            )
        return "\n".join(lines) if lines else "无存活敌人"

    def _normalize_combat_turn_advice(
        self,
        data: Dict,
        hand_names: List[str],
        enemy_count: int,
        turn_number: int,
    ) -> Dict:
        if not isinstance(data, dict):
            data = {}
        allowed_names = {str(name).strip() for name in hand_names if str(name).strip()}
        play_order = data.get("play_order", [])
        clean_play_order = []
        if isinstance(play_order, list):
            for item in play_order:
                if not isinstance(item, dict):
                    continue
                card_name = str(item.get("card_name", "") or "").strip()
                if not card_name or card_name not in allowed_names:
                    continue
                try:
                    priority = int(item.get("priority", len(clean_play_order) + 1) or len(clean_play_order) + 1)
                except Exception:
                    priority = len(clean_play_order) + 1
                clean_play_order.append({
                    "card_name": card_name,
                    "priority": max(1, priority),
                    "role": str(item.get("role", "utility") or "utility"),
                    "reason": self._trim(str(item.get("reason", "") or ""), 20),
                })
        clean_play_order.sort(key=lambda x: (x["priority"], x["card_name"]))
        clean_play_order = clean_play_order[: self.combat_bias_steps + 3]

        def _clean_name_list(key: str) -> List[str]:
            value = data.get(key, [])
            if not isinstance(value, list):
                return []
            out = []
            seen = set()
            for item in value:
                name = str(item or "").strip()
                if not name or name not in allowed_names or name in seen:
                    continue
                seen.add(name)
                out.append(name)
            return out

        priority_target = data.get("priority_target")
        try:
            priority_target = int(priority_target) if priority_target is not None else None
        except Exception:
            priority_target = None
        if priority_target is not None and (priority_target < 0 or priority_target >= enemy_count):
            priority_target = None

        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0

        goal = str(data.get("goal", "mixed") or "mixed").strip().lower()
        if goal not in {"aggressive", "defensive", "buff", "debuff", "mixed"}:
            goal = "mixed"

        return {
            "turn": int(turn_number),
            "goal": goal,
            "play_order": clean_play_order,
            "key_combo_sequence": _clean_name_list("key_combo_sequence"),
            "avoid_cards": _clean_name_list("avoid_cards"),
            "priority_target": priority_target,
            "confidence": max(0.0, min(confidence, 1.0)),
            "reasoning": self._trim(str(data.get("reasoning", "") or ""), 50),
        }

    def evaluate_combat_turn(
        self,
        state: Dict,
        turn_number: int,
        max_turns: int = 3,
    ) -> Dict:
        fallback = {
            "turn": int(turn_number),
            "goal": "mixed",
            "play_order": [],
            "key_combo_sequence": [],
            "avoid_cards": [],
            "priority_target": None,
            "confidence": 0.0,
            "reasoning": "LLM 调用失败",
        }
        if int(turn_number or 0) <= 0 or int(turn_number) > int(max_turns):
            return dict(fallback)

        combat = state.get("combat") or {}
        player = combat.get("player") or {}
        hand = combat.get("hand") or []
        enemies = combat.get("enemies") or combat.get("monsters") or []
        run = state.get("run") or {}
        run_deck = run.get("deck") if isinstance(run.get("deck"), list) else state.get("deck", [])
        run_relics = run.get("relics") if isinstance(run.get("relics"), list) else state.get("relics", [])

        hand_names = [str(card.get("name", "") or "").strip() for card in hand if isinstance(card, dict)]
        hand_text = "\n".join(
            f"[{card.get('index', idx)}] {card.get('name', '?')}"
            f"{'+' if card.get('upgraded') else ''}"
            f" | 费用:{card.get('energy_cost', '?')}"
            f" | {'可出' if card.get('playable') else '不可出'}"
            f" | {card.get('card_type', card.get('type', '?'))}"
            for idx, card in enumerate(hand)
            if isinstance(card, dict)
        ) or "无手牌"
        powers = player.get("powers") if isinstance(player.get("powers"), list) else []
        powers_text = ", ".join(
            f"{p.get('name', '?')}×{p.get('amount', '')}"
            f"({'debuff' if p.get('is_debuff') else 'buff'})"
            for p in powers[:6]
            if isinstance(p, dict)
        ) or "无"
        deck_text = "\n".join(
            f"{card.get('name', '?')}{'+' if card.get('upgraded') else ''}"
            f" | {card.get('card_type', card.get('type', '?'))}"
            f" | 费用:{card.get('energy_cost', '?')}"
            for card in run_deck[:18]
            if isinstance(card, dict)
        ) or "无牌库信息"
        relic_text = "\n".join(
            f"{relic.get('name', '?')}: {self._trim(relic.get('description', ''), 50)}"
            for relic in run_relics[:12]
            if isinstance(relic, dict)
        ) or "无遗物"
        user_prompt = (
            f"=== 战斗状态（第{turn_number}回合）===\n"
            f"玩家HP:{player.get('current_hp', player.get('hp', '?'))}/{player.get('max_hp', '?')} "
            f"格挡:{player.get('block', 0)} 能量:{player.get('energy', combat.get('energy', '?'))}\n"
            f"玩家Powers:{powers_text}\n"
            f"战斗类型:{combat.get('enemy_type', state.get('enemy_type', 'monster'))}\n\n"
            f"=== 当前手牌 ===\n{hand_text}\n\n"
            f"=== 敌人信息 ===\n{self._format_enemies(enemies)}\n\n"
            f"=== 完整牌库 ===\n{deck_text}\n\n"
            f"=== 当前遗物 ===\n{relic_text}\n\n"
            f"请规划第{turn_number}回合的出牌顺序。"
        )
        try:
            resp = self.llm.call(self.SYSTEM_PROMPT_COMBAT_TURN, user_prompt, max_tokens=280)
            data = _parse_json_response(resp)
            normalized = self._normalize_combat_turn_advice(
                data=data,
                hand_names=hand_names,
                enemy_count=len([e for e in enemies if isinstance(e, dict) and bool(e.get("is_alive", True))]),
                turn_number=int(turn_number),
            )
            if self._last_advice is None:
                self._last_advice = LLMAdvice(
                    deck_route="mixed",
                    route_score=0.5,
                    key_synergies=[],
                    reward_shaping=0.0,
                    reasoning="combat_turn_only",
                )
            self._last_advice.combat_opening = normalized
            return normalized
        except Exception as e:
            print(f"[LLMAdvisor] evaluate_combat_turn 失败: {e}")
            if self._last_advice is None:
                self._last_advice = LLMAdvice(
                    deck_route="mixed",
                    route_score=0.5,
                    key_synergies=[],
                    reward_shaping=0.0,
                    reasoning="combat_turn_failed",
                )
            self._last_advice.combat_opening = dict(fallback)
            return dict(fallback)

    def _ensure_advice(self) -> LLMAdvice:
        if self._last_advice is None:
            self._last_advice = LLMAdvice(
                deck_route="mixed",
                route_score=0.5,
                key_synergies=[],
                reward_shaping=0.0,
                reasoning="scene_eval_only",
            )
        return self._last_advice

    def _state_run_context(self, state: Dict) -> Tuple[List[Dict], List[Dict], int, int, int, int]:
        run = state.get("run") or {}
        deck = run.get("deck") if isinstance(run.get("deck"), list) else state.get("deck", [])
        relics = run.get("relics") if isinstance(run.get("relics"), list) else state.get("relics", [])
        floor = int(run.get("floor", state.get("floor", 0)) or 0)
        gold = int(run.get("gold", state.get("gold", 0)) or 0)
        current_hp = int(run.get("current_hp", ((state.get("combat") or {}).get("player") or {}).get("current_hp", 0)) or 0)
        max_hp = int(run.get("max_hp", ((state.get("combat") or {}).get("player") or {}).get("max_hp", 0)) or 0)
        return deck, relics, floor, gold, current_hp, max_hp

    def _merge_card_candidates(self, reward_cards: List[Dict], selection_cards: List[Dict]) -> List[Dict]:
        merged: List[Dict] = []
        seen_keys = set()

        def _card_key(card: Dict, default_idx: int) -> Tuple:
            return (
                int(card.get("index", default_idx) or default_idx),
                str(card.get("card_id", "") or ""),
                str(card.get("name", "") or ""),
            )

        selection_by_key = {}
        for idx, card in enumerate(selection_cards):
            if isinstance(card, dict):
                selection_by_key[_card_key(card, idx)] = dict(card)

        primary = reward_cards if reward_cards else selection_cards
        for idx, card in enumerate(primary):
            if not isinstance(card, dict):
                continue
            key = _card_key(card, idx)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_card = dict(selection_by_key.get(key, {}))
            merged_card.update(card)
            if "index" not in merged_card:
                merged_card["index"] = idx
            merged.append(merged_card)

        for idx, card in enumerate(selection_cards):
            if not isinstance(card, dict):
                continue
            key = _card_key(card, idx)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            extra = dict(card)
            if "index" not in extra:
                extra["index"] = idx
            merged.append(extra)
        return merged

    def evaluate_event_choice(
        self,
        state: Dict,
        event_payload: Dict,
    ) -> Tuple[int, float, str]:
        event = event_payload or {}
        options = event.get("options") if isinstance(event.get("options"), list) else []
        valid = [opt for opt in options if isinstance(opt, dict) and not bool(opt.get("is_locked", False))]
        if not valid:
            return -99, 0.0, "无可选事件选项"
        if len(valid) == 1 and bool(valid[0].get("is_proceed", False)):
            return -99, 0.0, "仅剩推进选项"

        deck, relics, floor, gold, current_hp, max_hp = self._state_run_context(state)
        options_text = "\n".join(
            f"[{opt.get('index', idx)}] title={opt.get('title', '?')} "
            f"text_key={opt.get('text_key', '')} "
            f"proceed={bool(opt.get('is_proceed', False))} "
            f"will_kill={bool(opt.get('will_kill_player', False))} "
            f"description={self._trim(str(opt.get('description', '') or ''), 80)}"
            for idx, opt in enumerate(valid)
        )
        user_prompt = (
            f"楼层:{floor} HP:{current_hp}/{max_hp} 金币:{gold}\n"
            f"事件ID:{event.get('event_id', '')}\n"
            f"标题:{event.get('title', '')}\n"
            f"描述:{self._trim(str(event.get('description', '') or ''), 180)}\n\n"
            f"当前牌组摘要:\n{self._summarize_deck(deck)}\n\n"
            f"当前遗物:{[r.get('name', '?') for r in relics[:12] if isinstance(r, dict)]}\n\n"
            f"事件选项:\n{options_text}\n\n"
            "请只返回JSON: {\"recommended_index\": 选项索引, \"confidence\": 0到1, \"reasoning\": \"简短原因\"}"
        )
        system_prompt = (
            "你是杀戮尖塔2战士专家。只根据当前事件信息、当前构筑和生存状态，"
            "选择最优事件选项。若存在明显危险或会破坏构筑，要避免。只返回JSON。"
        )
        try:
            resp = self.llm.call(system_prompt, user_prompt, max_tokens=220)
            data = _parse_json_response(resp)
            idx = int(data.get("recommended_index", valid[0].get("index", 0)) or valid[0].get("index", 0))
            valid_indices = {int(opt.get("index", i) or i) for i, opt in enumerate(valid)}
            if idx not in valid_indices:
                idx = int(valid[0].get("index", 0) or 0)
            conf = max(0.0, min(float(data.get("confidence", 0.0) or 0.0), 1.0))
            reason = self._trim(str(data.get("reasoning", "") or ""), 80)
            advice = self._ensure_advice()
            advice.event_recommendation = idx
            advice.event_confidence = conf
            return idx, conf, reason
        except Exception as e:
            print(f"[LLMAdvisor] evaluate_event_choice 失败: {e}")
            advice = self._ensure_advice()
            advice.event_recommendation = -99
            advice.event_confidence = 0.0
            return -99, 0.0, f"LLM 调用失败: {e}"

    def evaluate_card_selection(
        self,
        state: Dict,
        payload: Optional[Dict] = None,
    ) -> Tuple[str, int, float, str]:
        reward = state.get("reward") or {}
        selection = state.get("selection") or {}
        reward_cards = reward.get("card_options") if isinstance(reward.get("card_options"), list) else []
        selection_cards = selection.get("cards") if isinstance(selection.get("cards"), list) else []
        candidates = self._merge_card_candidates(reward_cards, selection_cards)
        if not candidates:
            return "other", -99, 0.0, "无可选卡牌"

        is_reward_card = bool(reward.get("pending_card_choice", False)) and bool(reward_cards) and "choose_reward_card" in [str(a) for a in (state.get("legal_actions") or [])]
        scene_hint = "reward_card" if is_reward_card else "unknown"
        deck, relics, floor, gold, current_hp, max_hp = self._state_run_context(state)
        candidate_context = self._retrieve_card_context(candidates, deck)
        synergy_context = self._retrieve_synergies(candidates, deck)
        selection_prompt = self._trim(str(selection.get("prompt", "") or ""), 120)
        cards_text = "\n".join(
            f"[{c.get('index', idx)}] {c.get('name', '?')}"
            f"{'+' if c.get('upgraded') else ''} | type={c.get('card_type', c.get('type', '?'))}"
            f" | rarity={c.get('rarity', '?')} | cost={c.get('energy_cost', '?')}"
            f" | text={self._trim(str(c.get('rules_text', '')), 80)}"
            for idx, c in enumerate(candidates)
            if isinstance(c, dict)
        )
        user_prompt = (
            f"楼层:{floor} HP:{current_hp}/{max_hp} 金币:{gold}\n"
            f"screen:{state.get('screen', '')}\n"
            f"selection.kind:{selection.get('kind', '')}\n"
            f"selection.prompt(可能不可靠):{selection_prompt}\n"
            f"reward.pending_card_choice:{bool(reward.get('pending_card_choice', False))}\n"
            f"legal_actions:{state.get('legal_actions', [])}\n\n"
            f"当前牌组摘要:\n{self._summarize_deck(deck)}\n\n"
            f"当前遗物:{[r.get('name', '?') for r in relics[:12] if isinstance(r, dict)]}\n\n"
            f"候选卡牌:\n{cards_text}\n\n"
            f"知识库卡牌信息:\n{candidate_context}\n\n"
            f"知识库协同信息:\n{synergy_context}\n\n"
            "请先判断当前属于哪类牌处理场景，再给出最优索引。"
            "只返回JSON: {\"scene_type\":\"reward_card|remove|upgrade|transform|enchant|other\","
            "\"recommended_index\":索引,\"confidence\":0到1,\"reasoning\":\"简短原因\"}"
        )
        system_prompt = (
            "你是杀戮尖塔2战士构筑专家。你需要根据当前run构筑、候选牌和场景信息，"
            "判断这是奖励卡、删牌、升级、变形、附魔还是其他牌处理场景，并给出当前最优选择。"
            "如果 reward.pending_card_choice=true，则优先视为奖励卡场景。只返回JSON。"
        )
        try:
            resp = self.llm.call(system_prompt, user_prompt, max_tokens=320)
            data = _parse_json_response(resp)
            scene_type = str(data.get("scene_type", scene_hint or "other") or "other").strip().lower()
            if is_reward_card:
                scene_type = "reward_card"
            if scene_type not in {"reward_card", "remove", "upgrade", "transform", "enchant", "other"}:
                scene_type = "other"
            idx = int(data.get("recommended_index", candidates[0].get("index", 0)) or candidates[0].get("index", 0))
            valid_indices = {int(card.get("index", i) or i) for i, card in enumerate(candidates) if isinstance(card, dict)}
            if idx not in valid_indices:
                idx = int(candidates[0].get("index", 0) or 0)
            conf = max(0.0, min(float(data.get("confidence", 0.0) or 0.0), 1.0))
            reason = self._trim(str(data.get("reasoning", "") or ""), 100)
            advice = self._ensure_advice()
            advice.card_recommendation = idx
            advice.card_confidence = conf
            advice.card_scene_type = scene_type
            return scene_type, idx, conf, reason
        except Exception as e:
            print(f"[LLMAdvisor] evaluate_card_selection 失败: {e}")
            advice = self._ensure_advice()
            advice.card_recommendation = -99
            advice.card_confidence = 0.0
            advice.card_scene_type = "other"
            return "other", -99, 0.0, f"LLM 调用失败: {e}"

    def evaluate_card_remove(
        self,
        state: Dict,
        removable_cards: List[Dict],
    ) -> Tuple[int, float, str]:
        """评估删牌选项，返回 (recommended_index, confidence, reasoning)"""
        if not removable_cards:
            return -99, 0.0, "无可删牌"

        deck = state.get("deck", [])
        floor = state.get("floor", 0)
        selection = state.get("selection") or {}
        prompt_text = str(selection.get("prompt", "") or "")
        options_text = "\n".join(
            f"[{i}] {c.get('name', '?')} | type={c.get('card_type', c.get('type', '?'))} | rarity={c.get('rarity', '?')} | upgraded={bool(c.get('upgraded', False))}"
            for i, c in enumerate(removable_cards)
        )
        user_prompt = (
            f"楼层: {floor}\n"
            f"删牌提示: {prompt_text}\n"
            f"当前牌组摘要:\n{self._summarize_deck(deck)}\n\n"
            f"可选删牌:\n{options_text}\n"
            "请给出最优删牌索引。"
        )

        try:
            resp = self.llm.call(self.SYSTEM_PROMPT_REMOVE_EVAL, user_prompt, max_tokens=260)
            data = _parse_json_response(resp)
            idx = int(data.get("recommended_index", 0))
            conf = float(data.get("confidence", 0.5))
            reason = str(data.get("reasoning", ""))

            if idx < 0:
                idx = 0
            if idx >= len(removable_cards):
                idx = len(removable_cards) - 1

            if self._last_advice is None:
                self._last_advice = LLMAdvice(
                    deck_route="mixed",
                    route_score=0.5,
                    key_synergies=[],
                    reward_shaping=0.0,
                    reasoning="remove_eval_only",
                )
            self._last_advice.remove_recommendation = idx
            self._last_advice.remove_confidence = conf
            return idx, conf, reason
        except Exception as e:
            print(f"[LLMAdvisor] evaluate_card_remove 失败: {e}")
            if self._last_advice is not None:
                self._last_advice.remove_recommendation = -99
                self._last_advice.remove_confidence = 0.0
            return -99, 0.0, f"LLM 调用失败: {e}"

    def evaluate_shop_purchase(
        self,
        state: Dict,
        shop_items: Dict,
    ) -> Tuple[str, Optional[int], float, str]:
        """评估商店购买动作，返回 (action, option_index, confidence, reasoning)"""
        if not isinstance(shop_items, dict) or not shop_items:
            return "NONE", None, 0.0, "无商店选项"

        gold = int(state.get("gold", 0) or 0)
        floor = state.get("floor", 0)
        cards = shop_items.get("cards") if isinstance(shop_items.get("cards"), list) else []
        relics = shop_items.get("relics") if isinstance(shop_items.get("relics"), list) else []
        potions = shop_items.get("potions") if isinstance(shop_items.get("potions"), list) else []
        can_remove = bool(shop_items.get("can_remove_card", False))
        remove_cost = shop_items.get("remove_cost", None)

        cards_text = "\n".join(
            f"[{i}] {c.get('name', '?')} | price={c.get('price', '?')} | rarity={c.get('rarity', '?')}"
            for i, c in enumerate(cards)
        ) or "（无）"
        relics_text = "\n".join(
            f"[{i}] {r.get('name', '?')} | price={r.get('price', '?')}"
            for i, r in enumerate(relics)
        ) or "（无）"
        potions_text = "\n".join(
            f"[{i}] {p.get('name', '?')} | price={p.get('price', '?')}"
            for i, p in enumerate(potions)
        ) or "（无）"

        user_prompt = (
            f"楼层: {floor} | 金币: {gold}\n"
            f"当前牌组摘要:\n{self._summarize_deck(state.get('deck', []))}\n\n"
            f"商店卡牌:\n{cards_text}\n\n"
            f"商店遗物:\n{relics_text}\n\n"
            f"商店药水:\n{potions_text}\n\n"
            f"可否删牌: {can_remove} | 删牌价格: {remove_cost}\n"
            "请给出本次最优购买动作与索引。"
        )

        fallback_action = "NONE"
        fallback_index: Optional[int] = None

        try:
            resp = self.llm.call(self.SYSTEM_PROMPT_SHOP_EVAL, user_prompt, max_tokens=320)
            data = _parse_json_response(resp)
            action = str(data.get("recommended_action", "NONE") or "NONE")
            raw_idx = data.get("option_index", None)
            conf = float(data.get("confidence", 0.5))
            reason = str(data.get("reasoning", ""))

            allowed_actions = {"buy_card", "buy_relic", "buy_potion", "remove_card_at_shop", "proceed", "NONE"}
            if action not in allowed_actions:
                action = "NONE"

            idx: Optional[int] = None
            if raw_idx is not None:
                try:
                    idx = int(raw_idx)
                except Exception:
                    idx = None

            if action == "buy_card" and idx is not None:
                if idx < 0 or idx >= len(cards):
                    idx = None
            elif action == "buy_relic" and idx is not None:
                if idx < 0 or idx >= len(relics):
                    idx = None
            elif action == "buy_potion" and idx is not None:
                if idx < 0 or idx >= len(potions):
                    idx = None
            else:
                idx = None

            if self._last_advice is None:
                self._last_advice = LLMAdvice(
                    deck_route="mixed",
                    route_score=0.5,
                    key_synergies=[],
                    reward_shaping=0.0,
                    reasoning="shop_eval_only",
                )
            self._last_advice.shop_recommendation_action = action
            self._last_advice.shop_recommendation_index = idx
            self._last_advice.shop_confidence = conf
            return action, idx, conf, reason
        except Exception as e:
            print(f"[LLMAdvisor] evaluate_shop_purchase 失败: {e}")
            if self._last_advice is not None:
                self._last_advice.shop_recommendation_action = "NONE"
                self._last_advice.shop_recommendation_index = None
                self._last_advice.shop_confidence = 0.0
            return fallback_action, fallback_index, 0.0, f"LLM 调用失败: {e}"

    def get_last_event_recommendation(self) -> Tuple[int, float]:
        if self._last_advice is None or self._last_advice.event_recommendation == -99:
            return -99, 0.0
        return self._last_advice.event_recommendation, self._last_advice.event_confidence

    def invalidate_event_recommendation(self):
        if self._last_advice is not None:
            self._last_advice.event_recommendation = -99
            self._last_advice.event_confidence = 0.0

    def get_reward_shaping_bonus(self, state: Dict) -> float:
        """全局路线质量的奖励塑形分 — 供 RewardShaper 在非 CARD_REWARD 时用"""
        advice = self.get_advice(state)
        if advice is None:
            return 0.0
        return advice.reward_shaping * 0.5

    def get_last_card_recommendation(self) -> Tuple[int, float]:
        """供 RewardShaper 读取上次选牌推荐的 (index, confidence)"""
        if self._last_advice is None or self._last_advice.card_recommendation == -99:
            return -99, 0.0
        return self._last_advice.card_recommendation, self._last_advice.card_confidence

    def get_last_card_scene_type(self) -> str:
        if self._last_advice is None:
            return ""
        return str(self._last_advice.card_scene_type or "")

    def invalidate_card_recommendation(self):
        """选牌动作执行后调用，重置缓存避免重复塑形"""
        if self._last_advice is not None:
            self._last_advice.card_recommendation = -99
            self._last_advice.card_confidence = 0.0
            self._last_advice.card_scene_type = ""

    def get_last_relic_recommendation(self) -> Tuple[int, float]:
        if self._last_advice is None or self._last_advice.relic_recommendation == -99:
            return -99, 0.0
        return self._last_advice.relic_recommendation, self._last_advice.relic_confidence

    def invalidate_relic_recommendation(self):
        if self._last_advice is not None:
            self._last_advice.relic_recommendation = -99
            self._last_advice.relic_confidence = 0.0

    def get_last_map_recommendation(self) -> Tuple[int, float, List[float]]:
        if self._last_advice is None or self._last_advice.map_recommendation == -99:
            return -99, 0.0, []
        return (
            self._last_advice.map_recommendation,
            self._last_advice.map_confidence,
            list(self._last_advice.map_route_scores),
        )

    def invalidate_map_recommendation(self):
        if self._last_advice is not None:
            self._last_advice.map_recommendation = -99
            self._last_advice.map_confidence = 0.0
            self._last_advice.map_route_scores = []

    def get_last_remove_recommendation(self) -> Tuple[int, float]:
        if self._last_advice is None or self._last_advice.remove_recommendation == -99:
            return -99, 0.0
        return self._last_advice.remove_recommendation, self._last_advice.remove_confidence

    def invalidate_remove_recommendation(self):
        if self._last_advice is not None:
            self._last_advice.remove_recommendation = -99
            self._last_advice.remove_confidence = 0.0

    def get_last_shop_recommendation(self) -> Tuple[str, Optional[int], float]:
        if self._last_advice is None:
            return "NONE", None, 0.0
        return (
            self._last_advice.shop_recommendation_action,
            self._last_advice.shop_recommendation_index,
            self._last_advice.shop_confidence,
        )

    def invalidate_shop_recommendation(self):
        if self._last_advice is not None:
            self._last_advice.shop_recommendation_action = "NONE"
            self._last_advice.shop_recommendation_index = None
            self._last_advice.shop_confidence = 0.0

    def get_last_combat_opening(self) -> Dict:
        if self._last_advice is None:
            return {}
        return dict(self._last_advice.combat_opening or {})

    def invalidate_combat_opening(self):
        if self._last_advice is not None:
            self._last_advice.combat_opening = {}

    # ── 内部：全局路线评估 ──────────────────────────────────────────────────

    def _query_llm_global(self, state: Dict) -> LLMAdvice:
        deck = state.get("deck", [])
        relics = state.get("relics", [])
        combat = state.get("combat", {})
        character = state.get("character", {}).get("name", "Unknown")
        floor = state.get("floor", 0)

        # 用当前牌组里高频牌做 Codex 检索（给全局评估用）
        deck_card_ctx = self._retrieve_deck_card_context(deck, max_cards=6)
        synergy_ctx = self._retrieve_synergies([], deck, deck_only=True)
        strategy_ctx = self._retrieve_strategy(character)

        user_prompt = (
            f"=== 当前游戏状态 ===\n"
            f"角色: {character} | 楼层: {floor}\n"
            f"HP: {combat.get('player', {}).get('hp', '?')}"
            f"/{combat.get('player', {}).get('max_hp', '?')}\n"
            f"金币: {state.get('gold', 0)}\n"
            f"遗物: {[r.get('name', '?') for r in relics[:12]]}\n\n"
            f"=== 当前牌组 ===\n{self._summarize_deck(deck)}\n\n"
            f"=== 牌组关键牌 Codex 数据 ===\n{deck_card_ctx}\n\n"
            f"=== 已知协同 ===\n{synergy_ctx}\n\n"
            f"=== 流派参考 ===\n{strategy_ctx}\n\n"
            f"请评估当前牌组路线质量，返回 JSON。"
        )

        resp_text = self.llm.call(self.SYSTEM_PROMPT_GLOBAL, user_prompt, max_tokens=512)
        data = _parse_json_response(resp_text)

        return LLMAdvice(
            deck_route=data.get("deck_route", "mixed"),
            route_score=float(data.get("route_score", 0.5)),
            key_synergies=data.get("key_synergies", []),
            reward_shaping=float(data.get("reward_shaping", 0.0)),
            reasoning=data.get("reasoning", ""),
        )

    # ── 内部：Codex 检索 ────────────────────────────────────────────────────

    def _retrieve_card_context(
        self,
        reward_cards: List[Dict],
        deck: List[Dict],
        max_deck_sample: int = 6,
    ) -> str:
        """
        Layer-1 Codex 检索：
        - 对每张候选奖励卡，从 knowledge_base["cards"] 取完整 Codex 数据
        - 从牌组里随机采样若干牌的 Codex 数据作为"牌组上下文"
        返回格式化的字符串供 prompt 使用
        """
        cards_db = self.knowledge_base.get("cards", {})
        lines = ["【候选奖励卡（Codex 数据）】"]

        for i, card in enumerate(reward_cards):
            card_name = card.get("name", "") or card.get("id", "")
            codex = self._lookup_card(card_name, cards_db)
            if codex:
                lines.append(
                    f"  [{i}] {codex.get('name', card_name)}"
                    f" | 类型:{codex.get('type','?')}"
                    f" | 费用:{codex.get('cost','?')}"
                    f" | 稀有:{codex.get('rarity','?')}"
                    f" | 效果: {self._trim(codex.get('description', ''), 80)}"
                )
            else:
                # Codex 未命中，用 Mod 状态里的信息兜底
                lines.append(
                    f"  [{i}] {card_name}"
                    f" | 类型:{card.get('type','?')}"
                    f" | 费用:{card.get('cost','?')}"
                    f" | 效果: {self._trim(card.get('description', ''), 80)}"
                )

        # 牌组采样
        if deck:
            import random
            sample = deck[:max_deck_sample]  # 取前 N 张（已按顺序排列）
            lines.append("\n【牌组采样（Codex 数据，供对比参考）】")
            for card in sample:
                card_name = card.get("name", "") or card.get("id", "")
                codex = self._lookup_card(card_name, cards_db)
                upg = "+" if card.get("upgraded") else ""
                if codex:
                    lines.append(
                        f"  {codex.get('name', card_name)}{upg}"
                        f" | {codex.get('type','?')}"
                        f" | 费用:{codex.get('cost','?')}"
                        f" | {self._trim(codex.get('description', ''), 60)}"
                    )
                else:
                    lines.append(f"  {card_name}{upg} | {card.get('type','?')}")

        return "\n".join(lines)

    def _retrieve_deck_card_context(self, deck: List[Dict], max_cards: int = 6) -> str:
        """全局评估用：取牌组前 N 张的 Codex 数据"""
        if not deck:
            return "（牌组为空）"
        cards_db = self.knowledge_base.get("cards", {})
        lines = []
        for card in deck[:max_cards]:
            card_name = card.get("name", "") or card.get("id", "")
            codex = self._lookup_card(card_name, cards_db)
            upg = "+" if card.get("upgraded") else ""
            if codex:
                lines.append(
                    f"  {codex.get('name', card_name)}{upg}"
                    f" | {codex.get('type','?')} | 费用:{codex.get('cost','?')}"
                    f" | {self._trim(codex.get('description', ''), 60)}"
                )
            else:
                lines.append(f"  {card_name}{upg}")
        return "\n".join(lines) if lines else "（Codex 无匹配）"

    def _lookup_card(self, name_or_id: str, cards_db: Dict) -> Optional[Dict]:
        """
        从 Codex cards 字典里匹配一张卡。
        先精确匹配 id，再精确匹配 name，最后做不区分大小写的 name 模糊匹配。
        """
        if not name_or_id or not cards_db:
            return None
        # 1. id 精确
        if name_or_id in cards_db:
            return cards_db[name_or_id]
        # 2. name 精确
        for v in cards_db.values():
            if v.get("name", "") == name_or_id:
                return v
        # 3. 忽略大小写 + 去掉升级符号
        name_lower = name_or_id.lower().rstrip("+")
        for v in cards_db.values():
            if v.get("name", "").lower() == name_lower:
                return v
        return None

    def _retrieve_synergies(
        self,
        reward_cards: List[Dict],
        deck: List[Dict],
        deck_only: bool = False,
    ) -> str:
        """
        Layer-2：从 knowledge_base["synergies"] 里找与候选牌 / 牌组有交集的条目。
        deck_only=True 时只检索牌组内部的协同。
        """
        synergies = self.knowledge_base.get("synergies", [])
        if not synergies:
            return "（无协同数据）"

        candidate_names = set(
            (c.get("name", "") or c.get("id", "")).lower()
            for c in (reward_cards if not deck_only else [])
        )
        deck_names = set(
            (c.get("name", "") or c.get("id", "")).lower()
            for c in deck
        )
        all_names = candidate_names | deck_names

        matched = []
        for entry in synergies:
            entry_cards = {s.lower() for s in entry.get("cards", [])}
            # 只要有 2 张以上交集就算命中
            if len(entry_cards & all_names) >= 2:
                matched.append(entry.get("description", ""))
            if len(matched) >= 3:
                break

        return "\n".join(f"  • {m}" for m in matched) if matched else "（无匹配协同）"

    def _retrieve_strategy(self, character: str) -> str:
        """Layer-3：流派攻略段落（角色匹配 + all）"""
        strategies = self.knowledge_base.get("strategies", [])
        if not strategies:
            return "（无攻略数据）"
        char_lower = character.lower()
        matched = []
        for entry in strategies:
            ec = entry.get("character", "").lower()
            if ec == "all" or ec == char_lower:
                matched.append(entry.get("text", ""))
            if len(matched) >= 3:
                break
        return "\n".join(f"  {m}" for m in matched) if matched else "（无匹配攻略）"

    # ── 内部：格式化工具 ────────────────────────────────────────────────────

    def _summarize_deck(self, deck: List[Dict]) -> str:
        if not deck:
            return "  （空牌组）"
        by_type: Dict[str, List[str]] = {}
        for card in deck:
            t = card.get("type", "OTHER")
            name = card.get("name", "?")
            if card.get("upgraded"):
                name += "+"
            by_type.setdefault(t, []).append(name)
        lines = [f"  {t}({len(ns)}张): {', '.join(ns)}" for t, ns in by_type.items()]
        lines.insert(0, f"  总计 {len(deck)} 张")
        return "\n".join(lines)

    @staticmethod
    def _trim(text: str, max_len: int) -> str:
        if not text:
            return ""
        text = text.replace("\n", " ").strip()
        return text[:max_len] + "…" if len(text) > max_len else text

    def _load_knowledge_base(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[LLMAdvisor] 知识库未找到: {path}，使用空知识库")
            return {"cards": {}, "relics": {}, "strategies": [], "synergies": []}
        except json.JSONDecodeError as e:
            print(f"[LLMAdvisor] 知识库 JSON 解析失败: {e}，使用空知识库")
            return {"cards": {}, "relics": {}, "strategies": [], "synergies": []}
