"""
src/llm/knowledge_builder.py

知识库构建器
从以下来源收集知识:
1. Spire Codex API (localhost:8000) - 卡牌/遗物/怪物数据
2. 硬编码的核心攻略知识（可扩展）
3. （可选）爬取攻略网站

输出: data/knowledge_base.json

卡牌条目字段名与 STS2AIAgent 状态结构对齐（见 docs/STS2AIAgent-API-中文调用文档.md）：
英文键名；name / description / flavor 等文本通过 Codex ?lang=zhs 采为简体中文。
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Union
import requests


class KnowledgeBuilder:
    """
    构建 LLM Advisor 使用的知识库

    知识库结构:
    {
        "meta": { "lang", "spire_codex_url", "built_at", "schema_note" },
        "built_at": "...",  # 与 meta.built_at 相同，兼容旧逻辑
        "cards": { card_id: {  # 键为游戏 id，对齐 Mod hand[] 常用字段
            "id", "name", "type", "cost", "star_cost", "description",
            "target_type", "keywords", "is_upgraded",
            ... 扩展字段 rarity, color, damage, block, ...
        }},
        "relics": { relic_id: { "id", "name", "description", "flavor", "rarity", "pool", ... } },
        "monsters": { monster_id: { "id", "name", "type", "min_hp", "moves", ... } },
        "strategies": [ { "character", "route", "text" }, ... ],
        "synergies": [ { "cards", "description" }, ... ]
    }
    """

    DEFAULT_SPIRE_CODEX_URL = "http://localhost:8000"
    DEFAULT_CODEX_LANG = "zhs"  # Spire Codex 简体中文

    def __init__(
        self,
        output_path: str = "data/knowledge_base.json",
        spire_codex_url: Optional[str] = None,
        codex_lang: str = DEFAULT_CODEX_LANG,
    ):
        self.output_path = output_path
        self.spire_codex_url = (
            spire_codex_url
            or os.environ.get("SPIRE_CODEX_URL")
            or self.DEFAULT_SPIRE_CODEX_URL
        ).rstrip("/")
        self.codex_lang = codex_lang
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def build(self, use_spire_codex: bool = True) -> Dict:
        """构建完整知识库"""
        built_at = time.strftime("%Y-%m-%d %H:%M:%S")
        knowledge: Dict[str, Any] = {
            "meta": {
                "lang": self.codex_lang,
                "spire_codex_url": self.spire_codex_url,
                "built_at": built_at,
                "schema_note": (
                    "cards[*] keys align with STS2AIAgent combat hand card fields "
                    "(id, name, type, cost, star_cost, description, target_type, keywords); "
                    "localized strings follow meta.lang."
                ),
            },
            "built_at": built_at,
            "cards": {},
            "relics": {},
            "monsters": {},
            "strategies": [],
            "synergies": [],
        }

        if use_spire_codex:
            print(
                f"📦 从 Spire Codex 加载游戏数据 (lang={self.codex_lang}) "
                f"{self.spire_codex_url} ..."
            )
            knowledge["cards"] = self._fetch_cards()
            knowledge["relics"] = self._fetch_relics()
            knowledge["monsters"] = self._fetch_monsters()
        else:
            print("⚠️  Spire Codex 未运行，使用内置基础数据")

        print("📚 加载内置攻略知识...")
        knowledge["strategies"] = self._get_builtin_strategies()
        knowledge["synergies"] = self._get_builtin_synergies()

        # 保存
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)

        print(f"✅ 知识库已保存: {self.output_path}")
        print(f"   卡牌: {len(knowledge['cards'])} | 遗物: {len(knowledge['relics'])} "
              f"| 怪物: {len(knowledge['monsters'])} | 攻略段落: {len(knowledge['strategies'])}")

        return knowledge

    # ──────────────────────────────────────────────────────────────────────────
    # Spire Codex API 拉取
    # ──────────────────────────────────────────────────────────────────────────

    def _codex_params(self) -> Dict[str, str]:
        return {"lang": self.codex_lang}

    @staticmethod
    def _cost_to_mod_string(cost: Any) -> Optional[str]:
        """Mod 手牌里 cost 为字符串；与 Raw API 返回一致便于对齐。"""
        if cost is None:
            return None
        return str(cost)

    @staticmethod
    def _normalize_keywords(
        raw: Optional[Union[List[str], List[Dict[str, Any]]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Mod: keywords 为 { "name", "description" }[]；
        Codex 常为字符串列表，转为同名结构，description 缺省为 null。
        """
        if raw is None:
            return None
        if not isinstance(raw, list):
            return None
        out: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, str):
                out.append({"name": item, "description": None})
            elif isinstance(item, dict):
                name = item.get("name") or item.get("id") or ""
                out.append(
                    {
                        "name": name,
                        "description": item.get("description"),
                    }
                )
        return out

    def _normalize_card(self, c: Dict[str, Any]) -> Dict[str, Any]:
        """英文键名，与 STS2AIAgent GET 状态中手牌字段对齐（扩展字段保留）。"""
        cid = c.get("id")
        return {
            "id": cid,
            "name": c.get("name"),
            "type": c.get("type"),
            "cost": self._cost_to_mod_string(c.get("cost")),
            "star_cost": c.get("star_cost"),
            "description": c.get("description"),
            "target_type": c.get("target"),
            "keywords": self._normalize_keywords(c.get("keywords")),
            "is_upgraded": False,
            "rarity": c.get("rarity"),
            "color": c.get("color"),
            "damage": c.get("damage"),
            "block": c.get("block"),
            "hit_count": c.get("hit_count"),
            "powers_applied": c.get("powers_applied"),
            "cards_draw": c.get("cards_draw"),
            "energy_gain": c.get("energy_gain"),
            "hp_loss": c.get("hp_loss"),
            "upgrade": c.get("upgrade"),
            "description_raw": c.get("description_raw"),
            "image_url": c.get("image_url"),
        }

    def _normalize_relic(self, r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": r.get("id"),
            "name": r.get("name"),
            "description": r.get("description"),
            "flavor": r.get("flavor"),
            "rarity": r.get("rarity"),
            "pool": r.get("pool"),
            "description_raw": r.get("description_raw"),
            "image_url": r.get("image_url"),
        }

    def _normalize_monster(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """与文档中敌人 name / 意图等可读字段一致；数值键保持英文。"""
        return {
            "id": m.get("id"),
            "name": m.get("name"),
            "type": m.get("type"),
            "min_hp": m.get("min_hp"),
            "max_hp": m.get("max_hp"),
            "min_hp_ascension": m.get("min_hp_ascension"),
            "max_hp_ascension": m.get("max_hp_ascension"),
            "moves": m.get("moves"),
            "damage_values": m.get("damage_values"),
            "block_values": m.get("block_values"),
            "image_url": m.get("image_url"),
        }

    def _fetch_cards(self) -> Dict[str, Dict[str, Any]]:
        try:
            resp = requests.get(
                f"{self.spire_codex_url}/api/cards",
                params=self._codex_params(),
                timeout=120,
            )
            resp.raise_for_status()
            cards_list = resp.json()
            out: Dict[str, Dict[str, Any]] = {}
            for c in cards_list:
                if not isinstance(c, dict) or "id" not in c:
                    continue
                out[c["id"]] = self._normalize_card(c)
            return out
        except Exception as e:
            print(f"  ⚠️  获取卡牌失败: {e}")
            return {}

    def _fetch_relics(self) -> Dict[str, Dict[str, Any]]:
        try:
            resp = requests.get(
                f"{self.spire_codex_url}/api/relics",
                params=self._codex_params(),
                timeout=120,
            )
            resp.raise_for_status()
            relics_list = resp.json()
            out: Dict[str, Dict[str, Any]] = {}
            for r in relics_list:
                if not isinstance(r, dict) or "id" not in r:
                    continue
                out[r["id"]] = self._normalize_relic(r)
            return out
        except Exception as e:
            print(f"  ⚠️  获取遗物失败: {e}")
            return {}

    def _fetch_monsters(self) -> Dict[str, Dict[str, Any]]:
        try:
            resp = requests.get(
                f"{self.spire_codex_url}/api/monsters",
                params=self._codex_params(),
                timeout=120,
            )
            resp.raise_for_status()
            monsters_list = resp.json()
            out: Dict[str, Dict[str, Any]] = {}
            for m in monsters_list:
                if not isinstance(m, dict) or "id" not in m:
                    continue
                out[m["id"]] = self._normalize_monster(m)
            return out
        except Exception as e:
            print(f"  ⚠️  获取怪物失败: {e}")
            return {}

    # ──────────────────────────────────────────────────────────────────────────
    # 内置攻略知识（核心策略文本，基于 STS1/STS2 社区攻略提炼）
    # ──────────────────────────────────────────────────────────────────────────

    def _get_builtin_strategies(self) -> List[Dict]:
        """
        内置核心攻略策略（基于社区经验总结）
        可以继续添加更多段落
        """
        return [
            # ── 通用策略 ──────────────────────────────────────────────────────
            {
                "character": "all",
                "route": "general",
                "text": (
                    "【卡组规模控制】理想卡组大小为12-18张。过多的卡会稀释关键牌，"
                    "导致需要的牌抽不到。除非有特殊遗物或无限循环流，否则谨慎选牌。"
                    "每次奖励时优先考虑：这张牌是否明显优于已有的牌？"
                ),
            },
            {
                "character": "all",
                "route": "general",
                "text": (
                    "【能量管理】每回合3能量是标准。出牌顺序很重要：先打格挡牌确保生存，"
                    "再打输出牌。带有'每次出牌'触发效果的卡应最后打，以触发前序牌效果。"
                ),
            },
            {
                "character": "all",
                "route": "defense",
                "text": (
                    "【防守策略】格挡流核心：堆叠大量格挡，消耗敌人每回合提升的攻击。"
                    "关键遗物：护符（回合开始格挡），角质鳞片（受伤获得格挡）。"
                    "关键牌：完美防御类、格挡翻倍类。注意不同角色的格挡缩放机制不同。"
                ),
            },
            {
                "character": "all",
                "route": "aggro",
                "text": (
                    "【爆发流】快速消灭敌人，减少受到伤害的回合数。"
                    "第1层精英战避开，优先打普通战获取金币买关键牌。"
                    "BOSS前必须确保有足够输出消灭精英怪，BOSS通常需要400-600总伤害。"
                ),
            },
            {
                "character": "all",
                "route": "general",
                "text": (
                    "【地图路线选择】火焰图标=精英（高风险高奖励），问号=事件（可能很好也可能很差），"
                    "营地=休息（恢复HP或强化牌），头骨=商店（买卡/遗物/删牌）。"
                    "HP低于50%时优先找营地休息。遗物短缺时优先打精英。"
                ),
            },
            {
                "character": "all",
                "route": "general",
                "text": (
                    "【删牌策略】初始牌组有大量弱牌（打击、防御）。"
                    "在商店或事件中及时删除：双重打击、防御、震击等低效牌。"
                    "瘦身牌组（10张以内）能大幅提升关键牌频率。"
                    "特别是无限循环流必须删到极致。"
                ),
            },
            {
                "character": "all",
                "route": "relic_synergy",
                "text": (
                    "【遗物评估】遗物的价值通常高于卡牌，因为遗物每场战斗都生效。"
                    "顶级遗物：不稳定炸弹（免费出牌），哲人石（开局3能量），"
                    "黑星（精英掉落变两个），冰霜之眼（进入战斗抽1牌）。"
                    "评估遗物时考虑与当前卡组的协同，1+1>2的协同往往决定run的成败。"
                ),
            },
            # ── 角色特定策略（STS2新角色，待补充）──────────────────────────
            {
                "character": "ironclad",
                "route": "strength",
                "text": (
                    "【力量流】通过力量遗物（燃烧的血，战斗嘶吼+等）堆叠力量属性，"
                    "每点力量增加所有攻击伤害。配合多段攻击卡（双重猛击+，旋风斩+）伤害爆炸。"
                    "关键牌：战吼，信念；关键遗物：胸口纹章，战神的鲜血。"
                ),
            },
            {
                "character": "silent",
                "route": "poison",
                "text": (
                    "【中毒流】不直接输出，通过毒素累积消耗敌人HP。"
                    "核心机制：毒素叠加，每回合结束毒素值等于已叠加毒素造成伤害后-1。"
                    "关键牌：毒云，催化剂+（毒素翻三倍），毒刺；关键遗物：毒蛇骷髅，杀手印记。"
                    "注意：毒素在buff清除时会被清掉，需应对 Artifact 怪物。"
                ),
            },
            {
                "character": "defect",
                "route": "orbs",
                "text": (
                    "【法球流】宝球管理是机器人的核心机制。"
                    "闪电球：触发/蒸发时造成伤害；冰霜球：触发时获得格挡；暗物质球：触发时造成诅咒。"
                    "关键遗物：奇点（额外法球槽），输出倍频器（法球效果翻倍）。"
                    "关键牌：双铸，天文台（多宝球触发），移频器。"
                ),
            },
            {
                "character": "watcher",
                "route": "infinite",
                "text": (
                    "【无限流/姿态流】观察者通过冷静/愤怒姿态切换获得奖励。"
                    "进入愤怒姿态：所有攻击造成双倍伤害；离开时：获得3格挡/造成追加伤害。"
                    "无限循环：用空手道/满月斩创建每回合可以无限出牌的循环。"
                    "关键牌：顿悟（冷静+），静坐，满月斩+；关键遗物：法器（出牌费用0）。"
                ),
            },
            # ── BOSS 战特殊注意 ──────────────────────────────────────────────
            {
                "character": "all",
                "route": "boss_tips",
                "text": (
                    "【第一幕BOSS-心脏守卫Gremlin Nob等】进入战斗时检查BOSS意图，"
                    "第一幕BOSS通常在特定回合使用强力技能，需在那轮前确保格挡/击杀。"
                    "携带至少一张能破坏 Artifact 的牌（诅咒/消耗技能）对付有护甲的精英/BOSS。"
                ),
            },
        ]

    def _get_builtin_synergies(self) -> List[Dict]:
        """内置卡牌协同知识"""
        return [
            {
                "cards": ["Whirlwind", "Offering", "Corruption"],
                "description": "旋风斩+燃烧祭品+腐败: 腐败让技能牌0费，祭品给6能量，旋风斩打出12+伤害",
            },
            {
                "cards": ["Catalyst", "Noxious Fumes", "Poison"],
                "description": "催化剂+毒雾+毒素: 毒雾每回合叠2毒，催化剂翻三倍，快速致命",
            },
            {
                "cards": ["Electrodynamics", "All for One", "Zap"],
                "description": "静电场+合而为一+充电: 每张闪电球伤害所有敌人，配合合而为一疯狂出牌",
            },
            {
                "cards": ["Deva Form", "Prostrate", "Reach Heaven"],
                "description": "神圣形态+五体投地: 神圣形态每回合叠加力量，五体投地提供格挡和力量叠加",
            },
        ]


# ──────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    builder = KnowledgeBuilder(output_path="data/knowledge_base.json")
    kb = builder.build(use_spire_codex=True)
    print("\n🎮 知识库构建完成！")
    print(
        f"条目: cards={len(kb['cards'])} relics={len(kb['relics'])} "
        f"monsters={len(kb['monsters'])} strategies={len(kb['strategies'])}"
    )