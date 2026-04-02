"""
scripts/train.py
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING
from datetime import datetime

import torch
import yaml
from torch.distributions import Categorical

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
_parent = os.path.dirname(_ROOT)
if os.path.basename(_ROOT) == "scripts":
    sys.path.insert(0, _parent)

from sts2_env import STS2Env
from ppo_agent import STS2PolicyNet, PPOAgent
from reward_shaper import RewardShaper
from rollout_buffer import RolloutBuffer

if TYPE_CHECKING:
    from llm_advisor import LLMAdvisor


# ── 配置加载 ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def _progress_state_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "training_state.json")


def _pending_buffer_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "pending_buffer.pt")


def _resolve_saved_path(path_str: str, base_dirs: Tuple[str, ...]) -> str:
    if not path_str:
        return ""
    if os.path.isabs(path_str):
        return path_str
    candidates = [
        os.path.abspath(path_str),
    ]
    for b in base_dirs:
        if not b:
            continue
        candidates.append(os.path.abspath(os.path.join(b, path_str)))
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def _load_progress_state(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as ex:
        print(f"⚠️ 读取断点状态失败: {path} | {ex}")
        return None
    return None


def _save_progress_state(path: str, state: Dict):
    # Windows 上目标文件偶发被杀软/索引器/编辑器短暂占用，
    # 这里做短重试，避免单次 PermissionError 中断训练。
    retries = 8
    base_delay = 0.05
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for i in range(retries):
        tmp = f"{path}.{os.getpid()}.{int(time.time() * 1000)}.{i}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except PermissionError as ex:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            if i < retries - 1:
                time.sleep(base_delay * (i + 1))
                continue
            print(f"⚠️ 训练状态写入被占用，已跳过本次保存: {path} | {ex}")
            return
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise


def _recover_episode_stats_from_log(run_dir: str) -> Tuple[float, int]:
    log_path = os.path.join(run_dir, "module_episode_summary.log")
    if not os.path.exists(log_path):
        return 0.0, 0

    reward_sum = 0.0
    count = 0
    reward_re = re.compile(r"reward=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                m = reward_re.search(line)
                if not m:
                    continue
                reward_sum += float(m.group(1))
                count += 1
    except Exception:
        return 0.0, 0
    return reward_sum, count


def _recover_update_stats_from_log(run_dir: str) -> Tuple[float, float, float, int]:
    log_path = os.path.join(run_dir, "module_ppo_update.log")
    if not os.path.exists(log_path):
        return 0.0, 0.0, 0.0, 0

    pg_sum = 0.0
    vf_sum = 0.0
    ent_sum = 0.0
    count = 0
    metrics_re = re.compile(
        r"pg_loss=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"vf_loss=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"entropy=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                m = metrics_re.search(line)
                if not m:
                    continue
                pg_sum += float(m.group(1))
                vf_sum += float(m.group(2))
                ent_sum += float(m.group(3))
                count += 1
    except Exception:
        return 0.0, 0.0, 0.0, 0
    return pg_sum, vf_sum, ent_sum, count


def _recover_global_max_floor_from_log(run_dir: str) -> int:
    log_path = os.path.join(run_dir, "module_episode_summary.log")
    if not os.path.exists(log_path):
        return 0

    best_floor = 0
    floor_re = re.compile(r"max_floor=(\d+)")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                m = floor_re.search(line)
                if not m:
                    continue
                best_floor = max(best_floor, int(m.group(1)))
    except Exception:
        return 0
    return best_floor


def _extract_progress_snapshot(
    total_steps: int,
    episode: int,
    best_episode_reward: float,
    latest_checkpoint: str,
    latest_run_dir: str = "",
    current_episode_reward: float = 0.0,
    extra_state: Optional[Dict] = None,
) -> Dict:
    snapshot = {
        "total_steps": int(total_steps),
        "episode": int(episode),
        "best_episode_reward": float(best_episode_reward),
        "latest_checkpoint": os.path.abspath(latest_checkpoint) if latest_checkpoint else "",
        "latest_run_dir": os.path.abspath(latest_run_dir) if latest_run_dir else "",
        "current_episode_reward": float(current_episode_reward),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if isinstance(extra_state, dict):
        snapshot.update(extra_state)
    return snapshot


# ── 组件构建 ──────────────────────────────────────────────────────────────────

def build_agent(cfg: Dict, device: str) -> PPOAgent:
    policy = STS2PolicyNet(
        num_actions=cfg["env"]["num_actions"],
        hidden_dim=cfg["model"]["hidden_dim"],
        card_d_model=cfg["model"].get("card_d_model", 64),
        monster_d_model=cfg["model"].get("monster_d_model", 32),
    )
    return PPOAgent(
        policy=policy,
        lr=cfg["train"]["lr"],
        clip_eps=cfg["train"]["clip_eps"],
        value_loss_coef=cfg["train"]["value_loss_coef"],
        entropy_coef=cfg["train"]["entropy_coef"],
        gamma=cfg["train"]["gamma"],
        gae_lambda=cfg["train"]["gae_lambda"],
        device=device,
    )


def build_llm_advisor(cfg: Dict) -> Optional["LLMAdvisor"]:
    llm_cfg = cfg.get("llm", {})
    if not llm_cfg.get("enabled", False):
        print("ℹ️  LLM Advisor 已禁用")
        return None

    # Lazy import: keep pure-RL path runnable even if llm_advisor has issues.
    from llm_advisor import LLMAdvisor, LLMBackend

    backend = LLMBackend(
        backend=llm_cfg.get("backend", "ollama"),
        model=llm_cfg.get("model", "qwen2.5:7b"),
        api_key=llm_cfg.get("api_key", ""),
        base_url=llm_cfg.get("base_url", ""),
    )
    advisor = LLMAdvisor(
        llm_backend=backend,
        knowledge_base_path=llm_cfg.get("knowledge_base_path", "data/knowledge_base.json"),
        call_interval_steps=llm_cfg.get("call_interval_steps", 10),
        cache_ttl=float(llm_cfg.get("cache_ttl", 30.0)),
        card_shaping_confidence_threshold=llm_cfg.get("confidence_threshold", 0.55),
        combat_bias_steps=llm_cfg.get("combat_bias_steps", 3),
    )
    print(f"✅ LLM Advisor: {llm_cfg.get('backend')} / {llm_cfg.get('model')}")
    return advisor


# ── 阶段性权重调整 ────────────────────────────────────────────────────────────────

def get_phase_adjusted_weights(total_steps: int, total_steps_limit: int, reward_cfg: Dict) -> Dict[str, float]:
    """
    根据训练进度计算阶段性权重调整。
    
    分为三个阶段（按 total_steps 比例切分，而不是固定步数）：
    - 前期: [0, total_steps * early_end_ratio)
    - 中期: [total_steps * early_end_ratio, total_steps * mid_end_ratio)
    - 后期: [total_steps * mid_end_ratio, total_steps]
    
    返回字典包含 5 个 layer 权重值。
    """
    schedule = reward_cfg.get("phase_schedule") or {}
    enabled = bool(schedule.get("enabled", True))

    base = {
        "layer_a": float(reward_cfg.get("layer_a_weight", 1.0)),
        "layer_b": float(reward_cfg.get("layer_b_weight", 1.0)),
        "layer_c": float(reward_cfg.get("layer_c_weight", 1.0)),
        "layer_d": float(reward_cfg.get("layer_d_weight", 0.3)),
        "layer_e": float(reward_cfg.get("layer_e_weight", 1.0)),
    }
    if not enabled or total_steps_limit <= 0:
        return base

    early_end_ratio = float(schedule.get("early_end_ratio", 0.2))
    mid_end_ratio = float(schedule.get("mid_end_ratio", 0.7))
    early_end_ratio = min(max(early_end_ratio, 0.05), 0.9)
    mid_end_ratio = min(max(mid_end_ratio, early_end_ratio + 0.05), 0.98)

    early_weights = schedule.get("early_weights") or {
        "layer_a": 0.8,
        "layer_b": 0.8,
        "layer_c": 1.2,
        "layer_d": 0.5,
        "layer_e": 0.3,
    }
    mid_start_weights = schedule.get("mid_start_weights") or dict(early_weights)
    mid_end_weights = schedule.get("mid_end_weights") or {
        "layer_a": 0.82,
        "layer_b": 0.82,
        "layer_c": 1.15,
        "layer_d": 0.5,
        "layer_e": 0.7,
    }
    late_weights = schedule.get("late_weights") or {
        "layer_a": 0.85,
        "layer_b": 0.85,
        "layer_c": 1.1,
        "layer_d": 0.5,
        "layer_e": 0.8,
    }

    phase_1_steps = max(1, int(total_steps_limit * early_end_ratio))
    phase_2_steps = max(phase_1_steps + 1, int(total_steps_limit * mid_end_ratio))

    if total_steps < phase_1_steps:
        return {
            "layer_a": float(early_weights.get("layer_a", base["layer_a"])),
            "layer_b": float(early_weights.get("layer_b", base["layer_b"])),
            "layer_c": float(early_weights.get("layer_c", base["layer_c"])),
            "layer_d": float(early_weights.get("layer_d", base["layer_d"])),
            "layer_e": float(early_weights.get("layer_e", base["layer_e"])),
        }

    if total_steps < phase_2_steps:
        denom = max(1, phase_2_steps - phase_1_steps)
        progress = (total_steps - phase_1_steps) / denom
        return {
            "layer_a": float(mid_start_weights.get("layer_a", base["layer_a"])) +
                (float(mid_end_weights.get("layer_a", base["layer_a"])) - float(mid_start_weights.get("layer_a", base["layer_a"]))) * progress,
            "layer_b": float(mid_start_weights.get("layer_b", base["layer_b"])) +
                (float(mid_end_weights.get("layer_b", base["layer_b"])) - float(mid_start_weights.get("layer_b", base["layer_b"]))) * progress,
            "layer_c": float(mid_start_weights.get("layer_c", base["layer_c"])) +
                (float(mid_end_weights.get("layer_c", base["layer_c"])) - float(mid_start_weights.get("layer_c", base["layer_c"]))) * progress,
            "layer_d": float(mid_start_weights.get("layer_d", base["layer_d"])) +
                (float(mid_end_weights.get("layer_d", base["layer_d"])) - float(mid_start_weights.get("layer_d", base["layer_d"]))) * progress,
            "layer_e": float(mid_start_weights.get("layer_e", base["layer_e"])) +
                (float(mid_end_weights.get("layer_e", base["layer_e"])) - float(mid_start_weights.get("layer_e", base["layer_e"]))) * progress,
        }

    return {
        "layer_a": float(late_weights.get("layer_a", base["layer_a"])),
        "layer_b": float(late_weights.get("layer_b", base["layer_b"])),
        "layer_c": float(late_weights.get("layer_c", base["layer_c"])),
        "layer_d": float(late_weights.get("layer_d", base["layer_d"])),
        "layer_e": float(late_weights.get("layer_e", base["layer_e"])),
    }


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _extract_agent_card_index(executed_action: Dict, screen: str) -> Optional[int]:
    """
    从已执行动作里解析 agent 选的奖励卡索引。
    返回 None 表示不是选牌动作，不触发 match bonus。

        STS2AIAgent API:
            - {"action": "choose_reward_card", "option_index": k}  k 为 0-based
            - {"action": "skip_reward_cards"}
    旧式封装（若日后在 env 里包一层）:
      - {"type": "choose_reward", "payload": {"skip": bool, "card_index": int}}
    """
    upper_screen = str(screen or "").upper()
    if upper_screen not in ("CARD_REWARD", "REWARD"):
        return None

    raw = executed_action.get("action", "")
    if raw == "skip_reward_cards":
        return -1
    if raw == "choose_reward_card":
        return int(executed_action.get("option_index", 0))

    if executed_action.get("type") == "choose_reward":
        payload = executed_action.get("payload", {})
        if payload.get("skip", False):
            return -1
        return int(payload.get("card_index", 0))

    return None


def _get_reward_cards_from_state(state: Dict) -> list:
    """从游戏状态里取出奖励卡列表（对齐 STS2AIAgent reward.card_options）"""
    cr = state.get("card_reward") or state.get("reward") or {}
    if isinstance(cr, dict):
        options = cr.get("card_options")
        if isinstance(options, list):
            return options
        cards = cr.get("cards")
        if isinstance(cards, list):
            return cards
        return []
    if isinstance(cr, list):
        return cr
    return []


def _get_map_options_from_state(state: Dict) -> list:
    m = state.get("map") or {}
    opts = m.get("next_options")
    return list(opts) if isinstance(opts, list) else []


def _get_relic_options_from_state(state: Dict) -> list:
    chest = state.get("chest") or {}
    relics = chest.get("relic_options")
    if isinstance(relics, list) and relics:
        return relics

    reward = state.get("reward") or {}
    rewards = reward.get("rewards")
    if isinstance(rewards, list):
        out = []
        for r in rewards:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name", "")).lower()
            kind = str(r.get("reward_type", r.get("type", ""))).lower()
            if "relic" in name or "relic" in kind:
                out.append(r)
        if out:
            return out
    return []


def _extract_agent_relic_index(executed_action: Dict) -> Optional[int]:
    action = str(executed_action.get("action", ""))
    if action == "choose_treasure_relic":
        return int(executed_action.get("index", executed_action.get("option_index", 0)))
    return None


def _extract_agent_map_index(executed_action: Dict) -> Optional[int]:
    action = str(executed_action.get("action", ""))
    if action == "choose_map_node":
        return int(executed_action.get("index", executed_action.get("option_index", 0)))
    return None


def _extract_combat_card_played(executed_action: Dict) -> Optional[int]:
    action = str(executed_action.get("action", ""))
    if action == "play_card":
        return int(executed_action.get("card_index", 0))
    return None


def _extract_agent_remove_index(executed_action: Dict, screen: str) -> Optional[int]:
    if str(screen or "").upper() != "CARD_SELECTION":
        return None
    if str(executed_action.get("action", "")) != "select_deck_card":
        return None
    return int(executed_action.get("option_index", executed_action.get("index", -1)) or -1)


def _extract_agent_shop_choice(executed_action: Dict) -> Tuple[Optional[str], Optional[int]]:
    action = str(executed_action.get("action", ""))
    if action in ("buy_card", "buy_relic", "buy_potion"):
        idx = int(executed_action.get("option_index", executed_action.get("index", -1)) or -1)
        return action, (idx if idx >= 0 else None)
    if action in ("remove_card_at_shop", "proceed"):
        return action, None
    return None, None


def _is_remove_selection_context(state: Dict) -> bool:
    selection = state.get("selection") or {}
    prompt = str(selection.get("prompt", "") or "").lower()
    kind = str(selection.get("kind", "") or "").lower()
    if "remove" in kind or "移除" in prompt or "remove" in prompt:
        return True
    return False


def _should_manual_intervention(raw_state: Dict, action_mask_list: list) -> bool:
    _ = action_mask_list
    return str(raw_state.get("screen", "")).upper() == "UNKNOWN"


def _is_potion_slots_full(raw_state: Dict) -> bool:
    potions = raw_state.get("potions")
    if not isinstance(potions, list) or not potions:
        return False

    entries = [p for p in potions if isinstance(p, dict)]
    if not entries:
        return False

    if any("occupied" in p for p in entries):
        return all(bool(p.get("occupied", False)) for p in entries)

    if any("potion_id" in p for p in entries):
        return all(p.get("potion_id") is not None for p in entries)

    return False


def _is_discard_only_wait_state(raw_state: Dict) -> bool:
    """
    STS2 在敌方行动/动画结算窗口里，偶发仅暴露 discard_potion。
    当药水槽已满时允许执行一次 discard_potion，否则继续等待状态推进。
    """
    if str(raw_state.get("screen", "")).upper() != "COMBAT":
        return False
    legal = [str(a) for a in (raw_state.get("legal_actions") or [])]
    return bool(legal) and all(a == "discard_potion" for a in legal) and not _is_potion_slots_full(raw_state)


def _mask_discard_potion_actions(action_mask_list: list, raw_state: Dict, action_handler) -> Tuple[list, list, Optional[int]]:
    """
    在有其它可执行动作时，禁止策略发送 discard_potion。
    当药水槽已满时，自动放行一次 discard_potion。
    返回 (新mask, 被屏蔽的action_id列表, 自动执行的discard action_id)。
    """
    legal = [str(a) for a in (raw_state.get("legal_actions") or [])]
    if "discard_potion" not in legal:
        return action_mask_list, [], None

    auto_discard_action_id: Optional[int] = None
    if _is_potion_slots_full(raw_state):
        for action_id, enabled in enumerate(action_mask_list):
            if not enabled:
                continue
            try:
                payload = action_handler.decode(action_id, raw_state)
            except Exception:
                continue
            if str(payload.get("action", "")) == "discard_potion":
                auto_discard_action_id = int(action_id)
                break
        if auto_discard_action_id is not None:
            return action_mask_list, [], auto_discard_action_id

    masked_ids = []
    new_mask = list(action_mask_list)
    for action_id, enabled in enumerate(action_mask_list):
        if not enabled:
            continue
        try:
            payload = action_handler.decode(action_id, raw_state)
        except Exception:
            continue
        if str(payload.get("action", "")) == "discard_potion":
            new_mask[action_id] = False
            masked_ids.append(action_id)

    if any(new_mask):
        return new_mask, masked_ids, None
    return action_mask_list, [], None


def _find_action_id_by_payload(
    action_mask_list: list,
    raw_state: Dict,
    action_handler,
    action_name: str,
    option_index: Optional[int] = None,
    card_index: Optional[int] = None,
) -> Optional[int]:
    for action_id, enabled in enumerate(action_mask_list):
        if not enabled:
            continue
        try:
            payload = action_handler.decode(action_id, raw_state)
        except Exception:
            continue

        if str(payload.get("action", "")) != action_name:
            continue

        if option_index is not None:
            if int(payload.get("option_index", -10**9) or -10**9) != int(option_index):
                continue

        if card_index is not None:
            if int(payload.get("card_index", -10**9) or -10**9) != int(card_index):
                continue

        return int(action_id)

    return None


def _resolve_llm_guided_action_id(
    llm_advisor,
    screen: str,
    raw_state: Dict,
    action_mask_list: list,
    action_handler,
    combat_step_counter: int,
    combat_guidance_steps: int,
) -> Tuple[Optional[int], float, str]:
    if llm_advisor is None:
        return None, 0.0, ""

    upper_screen = str(screen or "").upper()

    if upper_screen in ("REWARD", "CARD_REWARD"):
        reward = raw_state.get("reward") or {}
        pending_choice = bool(reward.get("pending_card_choice", False))
        if pending_choice:
            rec_idx, rec_conf = llm_advisor.get_last_card_recommendation()
            if rec_idx == -1:
                aid = _find_action_id_by_payload(action_mask_list, raw_state, action_handler, "skip_reward_cards")
                return aid, float(rec_conf), "card_skip"
            if rec_idx >= 0:
                aid = _find_action_id_by_payload(
                    action_mask_list,
                    raw_state,
                    action_handler,
                    "choose_reward_card",
                    option_index=int(rec_idx),
                )
                return aid, float(rec_conf), "card_pick"

    if upper_screen == "MAP":
        rec_idx, rec_conf, _ = llm_advisor.get_last_map_recommendation()
        if rec_idx >= 0:
            aid = _find_action_id_by_payload(
                action_mask_list,
                raw_state,
                action_handler,
                "choose_map_node",
                option_index=int(rec_idx),
            )
            return aid, float(rec_conf), "map"

    if upper_screen == "CHEST":
        rec_idx, rec_conf = llm_advisor.get_last_relic_recommendation()
        if rec_idx >= 0:
            aid = _find_action_id_by_payload(
                action_mask_list,
                raw_state,
                action_handler,
                "choose_treasure_relic",
                option_index=int(rec_idx),
            )
            return aid, float(rec_conf), "chest_relic"

    if upper_screen == "CARD_SELECTION" and _is_remove_selection_context(raw_state):
        rec_idx, rec_conf = llm_advisor.get_last_remove_recommendation()
        if rec_idx >= 0:
            aid = _find_action_id_by_payload(
                action_mask_list,
                raw_state,
                action_handler,
                "select_deck_card",
                option_index=int(rec_idx),
            )
            return aid, float(rec_conf), "remove"

    if upper_screen == "SHOP":
        rec_action, rec_index, rec_conf = llm_advisor.get_last_shop_recommendation()
        if rec_action in ("buy_card", "buy_relic", "buy_potion") and rec_index is not None:
            aid = _find_action_id_by_payload(
                action_mask_list,
                raw_state,
                action_handler,
                rec_action,
                option_index=int(rec_index),
            )
            return aid, float(rec_conf), "shop_item"
        if rec_action in ("remove_card_at_shop", "proceed"):
            aid = _find_action_id_by_payload(action_mask_list, raw_state, action_handler, rec_action)
            return aid, float(rec_conf), "shop_action"

    if upper_screen == "COMBAT" and combat_step_counter < combat_guidance_steps:
        opening = llm_advisor.get_last_combat_opening()
        seq = opening.get("opening_card_sequence") if isinstance(opening, dict) else []
        if isinstance(seq, list) and seq:
            try:
                card_idx = int(seq[0])
            except Exception:
                card_idx = -1
            if card_idx >= 0:
                aid = _find_action_id_by_payload(
                    action_mask_list,
                    raw_state,
                    action_handler,
                    "play_card",
                    card_index=card_idx,
                )
                # 战斗开局建议未单独提供 confidence，给一个保守默认值。
                return aid, 0.6, "combat_opening"

    return None, 0.0, ""


def _is_card_select_metadata_missing_wait_state(raw_state: Dict) -> bool:
    if str(raw_state.get("screen", "")).upper() != "CARD_SELECTION":
        return False
    legal = [str(a) for a in (raw_state.get("legal_actions") or [])]
    if "select_deck_card" not in legal:
        return False
    selection = raw_state.get("selection") or {}
    cards = selection.get("cards") if isinstance(selection.get("cards"), list) else []
    return len(cards) == 0


def _card_selection_signature(state: Dict) -> Tuple:
    selection = state.get("selection") or {}
    cards = selection.get("cards") if isinstance(selection.get("cards"), list) else []
    card_sig = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        card_sig.append(
            (
                int(c.get("index", -1) or -1),
                str(c.get("id", c.get("name", "")) or ""),
                bool(c.get("selected", False) or c.get("is_selected", False) or c.get("chosen", False)),
            )
        )
    return (
        int(selection.get("selected_count", 0) or 0),
        int(selection.get("min_select", 0) or 0),
        int(selection.get("max_select", 0) or 0),
        bool(selection.get("can_confirm", False)),
        bool(selection.get("requires_confirmation", False)),
        tuple(card_sig),
    )


def _state_progress_signature(state: Dict) -> Tuple:
    """Build a stable compact signature for generic no-progress detection."""
    screen = str(state.get("screen", "") or "").upper()
    phase = str(state.get("phase", "") or "").lower()
    can_act = bool(state.get("can_act", False))
    block_reason = str(state.get("block_reason", "") or "")
    floor = int(state.get("floor", 0) or 0)
    gold = int(state.get("gold", 0) or 0)
    legal = tuple(sorted(str(a) for a in (state.get("legal_actions") or [])))

    combat = state.get("combat") or {}
    player = combat.get("player") or {}
    turn = int(combat.get("turn", 0) or 0)
    player_sig = (
        int(player.get("hp", state.get("hp", 0)) or 0),
        int(player.get("max_hp", state.get("max_hp", 0)) or 0),
        int(player.get("block", 0) or 0),
        int(player.get("energy", combat.get("energy", 0)) or 0),
    )

    hand = combat.get("hand") if isinstance(combat.get("hand"), list) else []
    hand_sig = []
    for c in hand:
        if not isinstance(c, dict):
            continue
        cost_raw = c.get("energy_cost", c.get("cost", -1))
        cost_sig = cost_raw.strip().upper() if isinstance(cost_raw, str) else int(cost_raw or 0)
        hand_sig.append(
            (
                str(c.get("id", c.get("name", "")) or ""),
                cost_sig,
                bool(c.get("playable", False)),
                bool(c.get("upgraded", False)),
            )
        )

    monsters = combat.get("monsters") if isinstance(combat.get("monsters"), list) else []
    monster_sig = []
    for m in monsters:
        if not isinstance(m, dict):
            continue
        intent = m.get("intent") or {}
        if isinstance(intent, dict):
            intent_name = str(intent.get("name", intent.get("intent_type", "")) or "")
            intent_damage = int(intent.get("total_damage", 0) or 0)
        else:
            intent_name = str(intent or "")
            intent_damage = 0
        monster_sig.append(
            (
                str(m.get("id", m.get("name", "")) or ""),
                bool(m.get("alive", True)),
                int(m.get("hp", 0) or 0),
                int(m.get("block", 0) or 0),
                intent_name,
                intent_damage,
            )
        )

    potions = state.get("potions") if isinstance(state.get("potions"), list) else []
    potion_sig = []
    for p in potions:
        if not isinstance(p, dict):
            continue
        potion_sig.append(
            (
                str(p.get("id", p.get("name", "")) or ""),
                bool(p.get("can_use", p.get("usable", False))),
                bool(p.get("requires_target", False)),
            )
        )

    reward = state.get("reward") or {}
    rewards = reward.get("rewards") if isinstance(reward.get("rewards"), list) else []
    reward_sig = (
        bool(reward.get("pending_card_choice", False)),
        tuple(str(r.get("reward_type", "")) for r in rewards if isinstance(r, dict)),
        len(reward.get("card_options") if isinstance(reward.get("card_options"), list) else []),
    )

    selection_sig = _card_selection_signature(state) if screen == "CARD_SELECTION" else ()

    shop = state.get("shop") or {}

    def _count_affordable(items):
        if not isinstance(items, list):
            return 0
        count = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            affordable = bool(it.get("affordable", it.get("enough_gold", False)))
            stocked = bool(it.get("stocked", it.get("available", True)))
            if affordable and stocked:
                count += 1
        return count

    shop_sig = (
        bool(shop.get("is_open", False)),
        _count_affordable(shop.get("cards")),
        _count_affordable(shop.get("relics")),
        _count_affordable(shop.get("potions")),
    )

    map_block = state.get("map") or {}
    next_options = map_block.get("next_options") if isinstance(map_block.get("next_options"), list) else []
    map_sig = tuple(
        int((n or {}).get("index", i) or i) if isinstance(n, dict) else int(n or i)
        for i, n in enumerate(next_options)
    )

    return (
        screen,
        phase,
        can_act,
        block_reason,
        floor,
        gold,
        legal,
        player_sig,
        turn,
        tuple(hand_sig),
        tuple(monster_sig),
        tuple(potion_sig),
        reward_sig,
        selection_sig,
        shop_sig,
        map_sig,
    )


def _is_no_progress_step(prev_state: Dict, new_state: Dict, done: bool, truncated: bool) -> bool:
    if done or truncated:
        return False
    return _state_progress_signature(prev_state) == _state_progress_signature(new_state)


def _deadlock_retry_action_priority(action_name: str) -> int:
    order = (
        "collect_rewards_and_proceed",
        "proceed",
        "choose_event_option",
        "choose_map_node",
        "claim_reward",
        "choose_reward_card",
        "skip_reward_cards",
        "choose_treasure_relic",
        "open_chest",
        "choose_rest_option",
        "close_shop_inventory",
        "open_shop_inventory",
        "select_deck_card",
        "confirm_selection",
        "end_turn",
        "play_card",
        "use_potion",
        "buy_card",
        "buy_relic",
        "buy_potion",
        "remove_card_at_shop",
    )
    try:
        return order.index(action_name)
    except ValueError:
        return len(order)


def _pick_deadlock_retry_action_id(raw_state: Dict, blocked_actions: Set[int], action_handler) -> Optional[int]:
    if not blocked_actions:
        return None

    best_id: Optional[int] = None
    best_rank = 10**9
    best_option = 10**9

    for aid in sorted(blocked_actions):
        try:
            payload = action_handler.decode(aid, raw_state)
        except Exception:
            payload = {}
        action_name = str(payload.get("action", ""))
        rank = _deadlock_retry_action_priority(action_name)
        option = int(payload.get("option_index", 0) or 0)
        if rank < best_rank or (rank == best_rank and option < best_option):
            best_rank = rank
            best_option = option
            best_id = aid

    if best_id is not None:
        return best_id
    return min(blocked_actions)


def _is_menu_bootstrap_state(raw_state: Dict) -> bool:
    screen = str(raw_state.get("screen", "")).upper()
    phase = str(raw_state.get("phase", "")).lower()
    return screen in ("MAIN_MENU", "CHARACTER_SELECT", "MODAL", "UNKNOWN") and phase != "run"


class RunLogger:
    """
    运行期分模块日志：
    logs/<YYYYmmdd_HHMMSS>/
      - module_agent_decision.log
      - module_env_step_state.log
      - module_reward_shaping.log
      - module_ppo_update.log
      - module_episode_summary.log
      - module_error_recovery.log
      - run_config_snapshot.json
    """

    def __init__(self, cfg: Dict, resume_run_dir: Optional[str] = None):
        if resume_run_dir and os.path.isdir(resume_run_dir):
            self.run_dir = resume_run_dir
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join("logs", ts)
        os.makedirs(self.run_dir, exist_ok=True)
        self.paths = {
            "agent_decision": os.path.join(self.run_dir, "module_agent_decision.log"),
            "env_step_state": os.path.join(self.run_dir, "module_env_step_state.log"),
            "reward_shaping": os.path.join(self.run_dir, "module_reward_shaping.log"),
            "ppo_update": os.path.join(self.run_dir, "module_ppo_update.log"),
            "episode_summary": os.path.join(self.run_dir, "module_episode_summary.log"),
            "error_recovery": os.path.join(self.run_dir, "module_error_recovery.log"),
            "manual_intervention": os.path.join(self.run_dir, "module_manual_intervention.log"),
        }
        snapshot_name = "run_config_snapshot.json" if not resume_run_dir else f"run_config_snapshot_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(self.run_dir, snapshot_name), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f"🗂️  本次运行日志目录: {self.run_dir}")

    def log(self, key: str, msg: str):
        path = self.paths.get(key)
        if not path:
            return
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {msg}\n")


# ── 主训练循环 ────────────────────────────────────────────────────────────────

def train(cfg: Dict):
    device = cfg.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 开始训练 | 设备: {device}")

    config_dir = str(cfg.get("__config_dir__", _ROOT) or _ROOT)
    checkpoint_dir_raw = cfg.get("checkpoint_dir", "checkpoints")
    checkpoint_dir = checkpoint_dir_raw if os.path.isabs(checkpoint_dir_raw) else os.path.abspath(os.path.join(config_dir, checkpoint_dir_raw))
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_cfg = cfg.get("train", {})
    resume_on_restart = bool(train_cfg.get("resume_on_restart", True))
    save_latest_per_update = bool(train_cfg.get("save_latest_per_update", True))
    continue_logs_on_resume = bool(train_cfg.get("continue_logs_on_resume", True))
    progress_path = _progress_state_path(checkpoint_dir)
    pending_buffer_path = _pending_buffer_path(checkpoint_dir)
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest_model.pt")

    state_payload = _load_progress_state(progress_path) if resume_on_restart else None
    if resume_on_restart:
        if state_payload is None:
            print(f"ℹ️ 未发现可用断点状态文件: {progress_path}，将从头开始。")
        else:
            print(
                "ℹ️ 发现断点状态: "
                f"steps={state_payload.get('total_steps', 0)} "
                f"episode={state_payload.get('episode', 0)} "
                f"latest_ckpt={state_payload.get('latest_checkpoint', '')}"
            )
    resume_run_dir = ""
    if continue_logs_on_resume and state_payload:
        candidate_run_dir = str(state_payload.get("latest_run_dir", "")).strip()
        candidate_run_dir = _resolve_saved_path(candidate_run_dir, (config_dir, _ROOT))
        if candidate_run_dir and os.path.isdir(candidate_run_dir):
            resume_run_dir = candidate_run_dir
    run_logger = RunLogger(cfg, resume_run_dir=resume_run_dir or None)

    env = STS2Env(
        host=cfg["env"].get("host", "localhost"),
        port=cfg["env"].get("port", 18080),
        character_index=cfg["env"].get("character_index", 0),
        startup_debug=cfg["env"].get("startup_debug", False),
        action_poll_interval=cfg["env"].get("action_poll_interval", 0.5),
        action_min_interval=cfg["env"].get("action_min_interval", 0.5),
        post_action_settle=cfg["env"].get("post_action_settle", 0.5),
        action_retry_count=cfg["env"].get("action_retry_count", 1),
        render_mode="human" if cfg.get("render") else None,
    )

    configured_actions = int(cfg["env"].get("num_actions", env.action_space.n))
    actual_actions = int(env.action_space.n)
    if configured_actions != actual_actions:
        raise ValueError(
            f"env.num_actions({configured_actions}) 与环境动作维度({actual_actions})不一致，请修正配置。"
        )

    agent        = build_agent(cfg, device)
    llm_advisor  = build_llm_advisor(cfg)
    _r = cfg.get("reward", {})
    _l = cfg.get("llm", {})

    if not bool(_l.get("enabled", False)):
        _r = dict(_r)
        for k in (
            "llm_weight",
            "card_weight",
            "relic_choice_weight",
            "map_route_weight",
            "remove_choice_weight",
            "shop_choice_weight",
            "combat_opening_weight",
        ):
            _r[k] = 0.0
        print("ℹ️  单RL模式：已强制将所有LLM相关奖励系数置为0")

    reward_shaper = RewardShaper(
        llm_advisor=llm_advisor,
        llm_weight=_r.get("llm_weight", 0.3),
        # 层权重
        layer_a_weight=_r.get("layer_a_weight", 1.0),
        layer_b_weight=_r.get("layer_b_weight", 1.0),
        layer_c_weight=_r.get("layer_c_weight", 1.0),
        layer_d_weight=_r.get("layer_d_weight", 0.3),
        layer_e_weight=_r.get("layer_e_weight", 1.0),
        # Layer A
        action_damage_coef=_r.get("action_damage_coef", 0.004),
        action_block_coef=_r.get("action_block_coef", 0.002),
        action_kill_bonus=_r.get("action_kill_bonus", 0.3),
        action_card_pick_bonus=_r.get("action_card_pick_bonus", 0.05),
        action_potion_bonus=_r.get("action_potion_bonus", 0.05),
        # Layer B
        dmg_reward_cap=_r.get("dmg_reward_cap", 1.5),
        kill_reward_per_enemy=_r.get("kill_reward_per_enemy", 2.0),
        block_coverage_reward=_r.get("block_coverage_reward", 1.0),
        excess_block_penalty_cap=_r.get("excess_block_penalty_cap", 0.2),
        overflow_block_penalty_coef=_r.get("overflow_block_penalty_coef", 0.03),
        energy_waste_penalty=_r.get("energy_waste_penalty", 0.5),
        hp_loss_penalty=_r.get("hp_loss_penalty", 1.5),
        hp_loss_urgency_max_mul=_r.get("hp_loss_urgency_max_mul", 2.0),
        action_player_buff_gain_bonus=_r.get("action_player_buff_gain_bonus", 0.05),
        action_enemy_debuff_gain_bonus=_r.get("action_enemy_debuff_gain_bonus", 0.08),
        player_buff_gain_bonus=_r.get("player_buff_gain_bonus", 0.15),
        enemy_debuff_gain_bonus=_r.get("enemy_debuff_gain_bonus", 0.1),
        # Layer C
        normal_combat_bonus=_r.get("normal_combat_bonus", 3.0),
        elite_combat_bonus=_r.get("elite_combat_bonus", 8.0),
        boss_combat_bonus=_r.get("boss_combat_bonus", 20.0),
        boss_extra_bonus=_r.get("boss_extra_bonus", 10.0),
        hp_efficiency_max=_r.get("hp_efficiency_max", 4.0),
        elite_clean_bonus=_r.get("elite_clean_bonus", 3.0),
        elite_clean_threshold=_r.get("elite_clean_threshold", 0.3),
        high_hp_loss_threshold=_r.get("high_hp_loss_threshold", 0.5),
        high_hp_loss_penalty_scale=_r.get("high_hp_loss_penalty_scale", 3.0),
        # Layer D
        rest_low_hp_bonus=_r.get("rest_low_hp_bonus", 1.0),
        rest_mid_hp_bonus=_r.get("rest_mid_hp_bonus", 0.3),
        rest_high_hp_penalty=_r.get("rest_high_hp_penalty", 0.5),
        rest_low_threshold=_r.get("rest_low_threshold", 0.35),
        rest_mid_threshold=_r.get("rest_mid_threshold", 0.6),
        rest_high_threshold=_r.get("rest_high_threshold", 0.8),
        smith_bonus=_r.get("smith_bonus", 0.5),
        remove_card_bonus=_r.get("remove_card_bonus", 0.4),
        choose_card_meta_bonus=_r.get("choose_card_meta_bonus", 0.3),
        claim_gold_bonus=_r.get("claim_gold_bonus", 0.12),
        claim_card_bonus=_r.get("claim_card_bonus", 0.18),
        claim_potion_bonus=_r.get("claim_potion_bonus", 0.15),
        claim_relic_bonus=_r.get("claim_relic_bonus", 0.25),
        claim_remove_card_bonus=_r.get("claim_remove_card_bonus", 0.2),
        claim_special_card_bonus=_r.get("claim_special_card_bonus", 0.2),
        claim_linked_set_bonus=_r.get("claim_linked_set_bonus", 0.18),
        claim_unknown_bonus=_r.get("claim_unknown_bonus", 0.1),
        event_option_bonus=_r.get("event_option_bonus", 0.1),
        map_node_bonus=_r.get("map_node_bonus", 0.0),
        deck_optimal_min=_r.get("deck_optimal_min", 10),
        deck_optimal_max=_r.get("deck_optimal_max", 20),
        deck_optimal_bonus=_r.get("deck_optimal_bonus", 0.05),
        deck_large_threshold=_r.get("deck_large_threshold", 25),
        deck_large_penalty_per_card=_r.get("deck_large_penalty_per_card", 0.1),
        deck_too_small_threshold=_r.get("deck_too_small_threshold", 6),
        deck_too_small_penalty=_r.get("deck_too_small_penalty", 0.1),
        remove_curse_bonus=_r.get("remove_curse_bonus", 0.6),
        buy_card_common_bonus=_r.get("buy_card_common_bonus", 0.1),
        buy_card_uncommon_bonus=_r.get("buy_card_uncommon_bonus", 0.25),
        buy_card_rare_bonus=_r.get("buy_card_rare_bonus", 0.4),
        smith_low_upgrade_ratio_threshold=_r.get("smith_low_upgrade_ratio_threshold", 0.3),
        smith_low_upgrade_ratio_mul=_r.get("smith_low_upgrade_ratio_mul", 1.5),
        buy_bonus=_r.get("buy_bonus", 0.2),
        # Layer E
        terminal_victory_bonus=_r.get("terminal_victory_bonus", 100.0),
        terminal_defeat_penalty=_r.get("terminal_defeat_penalty", 30.0),
        terminal_floor_weight=_r.get("terminal_floor_weight", 1.5),
        terminal_hp_quality_weight=_r.get("terminal_hp_quality_weight", 10.0),
        # LLM match
        confidence_threshold=_l.get("confidence_threshold", 0.55),
        card_weight=_r.get("card_weight", 0.4),
        card_match_bonus=_r.get("card_match_bonus", 1.0),
        card_mismatch_penalty=_r.get("card_mismatch_penalty", 0.5),
        relic_choice_weight=_r.get("relic_choice_weight", 0.25),
        map_route_weight=_r.get("map_route_weight", 0.25),
        combat_opening_weight=_r.get("combat_opening_weight", 0.2),
        remove_choice_weight=_r.get("remove_choice_weight", 0.15),
        remove_match_bonus=_r.get("remove_match_bonus", 0.8),
        remove_mismatch_penalty=_r.get("remove_mismatch_penalty", 0.4),
        shop_choice_weight=_r.get("shop_choice_weight", 0.15),
        shop_match_bonus=_r.get("shop_match_bonus", 0.8),
        shop_mismatch_penalty=_r.get("shop_mismatch_penalty", 0.4),
        combat_bias_steps=_l.get("combat_bias_steps", 3),
        reward_clip=_r.get("reward_clip", 50.0),
    )

    llm_policy_guidance_enabled = bool(_l.get("policy_guidance_enabled", False)) and llm_advisor is not None
    llm_policy_guidance_alpha_max = max(0.0, float(_l.get("policy_guidance_alpha_max", 0.0)))
    llm_policy_guidance_max_bias = max(0.0, float(_l.get("policy_guidance_max_bias", 0.4)))
    llm_policy_guidance_confidence_threshold = float(_l.get("policy_guidance_confidence_threshold", 0.6))
    llm_policy_guidance_ramp_steps = max(1, int(_l.get("policy_guidance_ramp_steps", 10000)))
    llm_policy_guidance_combat_steps = max(0, int(_l.get("policy_guidance_combat_steps", 2)))
    llm_policy_guidance_screens = {
        str(s).upper()
        for s in (_l.get("policy_guidance_screens", ["REWARD", "CARD_REWARD", "MAP", "SHOP", "CARD_SELECTION", "CHEST", "COMBAT"]) or [])
    }
    if llm_policy_guidance_enabled:
        print(
            "🧭 已启用 LLM 概率引导: "
            f"alpha_max={llm_policy_guidance_alpha_max:.3f}, "
            f"conf>={llm_policy_guidance_confidence_threshold:.2f}, "
            f"ramp_steps={llm_policy_guidance_ramp_steps}, "
            f"screens={sorted(llm_policy_guidance_screens)}"
        )

    buffer = RolloutBuffer(buffer_size=cfg["train"]["buffer_size"])

    resume_path = cfg.get("resume")
    if resume_path:
        resume_path = _resolve_saved_path(str(resume_path), (config_dir, _ROOT))
    auto_resume_path = None
    if state_payload:
        candidate = str(state_payload.get("latest_checkpoint", "")).strip()
        candidate = _resolve_saved_path(candidate, (config_dir, _ROOT))
        if candidate and os.path.exists(candidate):
            auto_resume_path = candidate
        elif os.path.exists(latest_ckpt_path):
            auto_resume_path = latest_ckpt_path

    effective_resume = resume_path if (resume_path and os.path.exists(resume_path)) else auto_resume_path
    if effective_resume:
        agent.load(effective_resume)
        print(f"📂 已加载断点: {effective_resume}")

    total_steps = int(state_payload.get("total_steps", 0)) if state_payload else 0
    episode = int(state_payload.get("episode", 0)) if state_payload else 0
    best_episode_reward = float(state_payload.get("best_episode_reward", float("-inf"))) if state_payload else float("-inf")
    current_ep_reward_resume = float(state_payload.get("current_episode_reward", 0.0)) if state_payload else 0.0
    episode_max_floor_resume = int(
        state_payload.get("current_episode_max_floor", state_payload.get("episode_max_floor", 0))
    ) if state_payload else 0
    global_max_floor = int(
        state_payload.get("global_max_floor", state_payload.get("best_max_floor", 0))
    ) if state_payload else 0

    recovered_episode_reward_sum = 0.0
    recovered_episode_count = 0
    recovered_pg_sum = 0.0
    recovered_vf_sum = 0.0
    recovered_entropy_sum = 0.0
    recovered_update_count = 0
    recovered_global_max_floor = 0
    if resume_on_restart and resume_run_dir:
        recovered_episode_reward_sum, recovered_episode_count = _recover_episode_stats_from_log(resume_run_dir)
        recovered_pg_sum, recovered_vf_sum, recovered_entropy_sum, recovered_update_count = _recover_update_stats_from_log(resume_run_dir)
        recovered_global_max_floor = _recover_global_max_floor_from_log(resume_run_dir)

    episode_reward_sum = float(
        state_payload.get("episode_reward_sum", state_payload.get("total_episode_reward_sum", 0.0))
    ) if state_payload else 0.0
    episode_count_for_avg = int(
        state_payload.get("episode_count", state_payload.get("total_episode_count", 0))
    ) if state_payload else 0
    pg_loss_sum = float(state_payload.get("pg_loss_sum", 0.0)) if state_payload else 0.0
    vf_loss_sum = float(state_payload.get("vf_loss_sum", 0.0)) if state_payload else 0.0
    entropy_sum = float(state_payload.get("entropy_sum", 0.0)) if state_payload else 0.0
    update_count = int(state_payload.get("update_count", 0)) if state_payload else 0

    if episode_count_for_avg <= 0 and recovered_episode_count > 0:
        episode_reward_sum = recovered_episode_reward_sum
        episode_count_for_avg = recovered_episode_count
    if update_count <= 0 and recovered_update_count > 0:
        pg_loss_sum = recovered_pg_sum
        vf_loss_sum = recovered_vf_sum
        entropy_sum = recovered_entropy_sum
        update_count = recovered_update_count
    if global_max_floor <= 0 and recovered_global_max_floor > 0:
        global_max_floor = recovered_global_max_floor

    if state_payload and resume_on_restart:
        print(f"🔁 已恢复训练进度: total_steps={total_steps}, episode={episode}, best={best_episode_reward:.3f}")
        if episode_count_for_avg > 0:
            print(f"📈 已恢复累计回报统计: episodes={episode_count_for_avg}, global_avg_ep_r={episode_reward_sum / episode_count_for_avg:.3f}")
        if update_count > 0:
            print(
                "📉 已恢复累计优化统计: "
                f"updates={update_count}, "
                f"global_avg_pg={pg_loss_sum / update_count:.4f}, "
                f"global_avg_vf={vf_loss_sum / update_count:.4f}, "
                f"global_avg_ent={entropy_sum / update_count:.4f}"
            )
        if global_max_floor > 0:
            print(f"🗺️ 已恢复历史最高层数: {global_max_floor}")

    pending_loaded_once = False
    if resume_on_restart and os.path.exists(pending_buffer_path):
        try:
            pending_payload = torch.load(pending_buffer_path, map_location="cpu")
            if isinstance(pending_payload, dict):
                size = int(pending_payload.get("size", 0))
                p_buf_size = int(pending_payload.get("buffer_size", -1))
                if size > 0 and p_buf_size == cfg["train"]["buffer_size"]:
                    buffer.load_state(pending_payload)
                    pending_loaded_once = True
                    print(f"🧩 已恢复未满buffer样本: {size}/{cfg['train']['buffer_size']}")
                elif size <= 0:
                    print("ℹ️ pending_buffer.pt 存在但 size=0，本次不恢复 pending 样本。")
                else:
                    print(
                        "⚠️ pending_buffer.pt 与当前配置不匹配，跳过恢复: "
                        f"pending_buffer_size={p_buf_size}, config_buffer_size={cfg['train']['buffer_size']}"
                    )
        except Exception as ex:
            run_logger.log("error_recovery", f"pending_buffer_load_failed: {ex}")
            print(f"⚠️ pending_buffer.pt 加载失败，已跳过: {ex}")
    elif resume_on_restart:
        print(f"ℹ️ 未找到 pending buffer: {pending_buffer_path}")

    consecutive_env_errors = 0
    max_consecutive_env_errors = 10
    manual_max_wait = float(train_cfg.get("manual_intervention_max_wait", 180.0))
    manual_poll = float(train_cfg.get("manual_intervention_poll", 0.5))
    no_progress_blocked_actions: Dict[Tuple, Set[int]] = {}
    no_progress_wait_counts: Dict[Tuple, int] = {}
    no_progress_retry_counts: Dict[Tuple, int] = {}
    max_no_progress_states = 256
    no_progress_wait_threshold = max(1, int(train_cfg.get("no_progress_wait_threshold", 10)))
    no_progress_retry_limit = max(1, int(train_cfg.get("no_progress_retry_limit", 3)))

    print("⏳ 等待游戏就绪（确保 STS2 + STS2AIAgent Mod 已运行）...")
    obs, info = env.reset()
    run_logger.log("env_step_state", f"reset: screen={info.get('screen')} floor={info.get('floor')} hp={info.get('hp')}/{info.get('max_hp')}")
    prev_screen = ""
    combat_step_counter = 0
    llm_card_triggered = False
    current_ep_reward = current_ep_reward_resume
    current_ep_reward_resume = 0.0
    episode_max_floor = max(int(info.get("floor", 0) or 0), episode_max_floor_resume)
    global_max_floor = max(global_max_floor, episode_max_floor)

    def _save_resume_snapshot():
        snapshot = _extract_progress_snapshot(
            total_steps=total_steps,
            episode=episode,
            best_episode_reward=best_episode_reward,
            latest_checkpoint=effective_resume or "",
            latest_run_dir=run_logger.run_dir,
            current_episode_reward=current_ep_reward,
            extra_state={
                "episode_reward_sum": float(episode_reward_sum),
                "episode_count": int(episode_count_for_avg),
                "pg_loss_sum": float(pg_loss_sum),
                "vf_loss_sum": float(vf_loss_sum),
                "entropy_sum": float(entropy_sum),
                "update_count": int(update_count),
                "current_episode_max_floor": int(episode_max_floor),
                "global_max_floor": int(global_max_floor),
            },
        )
        _save_progress_state(progress_path, snapshot)

    while total_steps < cfg["train"]["total_steps"]:

        # 在收集 rollout 前更新分阶段权重，使当轮采样与权重一致。
        adjusted_weights = get_phase_adjusted_weights(total_steps, cfg["train"]["total_steps"], _r)
        reward_shaper.update_layer_weights(
            layer_a=adjusted_weights["layer_a"],
            layer_b=adjusted_weights["layer_b"],
            layer_c=adjusted_weights["layer_c"],
            layer_d=adjusted_weights["layer_d"],
            layer_e=adjusted_weights["layer_e"],
        )

        # ── 收集 Rollout ──────────────────────────────────────────────────
        if pending_loaded_once:
            pending_loaded_once = False
        else:
            buffer.reset()
        episode_rewards = []
        episode_max_floor = max(episode_max_floor, int(info.get("floor", 0) or 0))
        global_max_floor = max(global_max_floor, episode_max_floor)

        while not buffer.is_full():
            screen = str(info.get("screen", "") or "").upper()
            raw_state   = info.get("raw_state", {})
            episode_max_floor = max(episode_max_floor, int(info.get("floor", 0) or 0))
            global_max_floor = max(global_max_floor, episode_max_floor)
            run_logger.log(
                "env_step_state",
                f"pre_step total_steps={total_steps} buffer_size={len(buffer)} screen={screen} floor={info.get('floor')} gold={info.get('gold')}",
            )

            # 菜单/选角流程只用于自动开局，不计入样本与奖励，避免污染训练数据。
            if _is_menu_bootstrap_state(raw_state):
                run_logger.log(
                    "error_recovery",
                    f"menu_bootstrap_skip: screen={screen} phase={raw_state.get('phase')} legal={raw_state.get('legal_actions')}",
                )
                obs, info = env.reset()
                prev_screen = ""
                combat_step_counter = 0
                llm_card_triggered = False
                continue

            if _is_discard_only_wait_state(raw_state):
                run_logger.log(
                    "error_recovery",
                    f"discard_only_wait: screen={screen} floor={info.get('floor')} legal={raw_state.get('legal_actions')}",
                )
                time.sleep(manual_poll)
                obs, info = env.refresh_state()
                prev_screen = screen
                continue

            if _is_card_select_metadata_missing_wait_state(raw_state):
                run_logger.log(
                    "error_recovery",
                    f"card_select_wait_no_metadata: floor={info.get('floor')} legal={raw_state.get('legal_actions')}",
                )
                time.sleep(manual_poll)
                obs, info = env.refresh_state()
                prev_screen = screen
                continue

            # ── ① LLM战略触发节点（仅 llm.enabled=true 时生效）──────────────
            pending_card_choice = bool((raw_state.get("reward") or {}).get("pending_card_choice", False))
            reward_flow_active = screen in ("CARD_REWARD", "REWARD") and (
                pending_card_choice or "choose_reward_card" in [str(a) for a in (raw_state.get("legal_actions") or [])]
            )

            if reward_flow_active and not llm_card_triggered and llm_advisor is not None:
                reward_cards = _get_reward_cards_from_state(raw_state)
                if reward_cards:
                    rec_idx, rec_conf, rec_reason = llm_advisor.evaluate_card_reward(raw_state, reward_cards)
                    run_logger.log(
                        "reward_shaping",
                        f"llm_card_advice screen={screen} rec_idx={rec_idx} conf={rec_conf:.3f} reason={str(rec_reason).replace(chr(10), ' ')[:120]}",
                    )
                llm_card_triggered = True
            elif not reward_flow_active:
                llm_card_triggered = False

            if screen == "MAP" and prev_screen != "MAP" and llm_advisor is not None:
                route_options = _get_map_options_from_state(raw_state)
                if route_options:
                    rec_idx, rec_conf, rec_reason, route_scores = llm_advisor.evaluate_map_route(raw_state, route_options)
                    run_logger.log(
                        "reward_shaping",
                        f"llm_map_advice rec_idx={rec_idx} conf={rec_conf:.3f} scores={route_scores} reason={str(rec_reason).replace(chr(10), ' ')[:120]}",
                    )

            if screen in ("CHEST", "REWARD") and prev_screen not in ("CHEST", "REWARD") and llm_advisor is not None:
                relic_options = _get_relic_options_from_state(raw_state)
                if relic_options:
                    rec_idx, rec_conf, rec_reason = llm_advisor.evaluate_relic_choice(raw_state, relic_options)
                    run_logger.log(
                        "reward_shaping",
                        f"llm_relic_advice screen={screen} rec_idx={rec_idx} conf={rec_conf:.3f} reason={str(rec_reason).replace(chr(10), ' ')[:120]}",
                    )

            if screen == "CARD_SELECTION" and prev_screen != "CARD_SELECTION" and llm_advisor is not None:
                selection_cards = (raw_state.get("selection") or {}).get("cards") if isinstance((raw_state.get("selection") or {}).get("cards"), list) else []
                if selection_cards and _is_remove_selection_context(raw_state):
                    rec_idx, rec_conf, rec_reason = llm_advisor.evaluate_card_remove(raw_state, selection_cards)
                    run_logger.log(
                        "reward_shaping",
                        f"llm_remove_advice rec_idx={rec_idx} conf={rec_conf:.3f} reason={str(rec_reason).replace(chr(10), ' ')[:120]}",
                    )

            if screen == "SHOP" and prev_screen != "SHOP" and llm_advisor is not None:
                shop_items = raw_state.get("shop") or {}
                if shop_items:
                    rec_action, rec_index, rec_conf, rec_reason = llm_advisor.evaluate_shop_purchase(raw_state, shop_items)
                    run_logger.log(
                        "reward_shaping",
                        f"llm_shop_advice action={rec_action} idx={rec_index} conf={rec_conf:.3f} reason={str(rec_reason).replace(chr(10), ' ')[:120]}",
                    )

            if screen == "COMBAT" and prev_screen != "COMBAT":
                combat_step_counter = 0
                if llm_advisor is not None:
                    opening = llm_advisor.evaluate_combat_opening(raw_state)
                    run_logger.log(
                        "reward_shaping",
                        f"llm_opening_advice threat={opening.get('threat_level')} priority={opening.get('priority_action')} seq={opening.get('opening_card_sequence')}",
                    )

            # ── ② 构建 obs tensor ─────────────────────────────────────────
            obs_tensor = {
                k: torch.tensor(
                    v,
                    dtype=torch.float32 if k != "screen_type" else torch.long,
                ).unsqueeze(0).to(device)
                for k, v in obs.items()
            }

            # ── ③ 获取动作 mask ───────────────────────────────────────────
            action_mask_list = env.action_handler.get_valid_action_mask(raw_state)
            state_sig = _state_progress_signature(raw_state)
            blocked_actions = no_progress_blocked_actions.get(state_sig, set())
            forced_retry_action_id: Optional[int] = None
            if blocked_actions:
                for blocked_id in blocked_actions:
                    if 0 <= blocked_id < len(action_mask_list):
                        action_mask_list[blocked_id] = False
                if not any(action_mask_list):
                    wait_count = no_progress_wait_counts.get(state_sig, 0) + 1
                    no_progress_wait_counts[state_sig] = wait_count
                    if wait_count <= no_progress_wait_threshold:
                        run_logger.log(
                            "error_recovery",
                            f"no_progress_wait_state: screen={screen} blocked_actions={sorted(blocked_actions)} wait_count={wait_count}",
                        )
                        time.sleep(manual_poll)
                        obs, info = env.refresh_state()
                        prev_screen = screen
                        continue

                    retry_count = no_progress_retry_counts.get(state_sig, 0)
                    retry_action_id = _pick_deadlock_retry_action_id(raw_state, blocked_actions, env.action_handler)
                    if retry_action_id is not None and retry_count < no_progress_retry_limit:
                        forced_retry_action_id = retry_action_id
                        action_mask_list = [False] * len(action_mask_list)
                        if 0 <= retry_action_id < len(action_mask_list):
                            action_mask_list[retry_action_id] = True
                        no_progress_wait_counts[state_sig] = 0
                        no_progress_retry_counts[state_sig] = retry_count + 1
                        retry_payload = env.action_handler.decode(retry_action_id, raw_state)
                        run_logger.log(
                            "error_recovery",
                            f"no_progress_retry_post: screen={screen} wait_threshold_hit={wait_count} "
                            f"retry_count={retry_count + 1}/{no_progress_retry_limit} action_id={retry_action_id} payload={retry_payload}",
                        )
                    else:
                        no_progress_blocked_actions.pop(state_sig, None)
                        no_progress_wait_counts[state_sig] = 0
                        no_progress_retry_counts[state_sig] = 0
                        action_mask_list = env.action_handler.get_valid_action_mask(raw_state)
                        run_logger.log(
                            "error_recovery",
                            f"no_progress_unblock_all: screen={screen} reason=retry_limit_or_no_action "
                            f"blocked_actions={sorted(blocked_actions)}",
                        )
            else:
                no_progress_wait_counts.pop(state_sig, None)
                no_progress_retry_counts.pop(state_sig, None)

            action_mask_list, masked_discard_action_ids, auto_discard_action_id = _mask_discard_potion_actions(
                action_mask_list,
                raw_state,
                env.action_handler,
            )
            if masked_discard_action_ids:
                run_logger.log(
                    "error_recovery",
                    f"mask_discard_potion_actions: screen={screen} masked_action_ids={masked_discard_action_ids} legal={raw_state.get('legal_actions')}",
                )
            if auto_discard_action_id is not None:
                forced_retry_action_id = int(auto_discard_action_id)
                action_mask_list = [False] * len(action_mask_list)
                if 0 <= forced_retry_action_id < len(action_mask_list):
                    action_mask_list[forced_retry_action_id] = True
                run_logger.log(
                    "error_recovery",
                    f"auto_discard_potion_once: screen={screen} action_id={forced_retry_action_id} legal={raw_state.get('legal_actions')}",
                )

            action_mask_tensor = torch.tensor([action_mask_list], dtype=torch.bool).to(device)

            # ── ③.1 未知状态 / 无动作可执行 -> 人工介入 ───────────────────
            if _should_manual_intervention(raw_state, action_mask_list):
                pre_manual_record = {
                    "total_steps": total_steps,
                    "episode": episode,
                    "screen": screen,
                    "phase": raw_state.get("phase"),
                    "can_act": raw_state.get("can_act"),
                    "block_reason": raw_state.get("block_reason"),
                    "floor": raw_state.get("floor"),
                    "gold": raw_state.get("gold"),
                    "hp": (raw_state.get("combat", {}) or {}).get("player", {}).get("hp"),
                    "legal_actions": raw_state.get("legal_actions", []),
                    "valid_action_count": sum(1 for m in action_mask_list if m),
                    "trigger_reason": "unknown_screen_timeout",
                }
                run_logger.log(
                    "manual_intervention",
                    "manual_start " + json.dumps(pre_manual_record, ensure_ascii=False),
                )
                run_logger.log(
                    "error_recovery",
                    f"manual_intervention_wait_start: screen={screen} phase={raw_state.get('phase')} can_act={raw_state.get('can_act')} legal={raw_state.get('legal_actions')}",
                )
                print("🧑‍🔧 检测到 UNKNOWN 状态，需人工介入，等待人工操作后自动继续...")
                next_obs, env_reward, done, truncated, next_info = env.step_manual_intervention(
                    prev_state=raw_state,
                    max_wait=manual_max_wait,
                    poll=manual_poll,
                )
                if not bool(next_info.get("manual_intervention_changed", False)):
                    raise RuntimeError("人工介入等待超时：状态未发生变化。")

                run_logger.log(
                    "error_recovery",
                    f"manual_intervention_wait_done: delta={next_info.get('manual_state_delta')}",
                )
                post_manual_record = {
                    "total_steps": total_steps,
                    "episode": episode,
                    "changed": bool(next_info.get("manual_intervention_changed", False)),
                    "next_screen": next_info.get("screen"),
                    "next_floor": next_info.get("floor"),
                    "next_gold": next_info.get("gold"),
                    "next_hp": next_info.get("hp"),
                    "next_legal_actions": next_info.get("legal_actions", []),
                    "state_delta": next_info.get("manual_state_delta", {}),
                }
                run_logger.log(
                    "manual_intervention",
                    "manual_done " + json.dumps(post_manual_record, ensure_ascii=False),
                )
                executed_action = next_info.get("action_executed", {})
                run_logger.log(
                    "env_step_state",
                    f"post_step action={executed_action} next_screen={next_info.get('screen')} floor={next_info.get('floor')}",
                )

                shaped_reward = reward_shaper.shape(
                    base_reward=0.0,
                    prev_state=raw_state,
                    new_state=next_info.get("raw_state", {}),
                    action=executed_action,
                    done=done,
                    agent_card_index=None,
                    agent_relic_index=None,
                    agent_map_index=None,
                    agent_remove_index=None,
                    agent_shop_action=None,
                    agent_shop_index=None,
                    combat_step=combat_step_counter if screen == "COMBAT" else None,
                    agent_card_played=None,
                )
                run_logger.log(
                    "reward_shaping",
                    f"step={total_steps} action=manual_intervention action_reward={shaped_reward:.4f} "
                    f"A={reward_shaper.last_breakdown.get('A_action', 0.0):.4f} "
                    f"B={reward_shaper.last_breakdown.get('B_turn', 0.0):.4f} "
                    f"B_energy={reward_shaper.last_breakdown.get('B_energy_waste', 0.0):.4f} "
                    f"B_block={reward_shaper.last_breakdown.get('B_overflow_block', 0.0):.4f} "
                    f"B_hp={reward_shaper.last_breakdown.get('B_hp_loss', 0.0):.4f} "
                    f"B_trig={reward_shaper.last_breakdown.get('B_trigger', 0.0):.0f} "
                    f"C={reward_shaper.last_breakdown.get('C_combat', 0.0):.4f} "
                    f"D={reward_shaper.last_breakdown.get('D_meta', 0.0):.4f} "
                    f"E={reward_shaper.last_breakdown.get('E_terminal', 0.0):.4f} "
                    f"LLM={reward_shaper.last_breakdown.get('LLM_route', 0.0):.4f} "
                    f"matchCard={reward_shaper.last_breakdown.get('LLM_card', 0.0):.4f} "
                    f"matchRelic={reward_shaper.last_breakdown.get('LLM_relic', 0.0):.4f} "
                    f"matchMap={reward_shaper.last_breakdown.get('LLM_map', 0.0):.4f} "
                    f"matchRemove={reward_shaper.last_breakdown.get('LLM_remove', 0.0):.4f} "
                    f"matchShop={reward_shaper.last_breakdown.get('LLM_shop', 0.0):.4f} "
                    f"matchOpen={reward_shaper.last_breakdown.get('LLM_opening', 0.0):.4f} "
                    f"total={shaped_reward:.4f} done={done} truncated={truncated}",
                )

                # 人工介入步骤不写入 PPO buffer，避免 log_prob/value 伪造污染梯度。
                current_ep_reward += shaped_reward
                obs = next_obs
                info = next_info
                episode_max_floor = max(episode_max_floor, int(next_info.get("floor", 0) or 0))
                global_max_floor = max(global_max_floor, episode_max_floor)
                prev_screen = screen
                if screen == "COMBAT":
                    combat_step_counter += 1

                if done or truncated:
                    episode += 1
                    episode_rewards.append(current_ep_reward)
                    episode_reward_sum += current_ep_reward
                    episode_count_for_avg += 1
                    run_logger.log(
                        "episode_summary",
                        f"episode={episode} reward={episode_rewards[-1]:.4f} floor={info.get('floor', 0)} max_floor={global_max_floor} hp={info.get('hp', 0)} total_steps={total_steps}",
                    )
                    current_ep_reward = 0.0
                    obs, info = env.reset()
                    episode_max_floor = int(info.get("floor", 0) or 0)
                    prev_screen = ""
                    combat_step_counter = 0
                    llm_card_triggered = False
                    if resume_on_restart:
                        _save_resume_snapshot()
                continue

            # ── ④ Agent 决策（RL 主导 + LLM 轻量概率引导）─────────────────
            llm_guided_action_id: Optional[int] = None
            llm_guidance_conf = 0.0
            llm_guidance_source = ""
            llm_guidance_bias = 0.0

            with torch.no_grad():
                action_logits, value = agent.policy.forward(obs_tensor, action_mask_tensor)

                if (
                    llm_policy_guidance_enabled
                    and forced_retry_action_id is None
                    and screen in llm_policy_guidance_screens
                ):
                    candidate_id, candidate_conf, candidate_source = _resolve_llm_guided_action_id(
                        llm_advisor=llm_advisor,
                        screen=screen,
                        raw_state=raw_state,
                        action_mask_list=action_mask_list,
                        action_handler=env.action_handler,
                        combat_step_counter=combat_step_counter,
                        combat_guidance_steps=llm_policy_guidance_combat_steps,
                    )
                    if candidate_id is not None and float(candidate_conf) >= llm_policy_guidance_confidence_threshold:
                        ramp_ratio = min(1.0, max(0.0, total_steps / float(llm_policy_guidance_ramp_steps)))
                        effective_alpha = llm_policy_guidance_alpha_max * ramp_ratio
                        llm_guidance_bias = min(llm_policy_guidance_max_bias, max(0.0, effective_alpha * float(candidate_conf)))
                        if llm_guidance_bias > 0.0:
                            action_logits = action_logits.clone()
                            action_logits[0, int(candidate_id)] = action_logits[0, int(candidate_id)] + llm_guidance_bias
                            llm_guided_action_id = int(candidate_id)
                            llm_guidance_conf = float(candidate_conf)
                            llm_guidance_source = candidate_source

                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            action_id = action.item()
            if forced_retry_action_id is not None:
                action_id = int(forced_retry_action_id)
                llm_guided_action_id = None
                llm_guidance_conf = 0.0
                llm_guidance_source = ""
                llm_guidance_bias = 0.0
            legal_actions = [str(a) for a in (raw_state.get("legal_actions") or [])]
            combat_block = raw_state.get("combat") or {}
            hand_cards = combat_block.get("hand") or []
            playable_count = 0
            for c in hand_cards:
                if isinstance(c, dict) and bool(c.get("playable", True)):
                    playable_count += 1
            player_energy = int((combat_block.get("player") or {}).get("energy", combat_block.get("energy", 0)) or 0)
            run_logger.log(
                "agent_decision",
                f"step={total_steps} screen={screen} action_id={action_id} "
                f"log_prob={log_prob.item():.4f} value={value.item():.4f} "
                f"valid_actions={sum(1 for m in action_mask_list if m)} "
                f"forced_retry={forced_retry_action_id is not None} "
                f"llm_guided={llm_guided_action_id is not None} "
                f"llm_source={llm_guidance_source or '-'} "
                f"llm_conf={llm_guidance_conf:.3f} "
                f"llm_bias={llm_guidance_bias:.3f} "
                f"legal={legal_actions} hand={len(hand_cards)} playable={playable_count} energy={player_energy}",
            )

            # ── ⑤ 执行动作 ───────────────────────────────────────────────
            try:
                next_obs, env_reward, done, truncated, next_info = env.step(action_id)
                consecutive_env_errors = 0
            except RuntimeError as e:
                # 方案2：环境错误容错恢复，避免单次非法动作导致整轮训练崩溃
                msg = str(e)
                recoverable = (
                    "No unlocked event options available" in msg
                    or "status" in msg.lower()
                    or "Invalid" in msg
                )
                if recoverable:
                    run_logger.log("error_recovery", f"recoverable_runtime_error: {msg}")
                    consecutive_env_errors += 1
                    if consecutive_env_errors >= max_consecutive_env_errors:
                        raise RuntimeError(
                            f"连续环境错误达到上限: {msg}"
                        ) from e
                    time.sleep(manual_poll)
                    obs, info = env.refresh_state()
                    continue
                raise
            executed_action = next_info.get("action_executed", {})
            run_logger.log(
                "env_step_state",
                f"post_step action={executed_action} next_screen={next_info.get('screen')} floor={next_info.get('floor')}",
            )
            manual_intervention = bool(next_info.get("manual_intervention", False))
            if manual_intervention:
                reason = next_info.get("manual_intervention_reason", "unknown")
                run_logger.log("error_recovery", f"manual_intervention: reason={reason}")
                print(
                    f"🧑‍🔧 检测到人工介入步骤（{reason}），本步不计入 buffer/奖励，继续后续状态。"
                )
                no_progress_blocked_actions.pop(state_sig, None)
                no_progress_wait_counts.pop(state_sig, None)
                no_progress_retry_counts.pop(state_sig, None)
                obs = next_obs
                info = next_info
                continue

            if _is_no_progress_step(raw_state, next_info.get("raw_state", {}), done, truncated):
                blocked = no_progress_blocked_actions.setdefault(state_sig, set())
                blocked.add(int(action_id))
                no_progress_wait_counts[state_sig] = 0
                if len(no_progress_blocked_actions) > max_no_progress_states:
                    no_progress_blocked_actions.clear()
                    no_progress_wait_counts.clear()
                    no_progress_retry_counts.clear()
                run_logger.log(
                    "error_recovery",
                    f"no_progress_skip: screen={screen} action={executed_action.get('action', executed_action.get('type'))} "
                    f"action_id={action_id} blocked_actions={sorted(blocked)} legal={raw_state.get('legal_actions')}",
                )
                time.sleep(manual_poll)
                obs, info = env.refresh_state()
                prev_screen = screen
                continue
            no_progress_blocked_actions.pop(state_sig, None)
            no_progress_wait_counts.pop(state_sig, None)
            no_progress_retry_counts.pop(state_sig, None)

            # ── ⑥ 解析 agent 是否做了选牌动作 ────────────────────────────
            agent_card_index = _extract_agent_card_index(executed_action, screen)
            agent_relic_index = _extract_agent_relic_index(executed_action)
            agent_map_index = _extract_agent_map_index(executed_action)
            agent_remove_index = _extract_agent_remove_index(executed_action, screen)
            agent_shop_action, agent_shop_index = _extract_agent_shop_choice(executed_action)
            combat_card_played = _extract_combat_card_played(executed_action)

            # ── ⑦ 奖励塑形（含 match bonus）──────────────────────────────
            shaped_reward = reward_shaper.shape(
                base_reward=0.0,
                prev_state=raw_state,
                new_state=next_info.get("raw_state", {}),
                action=executed_action,
                done=done,
                agent_card_index=agent_card_index,   # ← 新增传参
                agent_relic_index=agent_relic_index,
                agent_map_index=agent_map_index,
                agent_remove_index=agent_remove_index,
                agent_shop_action=agent_shop_action,
                agent_shop_index=agent_shop_index,
                combat_step=combat_step_counter if screen == "COMBAT" else None,
                agent_card_played=combat_card_played,
            )
            run_logger.log(
                "reward_shaping",
                f"step={total_steps} action={executed_action.get('action', executed_action.get('type'))} "
                f"action_reward={shaped_reward:.4f} "
                f"A={reward_shaper.last_breakdown.get('A_action', 0.0):.4f} "
                f"B={reward_shaper.last_breakdown.get('B_turn', 0.0):.4f} "
                f"B_energy={reward_shaper.last_breakdown.get('B_energy_waste', 0.0):.4f} "
                f"B_block={reward_shaper.last_breakdown.get('B_overflow_block', 0.0):.4f} "
                f"B_hp={reward_shaper.last_breakdown.get('B_hp_loss', 0.0):.4f} "
                f"B_trig={reward_shaper.last_breakdown.get('B_trigger', 0.0):.0f} "
                f"C={reward_shaper.last_breakdown.get('C_combat', 0.0):.4f} "
                f"D={reward_shaper.last_breakdown.get('D_meta', 0.0):.4f} "
                f"E={reward_shaper.last_breakdown.get('E_terminal', 0.0):.4f} "
                f"LLM={reward_shaper.last_breakdown.get('LLM_route', 0.0):.4f} "
                f"matchCard={reward_shaper.last_breakdown.get('LLM_card', 0.0):.4f} "
                f"matchRelic={reward_shaper.last_breakdown.get('LLM_relic', 0.0):.4f} "
                f"matchMap={reward_shaper.last_breakdown.get('LLM_map', 0.0):.4f} "
                f"matchRemove={reward_shaper.last_breakdown.get('LLM_remove', 0.0):.4f} "
                f"matchShop={reward_shaper.last_breakdown.get('LLM_shop', 0.0):.4f} "
                f"matchOpen={reward_shaper.last_breakdown.get('LLM_opening', 0.0):.4f} "
                f"total={shaped_reward:.4f} done={done} truncated={truncated}",
            )

            # ── ⑧ 写入 buffer ─────────────────────────────────────────────
            buffer.add(
                obs=obs,
                action=action_id,
                log_prob=log_prob.item(),
                reward=shaped_reward,
                done=done or truncated,
                value=value.item(),
                action_mask=action_mask_list,
            )

            current_ep_reward += shaped_reward
            total_steps += 1
            obs                = next_obs
            info               = next_info
            episode_max_floor = max(episode_max_floor, int(next_info.get("floor", 0) or 0))
            global_max_floor = max(global_max_floor, episode_max_floor)
            if screen == "COMBAT":
                combat_step_counter += 1
            prev_screen = screen

            if resume_on_restart:
                if not (done or truncated):
                    _save_resume_snapshot()
                torch.save(buffer.export_state(), pending_buffer_path)

            if done or truncated:
                episode += 1
                episode_rewards.append(current_ep_reward)
                episode_reward_sum += current_ep_reward
                episode_count_for_avg += 1
                run_logger.log(
                    "episode_summary",
                    f"episode={episode} reward={episode_rewards[-1]:.4f} floor={info.get('floor', 0)} max_floor={global_max_floor} hp={info.get('hp', 0)} total_steps={total_steps}",
                )
                current_ep_reward = 0.0
                floor_r = info.get("floor", 0)
                hp_r    = info.get("hp", 0)
                print(
                    f"  Episode {episode:4d} | "
                    f"Reward: {episode_rewards[-1]:8.2f} | "
                    f"Floor: {floor_r:2d} | "
                    f"MaxFloor: {global_max_floor:2d} | "
                    f"HP: {hp_r:3d} | "
                    f"Steps: {total_steps}"
                )
                obs, info = env.reset()
                episode_max_floor = int(info.get("floor", 0) or 0)
                prev_screen = ""
                combat_step_counter = 0
                llm_card_triggered = False
                if resume_on_restart:
                    _save_resume_snapshot()

        # ── GAE ───────────────────────────────────────────────────────────
        obs_t = {
            k: torch.tensor(
                v, dtype=torch.float32 if k != "screen_type" else torch.long
            ).unsqueeze(0).to(device)
            for k, v in obs.items()
        }
        with torch.no_grad():
            _, last_value = agent.policy.forward(obs_t)
        last_value = last_value.item()

        advantages, returns = agent.compute_gae(
            rewards=buffer.rewards,
            values=buffer.values,
            dones=buffer.dones,
            last_value=last_value,
        )
        buffer.set_gae_results(advantages, returns)

        # ── PPO 更新 ──────────────────────────────────────────────────────
        metrics = agent.update(
            buffer,
            n_epochs=cfg["train"]["n_epochs"],
            batch_size=cfg["train"]["batch_size"],
        )
        
        run_logger.log(
            "ppo_update",
            f"update_at_steps={total_steps} batch_size={cfg['train']['batch_size']} "
            f"buffer_size={cfg['train']['buffer_size']} pg_loss={metrics['pg_loss']:.6f} "
            f"vf_loss={metrics['vf_loss']:.6f} entropy={metrics['entropy']:.6f} "
            f"|| layer_weights: A={adjusted_weights['layer_a']:.2f} B={adjusted_weights['layer_b']:.2f} "
            f"C={adjusted_weights['layer_c']:.2f} D={adjusted_weights['layer_d']:.2f} E={adjusted_weights['layer_e']:.2f}"
        )
        update_count += 1
        pg_loss_sum += float(metrics["pg_loss"])
        vf_loss_sum += float(metrics["vf_loss"])
        entropy_sum += float(metrics["entropy"])

        global_avg_pg = pg_loss_sum / update_count
        global_avg_vf = vf_loss_sum / update_count
        global_avg_ent = entropy_sum / update_count
        if save_latest_per_update:
            agent.save(latest_ckpt_path)
            effective_resume = latest_ckpt_path
            run_logger.log("ppo_update", f"latest_model_saved: {latest_ckpt_path}")
            if os.path.exists(pending_buffer_path):
                os.remove(pending_buffer_path)
            if resume_on_restart:
                _save_resume_snapshot()

        # ── 日志 & 存档 ───────────────────────────────────────────────────
        rollout_avg_ep_r = (sum(episode_rewards) / len(episode_rewards)) if episode_rewards else None
        rollout_avg_ep_r_text = f"{rollout_avg_ep_r:.3f}" if rollout_avg_ep_r is not None else "n/a"
        rollout_step_avg_r = (sum(buffer.rewards) / len(buffer.rewards)) if len(buffer.rewards) > 0 else 0.0
        global_avg_ep_r = (episode_reward_sum / episode_count_for_avg) if episode_count_for_avg > 0 else 0.0

        print(
            f"\n📊 step {total_steps:7d} | rollout_avg_ep_r: {rollout_avg_ep_r_text} | "
            f"global_avg_ep_r: {global_avg_ep_r:.3f} | "
            f"rollout_step_avg_r: {rollout_step_avg_r:.3f} | "
            f"episodes_in_rollout: {len(episode_rewards)} | episodes_total: {episode_count_for_avg} | "
            f"pg_cur: {metrics['pg_loss']:.4f} | vf_cur: {metrics['vf_loss']:.4f} | ent_cur: {metrics['entropy']:.4f} | "
            f"pg_avg: {global_avg_pg:.4f} | vf_avg: {global_avg_vf:.4f} | ent_avg: {global_avg_ent:.4f}\n"
        )

        if rollout_avg_ep_r is not None and rollout_avg_ep_r > best_episode_reward:
            best_episode_reward = rollout_avg_ep_r
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            agent.save(best_path)
            print(f"  💾 最优模型: {best_path} (rollout_avg_ep_r={rollout_avg_ep_r:.3f})")
            if resume_on_restart:
                _save_resume_snapshot()

        if total_steps % cfg["train"].get("save_interval", 50000) == 0:
            ckpt = os.path.join(checkpoint_dir, f"checkpoint_{total_steps}.pt")
            agent.save(ckpt)
            print(f"  💾 存档: {ckpt}")
            run_logger.log("ppo_update", f"checkpoint_saved: {ckpt}")
            effective_resume = ckpt
            if resume_on_restart:
                _save_resume_snapshot()

    print(f"\n🎉 训练完成! 总步数: {total_steps}")
    if resume_on_restart:
        _save_resume_snapshot()
    env.close()


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STS2 RL Agent 训练")
    parser.add_argument(
        "--config", type=str, default="ppo_default.yaml",
        help="YAML 配置路径",
    )
    parser.add_argument("--render",  action="store_true")
    parser.add_argument("--resume",  type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["__config_dir__"] = os.path.dirname(os.path.abspath(args.config))
    if args.render:
        cfg["render"] = True
    if args.resume:
        cfg["resume"] = args.resume

    train(cfg)
