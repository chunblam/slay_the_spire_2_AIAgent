"""
Deterministic evaluation entry for STS2 PPO policy.

Usage:
  python evaluate.py --config ppo_sts2agent.yaml --model checkpoints/best_model.pt --episodes 3
"""

import argparse
import json
import os
from typing import Dict, Tuple

import torch
import yaml

from ppo_agent import PPOAgent, STS2PolicyNet
from sts2_env import STS2Env


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def build_eval_agent(cfg: Dict, device: str) -> PPOAgent:
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


def to_obs_tensor(obs: Dict, device: str) -> Dict[str, torch.Tensor]:
    return {
        k: torch.tensor(v, dtype=torch.float32 if k != "screen_type" else torch.long)
        .unsqueeze(0)
        .to(device)
        for k, v in obs.items()
    }


def pick_action(
    agent: PPOAgent,
    obs: Dict,
    action_mask_list: list,
    device: str,
) -> Tuple[int, float]:
    obs_tensor = to_obs_tensor(obs, device)
    mask_tensor = torch.tensor([action_mask_list], dtype=torch.bool).to(device)
    with torch.no_grad():
        action, _, value = agent.policy.get_action(
            obs_tensor,
            action_mask=mask_tensor,
            deterministic=True,
        )
    return int(action.item()), float(value.item())


def top_actions(action_counter: Dict[str, int], k: int = 5) -> str:
    if not action_counter:
        return "none"
    ranked = sorted(action_counter.items(), key=lambda x: x[1], reverse=True)[:k]
    return ", ".join([f"{name}:{count}" for name, count in ranked])


def run_evaluation(cfg: Dict, model_path: str, episodes: int, max_steps_per_episode: int, device: str):
    env = STS2Env(
        host=cfg["env"].get("host", "localhost"),
        port=cfg["env"].get("port", 18080),
        character_index=cfg["env"].get("character_index", 0),
        startup_debug=cfg["env"].get("startup_debug", False),
        action_poll_interval=cfg["env"].get("action_poll_interval", 0.5),
        action_min_interval=cfg["env"].get("action_min_interval", 0.5),
        post_action_settle=cfg["env"].get("post_action_settle", 0.5),
        action_retry_count=cfg["env"].get("action_retry_count", 1),
        render_mode="human" if cfg.get("render", False) else None,
    )

    configured_actions = int(cfg["env"].get("num_actions", env.action_space.n))
    if configured_actions != int(env.action_space.n):
        raise ValueError(
            f"env.num_actions({configured_actions}) != action_space.n({env.action_space.n})"
        )

    agent = build_eval_agent(cfg, device)
    agent.load(model_path)
    agent.policy.eval()

    results = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        ep_max_floor = int(info.get("floor", 0) or 0)
        action_counter: Dict[str, int] = {}
        manual_interventions = 0

        done = False
        truncated = False

        while ep_steps < max_steps_per_episode:
            raw_state = info.get("raw_state", {})
            action_mask_list = env.action_handler.get_valid_action_mask(raw_state)

            screen = str(raw_state.get("screen", "") or "").upper()
            if screen == "UNKNOWN" or not any(action_mask_list):
                next_obs, reward, done, truncated, next_info = env.step_manual_intervention(
                    prev_state=raw_state,
                    max_wait=float(cfg.get("train", {}).get("manual_intervention_max_wait", 180.0)),
                    poll=float(cfg.get("train", {}).get("manual_intervention_poll", 0.5)),
                )
                manual_interventions += 1
                ep_reward += float(reward)
                ep_steps += 1
                ep_max_floor = max(ep_max_floor, int(next_info.get("floor", 0) or 0))
                obs, info = next_obs, next_info
                if done or truncated:
                    break
                continue

            action_id, value_est = pick_action(agent, obs, action_mask_list, device)
            next_obs, reward, done, truncated, next_info = env.step(action_id)

            executed = next_info.get("action_executed", {})
            action_name = str(executed.get("action", "unknown"))
            action_counter[action_name] = action_counter.get(action_name, 0) + 1

            ep_reward += float(reward)
            ep_steps += 1
            ep_max_floor = max(ep_max_floor, int(next_info.get("floor", 0) or 0))
            obs, info = next_obs, next_info

            if done or truncated:
                break

            _ = value_est

        last_raw = info.get("raw_state", {})
        game_over = last_raw.get("game_over") or {}
        victory = bool(game_over.get("victory", False))
        defeat = bool(game_over.get("defeat", False))

        row = {
            "episode": ep,
            "reward": ep_reward,
            "steps": ep_steps,
            "floor": int(info.get("floor", 0) or 0),
            "max_floor": ep_max_floor,
            "hp": int(info.get("hp", 0) or 0),
            "victory": victory,
            "defeat": defeat,
            "done": bool(done),
            "truncated": bool(truncated),
            "manual_interventions": manual_interventions,
            "top_actions": top_actions(action_counter),
        }
        results.append(row)

        print(
            f"Episode {ep:02d} | reward={row['reward']:.3f} | floor={row['floor']} "
            f"| max_floor={row['max_floor']} | hp={row['hp']} | steps={row['steps']} "
            f"| victory={row['victory']} | manual={row['manual_interventions']}"
        )
        print(f"  top_actions: {row['top_actions']}")

    env.close()

    mean_reward = sum(r["reward"] for r in results) / max(len(results), 1)
    mean_max_floor = sum(r["max_floor"] for r in results) / max(len(results), 1)
    mean_steps = sum(r["steps"] for r in results) / max(len(results), 1)
    victories = sum(1 for r in results if r["victory"])
    total_manual = sum(r["manual_interventions"] for r in results)

    summary = {
        "episodes": len(results),
        "mean_reward": mean_reward,
        "mean_max_floor": mean_max_floor,
        "mean_steps": mean_steps,
        "victories": victories,
        "win_rate": victories / max(len(results), 1),
        "total_manual_interventions": total_manual,
        "model": model_path,
    }

    print("\nEvaluation Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained STS2 PPO policy")
    parser.add_argument("--config", type=str, default="ppo_sts2agent.yaml", help="YAML config path")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt", help="Checkpoint path")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--max-steps-per-episode", type=int, default=3000, help="Step cap per episode")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = cfg.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"model checkpoint not found: {args.model}")

    run_evaluation(
        cfg=cfg,
        model_path=args.model,
        episodes=max(1, int(args.episodes)),
        max_steps_per_episode=max(1, int(args.max_steps_per_episode)),
        device=device,
    )
