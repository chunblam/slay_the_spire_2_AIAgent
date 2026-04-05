"""
src/agent/ppo_agent.py

PPO Agent with Transformer-based State Encoder
核心 RL 模型，负责出牌决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple
import numpy as np


# ─── 网络模块 ──────────────────────────────────────────────────────────────────

class CardEncoder(nn.Module):
    """
    用 Transformer 编码手牌序列
    - 每张卡 8 维特征
    - 输出每张卡的上下文感知表示
    """

    def __init__(self, card_dim: int = 8, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(card_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, hand: torch.Tensor, hand_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hand: [B, MAX_HAND, 8]
            hand_mask: [B, MAX_HAND], 1=有效卡
        Returns:
            [B, MAX_HAND, d_model]
        """
        x = self.proj(hand)  # [B, MAX_HAND, d_model]
        # Transformer: key_padding_mask=True 表示忽略该位置
        pad_mask = (hand_mask == 0)  # [B, MAX_HAND]
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return x


class MonsterEncoder(nn.Module):
    """编码敌人状态"""

    def __init__(self, monster_dim: int = 8, d_model: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(monster_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.d_model = d_model

    def forward(self, monsters: torch.Tensor) -> torch.Tensor:
        """
        Args: monsters [B, MAX_MONSTERS, 8]
        Returns: [B, MAX_MONSTERS * d_model] (展平)
        """
        B = monsters.shape[0]
        out = self.net(monsters)  # [B, MAX_MONSTERS, d_model]
        return out.view(B, -1)


class STS2PolicyNet(nn.Module):
    """
    Actor-Critic 网络

    输入: 游戏状态（各子模块编码后拼接）
    输出:
        - action_logits: [B, num_actions]  (Actor)
        - value: [B, 1]                    (Critic)
    """

    def __init__(
        self,
        num_actions: int = 16,
        player_dim: int = 10,
        deck_dim: int = 8,
        relic_dim: int = 30,
        card_d_model: int = 64,
        monster_d_model: int = 32,
        max_hand: int = 10,
        max_monsters: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_actions = num_actions

        # 子编码器
        self.card_encoder = CardEncoder(d_model=card_d_model)
        self.monster_encoder = MonsterEncoder(d_model=monster_d_model)

        # 全局状态特征维度
        # card_pool (取均值): card_d_model
        # monster_pool: max_monsters * monster_d_model
        # player: player_dim
        # deck: deck_dim
        # relics: relic_dim
        global_dim = card_d_model + max_monsters * monster_d_model + player_dim + deck_dim + relic_dim

        # 共享 backbone
        self.backbone = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Actor head: per-action logits
        # 先生成全局特征，再和每张卡的特征做 dot-product 得到 hand 动作分数
        self.actor_hand = nn.Linear(hidden_dim + card_d_model, 1)  # 对每张卡打分
        self.actor_other = nn.Linear(hidden_dim, num_actions - max_hand)  # 其余动作

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

        self.max_hand = max_hand

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_logits: [B, num_actions]
            value: [B]
        """
        B = obs["player"].shape[0]

        # 编码各子状态
        card_repr = self.card_encoder(obs["hand"], obs["hand_mask"])  # [B, MAX_HAND, card_d]
        card_global = card_repr.mean(dim=1)                           # [B, card_d]
        monster_repr = self.monster_encoder(obs["monsters"])          # [B, monsters_flat]

        # 拼接全局特征
        global_feat = torch.cat([
            card_global,
            monster_repr,
            obs["player"],
            obs["deck_stats"],
            obs["relics"],
        ], dim=-1)  # [B, global_dim]

        shared = self.backbone(global_feat)  # [B, hidden_dim]

        # Actor: 手牌动作分
        shared_exp = shared.unsqueeze(1).expand(-1, self.max_hand, -1)  # [B, max_hand, hidden]
        hand_input = torch.cat([shared_exp, card_repr], dim=-1)          # [B, max_hand, h+card_d]
        hand_logits = self.actor_hand(hand_input).squeeze(-1)             # [B, max_hand]

        # Actor: 其他动作分
        other_logits = self.actor_other(shared)  # [B, num_actions - max_hand]

        # 拼接所有动作 logits
        action_logits = torch.cat([hand_logits, other_logits], dim=-1)  # [B, num_actions]

        # 应用 action mask（将无效动作设为 -inf）
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask.bool(), float('-inf'))

        # Critic
        value = self.critic(shared).squeeze(-1)  # [B]

        return action_logits, value

    def get_action(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作

        Returns:
            action: [B]
            log_prob: [B]
            value: [B]
        """
        logits, value = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估已有动作（用于 PPO update）

        Returns:
            log_prob: [B]
            value: [B]
            entropy: [B]
        """
        logits, value = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy


# ─── PPO 训练逻辑 ──────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO Agent 封装

    超参数参考:
    - clip_eps: 0.2
    - value_loss_coef: 0.5
    - entropy_coef: 0.01
    - gamma: 0.99
    - gae_lambda: 0.95
    """

    def __init__(
        self,
        policy: STS2PolicyNet,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.device = device

        self.clip_eps = clip_eps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float,
    ) -> Tuple[List[float], List[float]]:
        """
        广义优势估计 (GAE)
        返回 (advantages, returns)
        """
        advantages = []
        gae = 0.0
        values_ext = values + [last_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(self, rollout_buffer, n_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """
        执行 PPO 更新

        Args:
            rollout_buffer: RolloutBuffer 实例，包含一轮收集的数据
            n_epochs: 每批数据重复训练的轮数
            batch_size: mini-batch 大小
        Returns:
            训练指标字典
        """
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        obs_tensors, actions, old_log_probs, advantages, returns, masks = rollout_buffer.get_tensors(self.device)

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = actions.shape[0]
        indices = torch.randperm(N)

        for epoch in range(n_epochs):
            for start in range(0, N, batch_size):
                idx = indices[start: start + batch_size]
                batch_obs = {k: v[idx] for k, v in obs_tensors.items()}
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_masks = masks[idx] if masks is not None else None

                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_obs, batch_actions, batch_masks
                )

                # Policy gradient loss (clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                pg_loss1 = -batch_advantages * ratio
                pg_loss2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value function loss
                vf_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + self.value_loss_coef * vf_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "pg_loss": total_pg_loss / n_updates,
            "vf_loss": total_vf_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "policy_config": {
                "num_actions": self.policy.num_actions,
                "max_hand": self.policy.max_hand,
            },
        }, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"], strict=strict)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
