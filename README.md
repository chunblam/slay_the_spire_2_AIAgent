# STS2 RL Agent (STS2AIAgent 适配版)

杀戮尖塔 2 的 PPO 强化学习项目，支持两种训练模式：

1. 单 RL：仅依赖环境奖励塑形（不调用 LLM）
2. RL+LLM：在奖励塑形基础上，增加 LLM 建议的概率轻引导（logit bias）

本 README 已按当前代码对齐，适用于本仓库实际运行方式。

---

## 1. 项目定位

当前仓库聚焦以下目标：

1. 用 PPO 在 STS2AIAgent HTTP 接口上稳定训练可持续提升的策略
2. 用结构化奖励（Layer A~E）提供可解释学习信号
3. 在不破坏已训练模型兼容性的前提下，渐进接入 LLM 引导
4. 支持中断恢复、日志延续、自动保存 latest/best checkpoint

---

## 2. 当前架构

### 2.1 模块分层

1. 环境层：`sts2_env.py`
2. 动作层：`action_space.py`
3. 编码层：`state_encoder.py`
4. 策略层：`ppo_agent.py` + `train.py`
5. 奖励层：`reward_shaper.py`
6. 顾问层：`llm_advisor.py`
7. 评估脚本：`evaluate.py`

### 2.2 数据流

1. `STS2Env` 拉取 session state + legal actions
2. `StateEncoder` 将 JSON 编码为固定形状观测
3. `PPO policy` 结合 action mask 采样动作
4. `RewardShaper` 合成训练奖励
5. 回放到 `RolloutBuffer`，周期性 PPO update

---

## 3. 近期关键更新（已落地）

### 3.1 关键词特征接入（保持模型兼容）

1. 从 STS2AIAgent 手牌 payload 透传 `key_words`
2. 在 `state_encoder.py` 中将关键词聚合为 `keyword_signal`
3. 保持 `CARD_FEATURE_DIM=8` 不变，避免旧 checkpoint 失效

### 3.2 丢药策略修正

1. 不再永久禁用 `discard_potion`
2. 当药水槽已满时，训练循环会自动执行一次丢药动作
3. 当仅剩 `discard_potion` 且槽未满时，进入等待态，避免无意义动作

### 3.3 RL+LLM 双引导

1. 奖励引导：保留 card/relic/map/remove/shop/opening 的匹配奖励
2. 概率引导：在采样前对 LLM 推荐动作做 logit 轻偏置
3. 安全约束：
   - 仅在指定 screen 生效
   - 仅在置信度超过阈值时生效
   - 偏置随训练步数 ramp-up
   - 永远服从 action mask

---

## 4. 环境与依赖

### 4.1 Python 依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 游戏侧要求

1. 安装并启用 STS2AIAgent Mod
2. 默认端口：`127.0.0.1:18080`
3. 训练前先在游戏内进入可交互状态

### 4.3 连接测试

```bash
python test_connection.py --host 127.0.0.1 --port 18080
```

成功时会打印 phase/screen/can_act/legal_actions。

---

## 5. 配置说明（ppo_sts2agent.yaml）

主配置文件：`ppo_sts2agent.yaml`

### 5.1 训练核心参数

1. `train.total_steps`: 总训练步数
2. `train.buffer_size`: rollout 缓冲区大小
3. `train.n_epochs`: 每次 update 的 PPO epoch 数
4. `train.entropy_coef`: 探索强度
5. `train.resume_on_restart`: 自动断点续训
6. `train.save_latest_per_update`: 每次 update 保存 latest

### 5.2 LLM 参数（当前默认可启用）

1. `llm.enabled`: 是否启用 LLM
2. `llm.backend/model/base_url`: 后端与模型
3. `llm.call_interval_steps`: 调用节流
4. `llm.cache_ttl`: 缓存 TTL
5. `llm.confidence_threshold`: 建议有效阈值

### 5.3 概率引导参数

1. `llm.policy_guidance_enabled`
2. `llm.policy_guidance_alpha_max`
3. `llm.policy_guidance_max_bias`
4. `llm.policy_guidance_confidence_threshold`
5. `llm.policy_guidance_ramp_steps`
6. `llm.policy_guidance_screens`
7. `llm.policy_guidance_combat_steps`

### 5.4 奖励参数

`reward` 段已实现 Layer A/B/C/D/E + LLM match。

可按实验重点调整：

1. `reward.llm_weight`
2. `reward.card_weight / relic_choice_weight / map_route_weight`
3. `reward.remove_choice_weight / shop_choice_weight / combat_opening_weight`
4. `reward.phase_schedule.*`（分阶段权重调度）

---

## 6. 运行方式

### 6.1 单 RL 训练（建议先做基线）

将配置改为：

1. `llm.enabled: false`

然后运行：

```bash
python train.py --config ppo_sts2agent.yaml
```

### 6.2 RL+LLM 训练

将配置改为：

1. `llm.enabled: true`
2. 正确设置 `api_key`（推荐用环境变量）

然后运行：

```bash
python train.py --config ppo_sts2agent.yaml
```

### 6.3 从 checkpoint 继续

`train.py` 已支持自动续训（默认读取 `checkpoints/training_state.json`）。

可直接重启同一命令继续训练。

---

## 7. Checkpoint 与日志

### 7.1 关键文件

1. `checkpoints/latest_model.pt`
2. `checkpoints/best_model.pt`
3. `checkpoints/training_state.json`
4. `checkpoints/pending_buffer.pt`

### 7.2 日志目录

1. `logs/<run_id>/module_episode_summary.log`
2. `logs/<run_id>/module_ppo_update.log`
3. `logs/<run_id>/module_reward_shaping.log`
4. `logs/<run_id>/module_error_recovery.log`

训练中可重点观察：

1. rollout 平均奖励变化
2. max_floor 是否持续抬升
3. entropy 是否过快塌陷
4. no_progress / manual_intervention 是否异常增多

---

## 8. 评估脚本

使用确定性策略评估：

```bash
python evaluate.py --config ppo_sts2agent.yaml --model checkpoints/best_model.pt --episodes 3
```

可替换 `--model checkpoints/latest_model.pt` 做最近策略快检。

输出包含：

1. 每局 reward / floor / max_floor / hp / steps
2. victory / defeat
3. top_actions
4. 汇总 JSON（均值奖励、均值层数、胜率）

---

## 9. LLM 接口建议

### 9.1 默认推荐

1. 后端：OpenAI 兼容
2. 模型：`deepseek-chat`
3. base_url：`https://api.deepseek.com/v1`

### 9.2 环境变量

Windows PowerShell 示例：

```powershell
$env:DEEPSEEK_API_KEY="你的key"
```

说明：`llm_advisor.py` 会优先读取配置项 `api_key`，否则回退环境变量。

---

## 10. 常见问题

### 10.1 连接失败

1. 确认游戏已启动且 Mod 已启用
2. 确认端口与 `ppo_sts2agent.yaml` 一致
3. 先跑 `python test_connection.py`

### 10.2 训练卡在异常状态

1. 查看 `module_error_recovery.log`
2. 关注 UNKNOWN screen、no_progress、manual_intervention
3. 必要时调大 `manual_intervention_max_wait`

### 10.3 LLM 请求异常

1. 检查 API key
2. 检查 base_url 与模型名
3. 临时切回 `llm.enabled: false` 验证 RL 主链路

---

## 11. 推荐实验顺序

1. 先跑单 RL 基线（5k~20k 步）
2. 同 checkpoint 开 RL+LLM 小权重分支
3. 先开奖励引导，再开概率引导
4. 对比同窗口指标后再放大引导强度

---

## 12. 相关文档

1. RL+LLM 全链路说明：`docs/4.2/rl_llm_pipeline_and_guided_policy.md`
2. API 调用文档：`docs/other/STS2AIAgent-API-中文调用文档.md`
3. 奖励与调试文档：`docs/other/Reward-v2- 参数与调试速查.md`

---

## 13. 快速命令清单

```bash
# 1) 依赖
pip install -r requirements.txt

# 2) 连接测试
python test_connection.py --host 127.0.0.1 --port 18080

# 3) 训练
python train.py --config ppo_sts2agent.yaml

# 4) 评估
python evaluate.py --config ppo_sts2agent.yaml --model checkpoints/best_model.pt --episodes 3
```
