# STS2 RL Agent (STS2AIAgent 适配版)

本项目用于在《杀戮尖塔2》上训练 PPO 策略，并在现有版本中支持两种模式：

1. 单 RL：只使用环境状态与奖励塑形。
2. RL+LLM：在单 RL 基础上叠加 LLM 双引导（奖励引导 + 策略轻引导）。

当前 README 已按代码现状对齐，重点说明 RL+LLM 方案与架构。

---

## 1. 项目目标

1. 在 STS2AIAgent HTTP 接口上进行稳定可恢复训练。
2. 通过分层奖励（Layer A~E）提升学习信号密度与可解释性。
3. 在不破坏单RL训练链路的前提下，引入 LLM 决策信息。
4. 支持断点续训、日志延续、latest/best 自动保存。

---

## 2. 模式与隔离策略

当前仓库已把单RL与RL+LLM分离为不同配置入口：

| 模式 | 配置文件 | scheme | llm.enabled | checkpoint_dir |
| --- | --- | --- | --- | --- |
| 单RL | ppo_sts2agent_rl.yaml | rl | false | checkpoints_rl |
| RL+LLM | ppo_sts2agent.yaml | rl_llm | true | checkpoints_rl_llm |

关键说明：

1. train.py 会根据 scheme 分流，且在单RL模式下强制把所有 LLM 相关奖励权重置 0。
2. 只要你用 ppo_sts2agent_rl.yaml 运行，RL+LLM 的改动不会进入单RL训练分支。
3. checkpoints 按配置目录隔离；RL+LLM 与单RL不会写到同一个 checkpoint_dir。
4. logs 默认按时间戳建新目录；若开启续训并允许续接日志，会复用上一次 run 目录（这是预期行为）。

---

## 3. RL+LLM 架构总览

### 3.1 模块分层

1. 环境层：sts2_env.py
2. 动作层：action_space.py
3. 编码层：state_encoder.py
4. 策略层：ppo_agent.py + train.py
5. 奖励层：reward_shaper.py
6. LLM顾问层：llm_advisor.py
7. 评估层：evaluate.py

### 3.2 主数据流

1. STS2Env 拉取 state + legal_actions。
2. StateEncoder 将 JSON 状态编码为固定观测张量。
3. PPO policy 在 action mask 约束下采样动作。
4. env.step 执行动作并返回 next_state。
5. RewardShaper 计算 shaped_reward（A~E + LLM相关项）。
6. RolloutBuffer 收集样本，周期性 PPO update。

---

## 4. RL+LLM 的“双引导”机制

RL+LLM并不是把动作完全交给 LLM，而是将 LLM 作为可控辅助信号：

1. 奖励引导：LLM 参与奖励塑形，影响回报。
2. 策略引导：对 LLM 推荐动作进行 logit 轻偏置，再由策略采样。

### 4.1 奖励引导（RewardShaper）

RewardShaper 内部有两类 LLM 信号：

1. 全局路线信号：LLM_route（来自 get_reward_shaping_bonus）。
2. 场景匹配信号：LLM_card / LLM_event / LLM_shop / LLM_relic / LLM_map / LLM_remove / LLM_opening。

总奖励是 A~E 层与各 LLM 项按权重加和，再做 reward_clip。

### 4.2 策略轻引导（Policy Guidance）

在 train 采样阶段：

1. 先由策略网络输出 masked logits。
2. 若命中允许场景、置信度达阈值，则给候选动作加小幅 bias。
3. bias 强度受 ramp_steps 渐进控制。
4. 始终服从 action mask，不会越权执行非法动作。

这使得 RL 仍是主控制器，LLM只做温和引导。

---

## 5. LLM 顾问当前覆盖场景

llm_advisor.py 已覆盖以下决策场景：

1. 选牌：evaluate_card_selection / evaluate_card_reward
2. 事件：evaluate_event_choice
3. 商店：evaluate_shop_purchase
4. 战斗开局：evaluate_combat_opening
5. 战斗回合：evaluate_combat_turn
6. 地图路线：evaluate_map_route（是否使用由配置开关控制）
7. 删牌：evaluate_card_remove

其内部带有：

1. 调用节流（call_interval_steps）
2. 缓存TTL（cache_ttl）
3. 置信度阈值（confidence_threshold）
4. 异常 fallback（调用失败时返回降级建议，不中断训练）

---

## 6. 稳定性与故障恢复机制

train.py 中已经实现多层保护，避免游戏流程异常时直接崩盘：

1. no-progress 检测：相同状态签名下屏蔽无进展动作。
2. deadlock 重试：等待阈值后尝试推进动作。
3. manual intervention：UNKNOWN/无合法动作时进入人工介入等待流程。
4. discard-only 等待态：处理仅剩丢药等非推进动作的短时状态。
5. 断点恢复：自动保存 training_state + pending_buffer + latest。
6. Windows 安全写入重试：降低 training_state.json 写入失败导致中断的概率。

---

## 7. 编码器兼容策略（与模型兼容相关）

state_encoder.py 当前设计：

1. 支持 variant：base / rl / rl_llm。
2. RL+LLM 使用扩展 player 特征维度（16）。
3. 手牌 CARD_FEATURE_DIM 保持 8，不破坏已有 checkpoint 的卡特征维度兼容。
4. keyword_signal 已接入手牌特征。

---

## 8. 配置建议（RL+LLM）

主配置：ppo_sts2agent.yaml

关键参数组：

1. 运行模式
- scheme: rl_llm
- llm.enabled: true

2. LLM后端
- llm.backend
- llm.model
- llm.base_url
- api_key（建议走环境变量）

3. LLM顾问开关
- use_card_advisor
- use_event_advisor
- use_shop_advisor
- use_combat_advisor
- use_map_advisor
- use_relic_advisor

4. 双引导参数
- reward.* 中各 LLM 权重
- llm.policy_guidance_*（logit 轻偏置）

5. 隔离目录
- checkpoint_dir: checkpoints_rl_llm

---

## 9. 运行方式

### 9.1 依赖安装

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 9.2 连接测试

```bash
python test_connection.py --host 127.0.0.1 --port 18080
```

### 9.3 启动 RL+LLM 训练

```bash
python train.py --config ppo_sts2agent.yaml
```

### 9.4 启动单RL基线训练

```bash
python train.py --config ppo_sts2agent_rl.yaml
```

### 9.5 常见环境变量（DeepSeek）

PowerShell 示例：

```powershell
$env:DEEPSEEK_API_KEY="你的key"
```

---

## 10. 产物目录说明

### 10.1 RL+LLM checkpoints（默认）

目录：checkpoints_rl_llm

典型文件：

1. latest_model.pt
2. best_model.pt
3. checkpoint_*.pt
4. training_state.json
5. pending_buffer.pt

### 10.2 单RL checkpoints（默认）

目录：checkpoints_rl

说明：与 RL+LLM 独立。

### 10.3 日志目录

目录：logs/<run_id>

关键日志：

1. module_agent_decision.log
2. module_env_step_state.log
3. module_reward_shaping.log
4. module_ppo_update.log
5. module_episode_summary.log
6. module_error_recovery.log
7. module_manual_intervention.log
8. run_config_snapshot*.json

---

## 11. 如何确认 RL+LLM 真的生效

建议按以下顺序检查：

1. run_config_snapshot 里确认 scheme=rl_llm 且 llm.enabled=true。
2. reward_shaping 日志出现 llm_card_advice / llm_event_advice / llm_shop_advice / llm_combat_turn。
3. 奖励分解行里观察 LLM、matchCard、matchEvent、matchShop 等字段。
4. ppo_update 持续增长，说明不是只在“等待循环”。
5. error_recovery 没有长期 no_progress 死锁式刷屏。

如果 LLM 后端短时失败：

1. 训练应继续推进（fallback生效）。
2. 但 match 命中会下降，策略引导也会减弱。

---

## 12. 评估

evaluate.py 是确定性评估入口。

示例（RL+LLM 模型）：

```bash
python evaluate.py --config ppo_sts2agent.yaml --model checkpoints_rl_llm/best_model.pt --episodes 3
```

示例（单RL模型）：

```bash
python evaluate.py --config ppo_sts2agent_rl.yaml --model checkpoints_rl/best_model.pt --episodes 3
```

输出包含：

1. 每局 reward/floor/max_floor/hp/steps
2. manual_interventions 次数
3. top_actions
4. summary JSON（均值奖励、胜率等）

---

## 13. 常见问题

### 13.1 RL+LLM 会污染单RL吗

默认不会。前提是：

1. 用 ppo_sts2agent.yaml 跑 RL+LLM。
2. 用 ppo_sts2agent_rl.yaml 跑单RL。
3. 不手动把两者 checkpoint_dir 改成同一目录。

### 13.2 为什么 logs 没新建目录

如果你开启了 resume_on_restart 且 continue_logs_on_resume，训练会续写到上一次 latest_run_dir，这是预期行为。

### 13.3 LLM 超时会不会直接中断训练

按当前实现不会直接中断，LLMAdvisor 会走 fallback；但建议尽快修复后端可用性，否则 RL+LLM 增益会明显下降。

---

## 14. 推荐实验流程

1. 先跑单RL基线，确认环境链路稳定。
2. 切到 RL+LLM，小权重开启（先奖励引导，再概率引导）。
3. 观察 3 类指标：
- 流程稳定性（无死锁）
- 学习稳定性（ppo_update连续、损失正常）
- LLM有效性（建议日志与match字段）
4. 稳定后再提高 LLM 相关权重与引导强度。

---

## 15. 相关文档

1. docs/4.3/rl_llm_run_checklist_2026-04-03.md
2. docs/4.2/rl_llm_pipeline_and_guided_policy.md
3. docs/other/STS2AIAgent-API-中文调用文档.md

---

## 16. 快速命令汇总

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 测试连接
python test_connection.py --host 127.0.0.1 --port 18080

# 3) RL+LLM训练
python train.py --config ppo_sts2agent.yaml

# 4) 单RL训练
python train.py --config ppo_sts2agent_rl.yaml

# 5) RL+LLM评估
python evaluate.py --config ppo_sts2agent.yaml --model checkpoints_rl_llm/best_model.pt --episodes 3
```
