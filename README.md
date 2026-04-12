# STS2 RL Agent (STS2AIAgent Adapter)

一个面向《杀戮尖塔2》的强化学习训练项目，基于 PPO 构建，并适配 STS2AIAgent HTTP 接口。

该仓库当前公开结果聚焦于单 RL 训练流程。RL+LLM 方案在代码层面已预留

---

## 项目概览

- 训练范式：单 RL（PPO）
- 任务场景：对接 STS2AIAgent 的状态感知与动作决策
- 工程目标：稳定训练、可恢复、可评估、可扩展

---

## 项目亮点

- 完整训练闭环：环境交互、轨迹采样、PPO 更新、模型保存、离线评估。
- 分层奖励设计：通过 reward shaping 提升学习信号密度与训练可解释性。
- 动作安全约束：策略采样始终受 action mask 约束，保证动作合法。
- 训练可恢复：支持断点续训，便于长周期实验。
- 模块化架构：环境、编码、策略、奖励、评估模块解耦，便于迭代。

---

## 技术栈

- Python
- PyTorch
- PPO (Proximal Policy Optimization)
- STS2AIAgent HTTP API

---

## 训练架构

训练主链路如下：

1. `sts2_env.py` 获取状态与合法动作。
2. `state_encoder.py` 将状态编码为固定维度观测。
3. `ppo_agent.py` 在动作掩码约束下进行策略采样与更新。
4. `reward_shaper.py` 计算训练奖励。
5. `rollout_buffer.py` 收集轨迹，`train.py` 执行周期性 PPO 更新。
6. `evaluate.py` 对模型做确定性评估。

---

## 快速开始（单 RL）

### 1) 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) 测试连接

```bash
python test_connection.py --host 127.0.0.1 --port 18080
```

### 3) 启动训练

```bash
python train.py --config ppo_sts2agent_rl.yaml
```

### 4) 运行评估

```bash
python evaluate.py --config ppo_sts2agent_rl.yaml --model checkpoints_rl/best_model.pt --episodes 3
```

---

## 项目结构（核心文件）

```text
RL_slay the spire_sts2agent/
├─ train.py                 # 训练入口
├─ evaluate.py              # 评估入口
├─ sts2_env.py              # 环境交互
├─ state_encoder.py         # 状态编码
├─ action_space.py          # 动作映射
├─ ppo_agent.py             # PPO策略与更新
├─ rollout_buffer.py        # 轨迹缓存
├─ reward_shaper.py         # 奖励塑形
├─ ppo_sts2agent_rl.yaml    # 单RL配置
├─ llm_advisor.py           # RL+LLM扩展模块（预留）
├─ ppo_sts2agent.yaml       # RL+LLM配置（预留）
└─ requirements.txt
```

---

## 输出产物

训练过程中会生成：

- 模型权重：`latest_model.pt`、`best_model.pt`
- 训练状态：`training_state.json`
- 轨迹缓存：`pending_buffer.pt`
- 运行日志：`logs/<run_id>/...`

---

## 备注

RL+LLM 相关代码目前作为后续扩展，未纳入当前公开结果统计。

---

## 说明

STS2AIAgent 参考项目：https://github.com/CharTyr/STS2-Agent

本项目为研究与工程实践用途，依赖 STS2AIAgent 接口与对应游戏环境。