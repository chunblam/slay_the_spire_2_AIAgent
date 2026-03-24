# STS2-RL-AGENT

**杀戮尖塔 2 × 强化学习 × 大语言模型**

项目全景分析 · 模型原理 · 方案设计 · 运行手册

| **模块** | **技术** |
| :--- | :--- |
| 强化学习算法 | PPO (Proximal Policy Optimization) |
| 神经网络架构 | Transformer Encoder + Actor-Critic |
| 大模型角色 | 战略顾问 + 奖励塑形 + 知识检索 |
| 游戏接口 | STS2AIAgent Mod（HTTP API，默认 `127.0.0.1:18080`） |
| 知识数据库 | Spire Codex (576卡 / 289遗物 / 121怪物) |
| 主语言 | Python 3.10+ / PyTorch 2.0+ |

---

## 1. 项目整体分析与定位

### 1.1 项目价值与简历亮点

本项目融合多个当前 AI 研究热点，在简历中具有很高的区分度：

- **多技术栈融合：** 强化学习 + 大语言模型 + 游戏 AI，覆盖三个独立研究方向
- **完整 E2E 工程：** 从游戏接口、状态编码、神经网络设计到训练流程，完整端到端项目
- **前沿研究方向：** LLM-guided RL 是 2024-2025 年活跃方向（ELLM、Eureka 等论文）
- **可演示性强：** 能实时看到 AI 打游戏，面试演示直观有力
- **领域知识深度：** Spire Codex 集成展示了对知识工程的理解

### 1.2 两个仓库的角色与使用方案

#### STS2AIAgent --- 游戏接口层（直接复用）

| **使用策略** |
| :--- |
| 复用 `STS2AIAgent` 暴露的本地 HTTP API（与 MCP 工具链无关）。RL Agent 推荐走 **session 三端点**：`GET /api/v1/session/state`、`GET /api/v1/session/legal_actions`、`POST /api/v1/session/action`；保留兼容旧端点 `GET /state`、`GET /actions/available`、`POST /action`。详细约定见仓库内 **`docs/STS2AIAgent-API-中文调用文档.md`**（依据 `STS2-Agent/docs/api.md` 整理）。 |

| **模式** | **HTTP** | **说明** |
| :--- | :--- | :--- |
| 推荐（session） | `GET http://127.0.0.1:18080/api/v1/session/state`<br>`GET http://127.0.0.1:18080/api/v1/session/legal_actions`<br>`POST http://127.0.0.1:18080/api/v1/session/action` | 生命周期友好；支持 `can_act` / `block_reason` / `legal_actions` |
| 兼容（legacy） | `GET http://127.0.0.1:18080/state`<br>`GET http://127.0.0.1:18080/actions/available`<br>`POST http://127.0.0.1:18080/action` | 与旧脚本兼容 |

| **常用 `action`（节选）** | **含义** |
| :--- | :--- |
| `play_card` | 出牌：`card_index`，需要时带 `target`（敌人 `entity_id`） |
| `use_potion` | 用药水：`slot`，需要时带 `target` |
| `end_turn` | 结束回合 |
| `select_card_reward` / `skip_card_reward` | 奖励牌界面 |
| `choose_map_node` | 地图：`index` 对应状态里的 `next_options` |
| `choose_rest_option` | 营地选项：`index` |
| `shop_purchase` / `proceed` | 商店购买 / 离开等继续流程 |
| `choose_event_option` | 事件选项：`index` |

| **提示** | `STS2AIAgent` 默认监听 **`127.0.0.1:18080`**。若使用自定义端口，可通过环境变量 **`STS2_API_PORT`** 或启动脚本参数覆盖，并在 **`ppo_default.yaml`** / **`test_connection.py --port`** 同步。 |

#### Spire Codex --- 游戏知识库（离线使用）

| **使用策略** |
| :--- |
| 训练开始前一次性从 Spire Codex API 拉取所有卡牌/遗物/怪物数据，生成 knowledge_base.json。训练期间 LLM Advisor 检索该文件获取游戏知识上下文。Spire Codex 不需要在训练期间保持运行。 |

| **数据类型** | **关键字段** | **在项目中的作用** |
| :--- | :--- | :--- |
| 卡牌 (576张) | cost / damage / block / type / rarity | LLM 知识上下文 / 奖励函数参考 |
| 遗物 (289个) | name / rarity / pool / description | 高价值遗物识别 / 协同判断 |
| 怪物 (121个) | hp_range / moves / type(boss/elite) | 威胁评估 / 奖励设计 |
| 角色 (5个) | starting_deck / starting_hp / max_energy | 角色特化训练初始化 |
| 药水 (63个) | rarity / description | 药水价值评估 |

### 1.3 整体系统架构

| **层级** | **模块** | **职责** |
| :--- | :--- | :--- |
| 接口层 | STS2AIAgent Mod | 暴露游戏状态和操作的 HTTP API |
| 环境层 | STS2Env | Gymnasium 标准接口封装，基础奖励计算 |
| 编码层 | StateEncoder | JSON 状态 → 神经网络输入张量 |
| 动作层 | STS2ActionSpace | 动作 ID → API `POST` JSON（含 `action`），合法动作 Mask |
| 决策层 | STS2PolicyNet + PPOAgent | Actor-Critic，PPO 训练核心 |
| 知识层 | KnowledgeBuilder | 构建攻略知识库 JSON |
| 顾问层 | LLMAdvisor | 大模型战略建议，卡牌选择推荐 |
| 奖励层 | RewardShaper | 合并基础 + LLM + 规则三层奖励 |
| 训练层 | train.py | 主训练循环，GAE，日志保存 |

---

## 2. 模型原理与函数设计分析

### 2.1 强化学习框架：PPO 算法

选择 PPO 的理由：

- **稳定性：** Clip 机制防止策略更新过大，适合样本噪声大的真实游戏环境
- **样本效率：** 同一批数据可训练多个 epoch，比原始 PG 利用率高
- **离散动作天然兼容：** PPO + Categorical 分布直接对应卡牌出牌的离散决策
- **工程成熟度：** 实现简洁，社区资源丰富，调试容易

**PPO 核心数学公式**

| **公式** | **含义** |
| :--- | :--- |
| $r_t = \pi_{new}(a|s) / \pi_{old}(a|s)$ | 新旧策略概率比值（重要性采样权重） |
| $L_{CLIP} = \mathbb{E}[\min(r \cdot A, \text{clip}(r, 1-0.2, 1+0.2) \cdot A)]$ | 策略损失，clip 防止更新幅度过大 |
| $L_{VF} = \mathbb{E}[(V(s) - R_t)^2]$ | 价值函数均方误差损失 |
| $L_{ENT} = -\mathbb{E}[H(\pi(\cdot|s))]$ | 熵正则项，鼓励探索（系数 0.01） |
| $L_{total} = L_{CLIP} - 0.5 \cdot L_{VF} + 0.01 \cdot L_{ENT}$ | 总损失 |
| $A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots$ | 广义优势估计 GAE（gamma=0.99, lambda=0.95） |

### 2.2 神经网络架构：STS2PolicyNet

设计核心挑战是游戏状态的异构性——手牌是变长序列，敌人是小集合，遗物是集合。针对各子状态设计了专门编码器：

#### CardEncoder --- Transformer 手牌编码

手牌是无序集合，但卡牌间存在策略性交互（如「先打格挡再打攻击」），Transformer 能自然建模这种关系：

| **层** | **输入形状** | **输出形状** | **作用** |
| :--- | :--- | :--- | :--- |
| Linear Projection | [B, 10, 8] | [B, 10, 64] | 将 8 维卡牌特征投影到 64 维 |
| TransformerEncoder x2 | [B, 10, 64] | [B, 10, 64] | 建模手牌内卡牌间的注意力关系 |
| Key Padding Mask | hand_mask | — | 屏蔽空槽不参与注意力计算 |
| Global Average Pool | [B, 10, 64] | [B, 64] | 聚合为全局手牌表示 |

#### MonsterEncoder --- 敌人状态编码

敌人数量少（最多5个），用 MLP 编码后展平拼接：

| **特征** | **维度** | **说明** |
| :--- | :--- | :--- |
| hp_ratio | 1 | 当前 HP / 最大 HP |
| block_ratio | 1 | 当前格挡 / 100（归一化） |
| intent_damage | 1 | 意图伤害 / 100（归一化） |
| alive | 1 | 是否存活 0/1 |
| intent_type_onehot | 4 | ATTACK / DEFEND / BUFF / DEBUFF 四类 |

#### Actor-Critic 双头输出

| **头部** | **输入** | **输出** | **计算方式** |
| :--- | :--- | :--- | :--- |
| Actor（手牌） | [B, hidden+card_d] | [B, 10] | 全局特征与每张 card_repr 拼接后 Linear→1 |
| Actor（其他） | [B, hidden] | [B, 6] | 直接 Linear 投影 |
| Critic | [B, hidden] | [B, 1] | Linear 投影后 squeeze |

| **设计亮点** |
| :--- |
| 手牌动作的打分方式（全局特征 × 卡牌特征交叉打分）本质上是一种注意力评分机制——模型在「全局策略意图」和「具体卡牌特征」之间做交叉评估。相比简单地用全局特征预测所有动作，这种方式能让模型区分同类型但数值不同的卡牌，学到更精细的出牌策略。 |

### 2.3 状态编码设计

StateEncoder 将游戏 JSON 转化为固定维度的 numpy 数组字典：

| **观测子空间** | **维度** | **内容描述** |
| :--- | :--- | :--- |
| player | (10,) | HP/格挡/能量/金币/楼层/遗物数/牌组大小/是否战斗/buff数/debuff数 |
| hand | (10, 8) | 每张手牌：费用/伤害/格挡/是否虚空/是否耗尽/是否天赋 |
| hand_mask | (10,) | 1=有效可出，0=空槽或无法出 |
| monsters | (5, 8) | 每个敌人：HP比/格挡比/意图伤害/alive/意图类型 one-hot |
| relics | (30,) | 遗物 one-hot（哈希映射到固定槽位） |
| deck_stats | (8,) | 总数/各类型比例/升级率/平均费用 |
| screen_type | scalar | 当前屏幕类型整数 ID（16 种） |

### 2.4 动作空间设计

动作空间是固定大小的离散空间（16个动作），通过 Action Masking 屏蔽当前无效动作：

| **动作 ID** | **动作类型** | **对应屏幕** |
| :--- | :--- | :--- |
| 0 ~ 9 | 打出第 i 张手牌 | COMBAT |
| 10 ~ 14 | 使用第 i 个药水 | COMBAT |
| 15 | 结束回合 | COMBAT |
| 0 ~ 3 | 选择奖励卡（0=跳过） | CARD_REWARD |
| 0 ~ 6 | 选择地图路径节点 | MAP |
| 0=休息 / 1=锻造 | 营地选择 | REST |
| 0 ~ 4 | 事件选项 | EVENT |

| **Action Masking 关键作用** |
| :--- |
| get_valid_action_mask() 在每步返回布尔向量，将无效动作的 logit 设为 -inf，确保采样只来自合法动作。不屏蔽无效动作会导致大量无效探索（如用3费卡但只剩2能量），严重拖慢训练速度。 |

---

## 3. 奖励函数设计

### 3.1 三层奖励架构

| **分量** | **权重** | **来源** | **信号类型** |
| :--- | :--- | :--- | :--- |
| 基础奖励 Base | 1.0 | STS2Env 环境 | 稀疏 + 部分稠密 |
| 规则奖励 Rule | 0.5 | RewardShaper 硬编码规则 | 稠密，基于游戏知识 |
| LLM 路线奖励 | 0.3 | LLMAdvisor 全局评估 | 稠密，战略层面引导 |
| 选牌匹配奖励（方案A） | 0.4（默认） | LLM 推荐 vs Agent 实际选牌 | 稠密，构筑层面引导 |

### 3.2 基础奖励（Base Reward）

| **事件** | **奖励值** | **设计意图** |
| :--- | :--- | :--- |
| 击杀敌人（每个） | +5.0 | 鼓励消灭威胁 |
| 玩家受伤（每点 HP） | -0.1 | 惩罚低效防守 |
| 进入新楼层 | +1.0 | 鼓励推进游戏进程 |
| 获得金币（每枚） | +0.01 | 鼓励资源积累 |
| 通关 BOSS | +50.0 | 幕终极正奖励 |
| 死亡 | -20.0 | 终局惩罚 |

### 3.3 规则奖励（Rule Bonus）

将游戏攻略知识直接编码为奖励信号，引导模型学习正确策略：

| **规则** | **奖励逻辑** | **游戏知识依据** |
| :--- | :--- | :--- |
| 出牌造成伤害 | +0~0.5（按伤害比例） | 鼓励有效输出而非空过 |
| 出牌增加格挡 | +0~0.3（HP越低权重越高） | 低血量时格挡价值更高 |
| 有牌可打却结束回合 | -0.1 × 可出牌数 | 惩罚浪费能量 |
| HP<40% 时休息 | +0.5 | 正确的补血时机判断 |
| HP>80% 时休息 | -0.2 | 惩罚浪费营地 |
| 锻造强化牌 | +0.3 | 鼓励牌组质量提升 |
| 牌组含诅咒/状态牌 | 持续小惩罚 | 引导主动删除废牌 |

### 3.4 LLM 奖励（LLM Bonus）

LLM Advisor 在非战斗屏幕评估卡组路线质量，返回 [-1, +1] 分数，经权重缩放后加入总奖励：

- **+1.0：** 卡组路线极清晰，关键牌全拿到（如毒素流：催化剂+ 毒云+ 毒刺）
- **+0.5：** 路线正在成形，有不错的协同组合
- **0.0：** 卡组一般，无明显路线
- **-0.5：** 卡组混乱，路线冲突
- **-1.0：** 明显错误（如已有力量流却继续拿大量防守牌）

| **重要：LLM 奖励只在非战斗屏幕触发** |
| :--- |
| 战斗中出牌决策完全由 RL Agent 实时执行（LLM 响应延迟 0.5~2秒，无法用于实时战斗）。LLM 的价值在于战略层面的指导——选什么牌、走哪条路线——这些决策不需要毫秒级响应。 |

### 3.5 方案A：CARD_REWARD 选牌匹配奖励（已接入）

在 `CARD_REWARD` 屏幕，训练流程会先调用 LLM 评估候选奖励牌（结合 Codex 卡牌条目 + synergies + strategies），再由 RL Agent 自主选牌。动作执行后，RewardShaper 对比「LLM 推荐索引」与「Agent 实际索引」：

- **一致**：给 `card_match_bonus`（默认 1.0）并乘 `card_weight`（默认 0.4）
- **LLM 建议跳过且 Agent 也跳过**：给半额正奖励
- **高置信度下不一致**：给 `card_mismatch_penalty`（默认 0.5，对应负向）
- **低置信度**（`confidence < llm.confidence_threshold`）：不施加匹配塑形

这属于 **方案A（Reward Shaping）**：不修改 PPO 损失函数，只通过奖励信号引导构筑探索。

---

## 4. 大模型集成方案

### 4.1 LLM 在项目中的四个角色

| **角色** | **触发时机** | **输出形式** | **对 RL 的影响** |
| :--- | :--- | :--- | :--- |
| 战略顾问 | 每 `call_interval_steps` 步（非战斗） | JSON: route/synergies/score | 提供路线奖励塑形 |
| 卡牌选择推荐 | `CARD_REWARD` 屏幕前（强制刷新） | `recommended_index` + `confidence` + 推理 | 触发方案A选牌匹配奖励 |
| 知识检索器 | 构建 prompt 时 | 卡级 Codex + 协同 + 策略段落 | 丰富 LLM 上下文，降低幻觉 |
| 奖励设计师（离线） | 训练前人工阶段 | 攻略知识 JSON | 奠定规则奖励与协同条目基础 |

### 4.2 三种 LLM 后端对比

| **后端** | **推荐模型** | **优点** | **缺点** | **适用场景** |
| :--- | :--- | :--- | :--- | :--- |
| 本地 Ollama | Qwen2.5:7B | 免费，无网络延迟，隐私安全 | 需 8GB+ 显存，效果弱于 GPT-4 | 日常训练（推荐） |
| OpenAI API | gpt-4o-mini | 效果最好，稳定 | 有费用（约\$0.15/1M tokens） | 最终评估/对比实验 |
| Anthropic API | claude-haiku | 效果好，费用中等 | 需 API Key | 实验对比 |

| **推荐方案** |
| :--- |
| 开发阶段使用 Ollama + Qwen2.5:7B（本地免费），最终评估时切换到 gpt-4o-mini 做效果对比。只需修改 configs/ppo_default.yaml 中 llm.backend 和 llm.model 两个字段即可，代码无需任何改动。 |

### 4.3 Prompt 设计原则（更新）

- **强制 JSON 输出：** System Prompt 明确要求只返回 JSON，避免自然语言无法解析
- **字段语义明确：** 每个字段给出取值范围和含义，减少模型幻觉
- **三层知识注入：** 候选牌/牌组卡级 Codex 条目 + synergies + strategies（不再只依赖攻略段落）
- **状态摘要而非完整 JSON：** 原始游戏 JSON 可能数千 token，摘要版控制在合理 token 范围
- **文本清洗可选：** 可在知识库构建后去除颜色标签（如 `[gold]...[/gold]`），避免 LLM 被 UI 标记干扰

### 4.4 调用频率控制

| **策略** | **具体实现** | **效果** |
| :--- | :--- | :--- |
| 间隔调用 | 每 `call_interval_steps` 步调用一次（默认 10） | 减少全局评估开销 |
| TTL 缓存 | 同一建议在 `cache_ttl` 秒内复用（默认 30s） | 应对连续快速状态变化 |
| 选牌强制刷新 | `CARD_REWARD` 屏幕前强制调用 `evaluate_card_reward()` | 确保选牌推荐是当前局面 |
| 战斗跳过 | `COMBAT` 屏幕不做全局 LLM 评估 | 消除战斗中的延迟瓶颈 |
| 未来优化：异步化 | 后台线程调用，主线程继续 RL 更新 | 进一步降低等待 |

---

## 5. 训练知识来源与数据方案

### 5.1 三层知识体系

**层次一：游戏结构化数据（Spire Codex）**

通过 `build_knowledge_base.py` 一键拉取，涵盖卡牌/遗物/怪物等结构化数据，是最可靠的知识来源。当前默认支持：

- `--lang zhs`（默认）：拉取简体中文描述
- `--codex-url`：可指定本地或远程 Spire Codex URL

**层次二：内置攻略知识（代码硬编码）**

knowledge_builder.py 的 _get_builtin_strategies() 内置了提炼自社区的核心策略知识：

- 卡组规模控制原则（12-18 张最优区间）
- 删牌策略（何时删弱牌，如何减小牌组体积）
- 地图路线选择逻辑（精英/商店/营地优先级）
- 各角色特化流派（力量流/毒素流/法球流/姿态流）
- BOSS 战特殊注意事项
- 遗物优先级评估框架

**层次三：外部攻略扩充（可选）**

- **Slay the Spire Wiki：** wiki.gg，包含所有卡牌协同详细说明
- **Reddit r/slaythespire：** 真实玩家的 run 分析和策略讨论
- **YouTube 攻略视频：** 用 Whisper 转录字幕后整理

| **知识库扩充方法** |
| :--- |
| 将新攻略文本整理成 `{character, route, text}` 追加到 `strategies`；将显式卡组配合追加到 `synergies`（`{"cards":[...], "description":"..."}`）。未来可升级为 Embedding + 向量检索（RAG）。 |

### 5.2 为什么以 RL 为主而非模仿学习？

| **方案** | **优点** | **缺点** |
| :--- | :--- | :--- |
| 纯模仿学习 IL | 训练快，行为直观 | 受限于数据质量，无法超越示范者水平 |
| 纯强化学习 RL | 理论上可超越人类 | 探索困难，奖励稀疏，样本效率低 |
| 本项目：RL + LLM 引导 | LLM 提供战略先验加速探索，RL 负责执行层优化 | 需要 LLM 调用开销，系统复杂度略高 |

---

## 6. 完整运行、调试与使用步骤

### 6.0 使用 Anaconda 创建虚拟环境（推荐）

在 **Anaconda Prompt** 或已初始化 conda 的终端中执行。下面将环境名设为 **`sts2rl`**，可自行改名。

**1. 创建环境**（Python 3.10 与当前代码兼容）

```bash
conda create -n sts2rl python=3.10 -y
```

**2. 激活环境**

```bash
conda activate sts2rl
```

**3. 进入项目根目录**（包含 `train.py`、`requirements.txt` 的文件夹）

```bash
cd "你的路径\RL_slay the spire"
```

**4. 安装依赖**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**5.（可选）NVIDIA GPU + CUDA 版 PyTorch**

若需要 GPU 训练，请到 [PyTorch 官网](https://pytorch.org/get-started/locally/) 按本机 CUDA 版本复制安装命令，在**已激活的 `sts2rl` 环境**中执行（通常会覆盖或重装 `torch`）。仅 CPU 训练可跳过本步。

**6. 退出环境**

```bash
conda deactivate
```

之后每次新开终端要先执行 **`conda activate sts2rl`** 再运行本项目脚本。

### 6.1 环境准备

**Step 1：Python 依赖**

| **包名** | **用途** |
| :--- | :--- |
| torch | 神经网络与 PPO 训练 |
| gymnasium | RL 环境接口 |
| numpy | 数值计算 |
| requests | 调用 Mod 的 HTTP API |
| PyYAML | 读取 `ppo_default.yaml` |

一键安装：在项目根目录执行 **`pip install -r requirements.txt`**（若已按 **6.0** 建好 Conda 环境，请在激活该环境后执行）。

**Step 2：安装 STS2AIAgent Mod**

1. 从 STS2-Agent Releases 下载，或从源码构建 `STS2AIAgent`
2. 将 **STS2AIAgent.dll**、**STS2AIAgent.pck**、**mod_id.json** 复制到 **\<游戏目录\>/mods/**
3. 启动游戏 → 设置 → 启用 Mod → 同意弹出的权限说明
4. Mod 启动后默认在 **`http://127.0.0.1:18080`** 提供 API（具体端口以 Mod / 环境变量为准）

**Step 3（可选）：运行 Spire Codex（仅构建知识库时需要）**

5. 克隆：**git clone https://github.com/ptrlrd/spire-codex**
6. 启动（在 `spire-codex/backend` 目录）：**uvicorn app.main:app --host 0.0.0.0 --port 8000**
7. 验证：访问 **http://localhost:8000/api/stats** 应看到统计数据
8. 本项目只依赖 Codex 后端 API；不要求本地运行前端

**Step 4（可选）：安装本地 LLM**

8. 从 https://ollama.com 安装 Ollama
9. 拉取模型：**ollama pull qwen2.5:7b**（约 4GB 下载）

### 6.2 首次运行流程

**说明：** 下列命令均在 **项目根目录** 执行；若使用 Conda，请先 **`conda activate sts2rl`**。

**Step 1：测试游戏连接**

`python test_connection.py`

可选参数：`--host 127.0.0.1`、`--port 18080`（或你的 Mod 端口）。

预期输出（游戏正在运行且 Mod 已启用时）：**连接成功**，并打印 **`state_type`**、楼层、金币等摘要。

若失败，请检查游戏是否运行、Mod 是否已启用、防火墙，以及 **`/health`** 是否可访问。

**Step 2：构建知识库**

推荐（中文 + 本地 Codex）：

`python build_knowledge_base.py --lang zhs`

可选参数：

- **`--codex-url http://localhost:8000`**：显式指定 Codex 地址
- **`--no-codex`**：Codex 未运行时，只使用内置策略/协同

可选清洗（只移除颜色高亮标签，不改游戏语义占位）：

`python strip_kb_color_tags.py`

**Step 3：修改训练配置**

编辑项目根目录下的 **`ppo_default.yaml`**，建议初次训练：

- **`llm.enabled: false`** 先关闭 LLM，验证 RL 部分正常
- **`train.total_steps: 50000`** 快速验证后再增大
- **`device: cuda`** 有 GPU 且已装好 CUDA 版 PyTorch 时使用；否则 **`cpu`** 或 **`auto`**
- **`env.port` / `env.api_mode`** 与 Mod 实际端口、单机/联机一致

**Step 4：开始训练**

`python train.py --config ppo_default.yaml`

训练过程打印示例：

**Episode 1 | Reward: 8.50 | Floor: 3 | HP: 45**

**Episode 2 | Reward: 12.30 | Floor: 5 | HP: 0**

**Update @ step 2048 | avg_reward: 10.40 | pg_loss: 0.023 | vf_loss: 0.089**

### 6.3 调试问题速查

| **问题现象** | **可能原因** | **解决方法** |
| :--- | :--- | :--- |
| test_connection.py 超时或连接失败 | 游戏未运行、Mod 未启用或端口错误 | 确认 API 地址为 **`http://127.0.0.1:<端口>/health`** 可访问，并与 **`ppo_default.yaml`** 中 **`env.port`** 一致 |
| HTTP 409 | 单机/联机 API 混用 | 使用与当前对局一致的 **`env.api_mode`** |
| 训练开始后立即报错 | 游戏不在可操作屏幕 | 在游戏中进入一局，保持在战斗/地图屏幕 |
| reward 全为 0 | 奖励函数未触发 | 检查 screen_type 是否为 COMBAT，打印 info 字典 |
| 选牌 match 奖励始终不触发 | `action_executed` 字段与解析逻辑不一致 | 确认执行动作与当前 API 命名一致（推荐 session/legacy 二选一），并检查 `train.py` 中选牌索引解析 |
| pg_loss 为 nan | 学习率太大或梯度爆炸 | 将 lr 从 3e-4 降到 1e-4，检查 max_grad_norm |
| LLM 调用报错 | Ollama 未运行 | 执行 ollama serve，确认监听 11434 端口 |
| 训练极慢（CPU） | 矩阵运算在 CPU | 设置 device: cuda 或减小 buffer_size |
| ImportError | 未激活环境或工作目录错误 | 先 **`conda activate`** 对应环境，在 **项目根目录** 运行；依赖是否已 **`pip install -r requirements.txt`** |

### 6.4 开启 LLM 辅助训练（方案A）

验证纯 RL 部分正常后，按以下步骤开启 LLM 功能：

10. 修改 config：**`llm.enabled: true`**，设置 backend/model/api_key
11. 确认 LLM 服务可用（Ollama 或 OpenAI/Anthropic）
12. 检查以下关键参数：  
   - `llm.confidence_threshold`（默认 0.55）  
   - `reward.card_weight` / `card_match_bonus` / `card_mismatch_penalty`
13. 重新运行训练，观察日志中是否出现选牌推荐与匹配奖励信息
14. 对比 `llm.enabled: false` 与 `true` 的平均楼层、胜率、奖励曲线，评估方案A收益

### 6.5 训练结果评估指标

| **评估指标** | **说明** | **参考目标** |
| :--- | :--- | :--- |
| 平均楼层深度 | 每局游戏平均到达的楼层数（满层55） | 大于 20 层为良好起步 |
| Win Rate | 通关第一幕 BOSS 的比率 | 大于 30% 为良好起步 |
| 平均剩余 HP | 结束时 HP 占最大 HP 的比例 | 反映防御和伤害控制质量 |
| 平均奖励曲线 | 随训练步数的收益变化折线图 | 应持续上升后趋于平稳 |
| 策略熵 Entropy | 策略的随机性指标 | 不应过低（过拟合）也不应过高（随机） |

---

## 7. 后续扩展与进阶方向

### 7.1 近期可做的工程改进

- **TensorBoard 日志：** 可视化 reward/loss 曲线，方便调参和展示
- **RAG 知识检索：** 用 sentence-transformers 做向量索引，提升 LLM 上下文质量
- **角色特化模型：** 分别训练不同角色的 Agent，效果通常优于通用模型
- **课程学习：** 从简单关卡（仅第1层）开始，逐步提高难度和 Ascension 等级
- **异步 LLM 调用：** 后台线程调用 LLM，主线程继续 RL 更新，消除等待时间

### 7.2 研究层面的进阶方向

- **Eureka 式自动奖励生成：** 让 LLM 根据游戏描述自动生成奖励函数代码（NVIDIA 2023年论文方向）
- **RLHF 人工反馈：** 对 Agent 游玩录像进行人工评分，训练奖励模型
- **MCTS + RL：** 卡牌选择时用蒙特卡洛树搜索前瞻，类似 AlphaGo 思路
- **离线 RL：** 收集人类游玩数据后用 Conservative Q-Learning 训练，无需实时交互

### 7.3 简历展示建议

14. **GitHub 仓库：** 代码结构清晰，README 有架构图和训练结果截图
15. **训练结果可视化：** 导出奖励曲线图，展示模型在训练过程中的提升轨迹
16. **游玩录像：** 录制 AI 游玩片段，展示 Agent 的具体决策行为
17. **量化成果：** 「Agent 能稳定通关第一幕 BOSS（Win Rate 35%）」比「训练了游戏 AI」有说服力得多
18. **技术博客：** 写一篇关于 LLM + RL 方案设计的技术文章（知乎/CSDN/Medium）

---

STS2-RL-Agent 项目分析文档 · PPO + Transformer + LLM Reward Shaping