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
| `play_card` | 出牌：`card_index`，需要时带 `target_index`（敌人索引） |
| `use_potion` | 用药水：`option_index`，需要时带 `target_index` |
| `end_turn` | 结束回合 |
| `choose_reward_card` / `skip_reward_cards` | 奖励牌界面 |
| `choose_map_node` | 地图：`option_index` 对应状态里的 `next_options` |
| `choose_rest_option` | 营地选项：`option_index` |
| `buy_card` / `buy_relic` / `buy_potion` / `proceed` | 商店购买 / 离开等继续流程 |
| `choose_event_option` | 事件选项：`option_index` |

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

### 3.1 Reward v2（严格接管）架构

当前项目已切换为 **Reward v2 五层奖励**（A/B/C/D/E），训练奖励不再使用旧 `STS2Env base_reward` 数值。

| **层** | **含义** | **示例信号** |
| :--- | :--- | :--- |
| Layer A | 动作级即时奖励 | 本步伤害、格挡、选牌动作 |
| Layer B | 回合级结算奖励 | 回合输出效率、击杀、格挡覆盖率、能量浪费、掉血惩罚 |
| Layer C | 战斗级结算奖励 | 战斗结束奖励、精英/首领附加奖励、战斗HP效率 |
| Layer D | 局外决策奖励 | 休息/锻造/删牌/选牌等局外决策偏好 |
| Layer E | 终局奖励 | 胜利/失败、到达层数、终局HP质量 |

总奖励由 `RewardShaper` 统一计算：  
`final_reward = A + B + C + D + E (+ LLM路线/匹配项)`

### 3.2 LLM 在 v2 中的位置（可选叠加）

- `llm.enabled=false`：仅使用 A/B/C/D/E（纯RL）
- `llm.enabled=true`：在 v2 基础上叠加：
  - 全局路线奖励（`llm_weight`）
  - 选牌/遗物/路线/战斗开局一致性奖励（match bonus）

| **重要** |
| :--- |
| 菜单/选角流程在训练主循环中被跳过，不计入样本与奖励。 |

### 3.3 多节点匹配奖励（已接入）

当前实现是“LLM做战略评估，RL做实时执行”：不让 RL 去显式理解流派概念，而是让 LLM 在关键节点给出当前状态下的建议，再由 RewardShaper 做一致性塑形。

- **选牌节点（CARD_REWARD）**：`evaluate_card_reward()` 推荐索引与置信度
- **遗物节点（CHEST/REWARD）**：`evaluate_relic_choice()` 推荐遗物索引
- **路线节点（MAP）**：`evaluate_map_route()` 推荐路线与评分
- **战斗开局节点（COMBAT 首步）**：`evaluate_combat_opening()` 给 opening card sequence

一致性塑形统一遵循：

- **高置信度一致**：正奖励
- **高置信度不一致**：轻惩罚
- **低置信度**：中性（不加不减）

该机制仍属于 Reward Shaping，不改 PPO 损失函数。

### 3.4 阶段性权重调度（已实现，按 `total_steps` 比例）

当前 `train.py` 已支持三阶段动态权重，不再使用固定步数阈值（如 0~5k / 5k~50k）。

- 配置入口：`ppo_sts2agent.yaml` → `reward.phase_schedule`
- 切分方式：按 `train.total_steps` 比例切分
  - 前期：`[0, total_steps * early_end_ratio)`
  - 中期：`[total_steps * early_end_ratio, total_steps * mid_end_ratio)`
  - 后期：`[total_steps * mid_end_ratio, total_steps]`
- 中期支持线性插值（`mid_start_weights` → `mid_end_weights`），避免权重突变带来的训练抖动

| **参数** | **说明** |
| :--- | :--- |
| `enabled` | 是否启用阶段调度 |
| `early_end_ratio` | 前期结束比例（如 0.2） |
| `mid_end_ratio` | 中期结束比例（如 0.7） |
| `early_weights` | 前期 A/B/C/D/E 权重 |
| `mid_start_weights` | 中期开头权重 |
| `mid_end_weights` | 中期结尾权重 |
| `late_weights` | 后期权重 |

> 建议：在单 RL 验证阶段保持温和权重变化，先验证奖励方向正确，再在 RL+LLM 阶段做强策略偏置。

### 3.5 `claim_reward` 多类型差异化奖励（已实现）

基于 STS2AIAgent 最新 API（`reward.rewards[]` 包含 `reward_type`），Layer D 已支持按奖励类型给不同 bonus：

- `Gold`
- `Card`
- `Potion`
- `Relic`
- `RemoveCard`
- `SpecialCard`
- `LinkedRewardSet`
- 未知类型兜底 `claim_unknown_bonus`

实现逻辑：

1. 根据动作中的 `option_index`，在 `prev_state.reward.rewards[]` 中定位目标奖励项
2. 读取该项的 `reward_type`
3. 映射到对应的 Layer D bonus 参数

对应可调参数（`ppo_sts2agent.yaml`）：

- `claim_gold_bonus`
- `claim_card_bonus`
- `claim_potion_bonus`
- `claim_relic_bonus`
- `claim_remove_card_bonus`
- `claim_special_card_bonus`
- `claim_linked_set_bonus`
- `claim_unknown_bonus`

这部分改动的目标是为后续 RL+LLM 联训提供可解释的策略信号：当 LLM 推荐某类奖励时，PPO 能在 reward 侧接收到类型差异。

---

## 4. 大模型集成方案

### 4.1 LLM 在项目中的四个角色

| **角色** | **触发时机** | **输出形式** | **对 RL 的影响** |
| :--- | :--- | :--- | :--- |
| 战略顾问 | 每 `call_interval_steps` 步（非战斗） | JSON: route/synergies/score | 提供路线奖励塑形 |
| 卡牌选择推荐 | `CARD_REWARD` 屏幕前（强制刷新） | `recommended_index` + `confidence` + 推理 | 触发选牌匹配奖励 |
| 遗物选择推荐 | `CHEST/REWARD` 遗物选择场景 | `recommended_index` + `confidence` + 协同说明 | 触发遗物匹配奖励 |
| 地图路线推荐 | 进入 `MAP` 的边沿触发 | `recommended_option_index` + `route_value_scores` | 触发路线匹配奖励 |
| 战斗开局建议 | 进入 `COMBAT` 的第一步前 | `opening_card_sequence` + threat/target | 触发前几步开局匹配奖励 |
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
| 节点边沿触发 | `CARD_REWARD/MAP/CHEST-COMBAT` 仅在边沿触发评估 | 避免同屏重复调用，减轻开销 |
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
- **`train.resume_on_restart: true`** 开启中断续训（默认开启）
- **`train.save_latest_per_update: true`** 每次 PPO 更新保存 `checkpoints/latest_model.pt`
- **`device: cuda`** 有 GPU 且已装好 CUDA 版 PyTorch 时使用；否则 **`cpu`** 或 **`auto`**
- **`env.port`** 与 Mod 实际端口一致（当前项目已统一为 STS2AIAgent session 流程，不再使用 `api_mode`）
- **`env.action_poll_interval` / `env.action_min_interval` / `env.post_action_settle`** 控制轮询与动作节奏（观感卡顿优先调这些参数）

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
| HTTP 409 | 动作不在当前 `legal_actions`、或参数不合法 | 检查 `session/state` 的 `can_act` 与 `legal_actions`；参数统一使用 `option_index` / `target_index` |
| 训练开始后立即报错 | 游戏不在可操作屏幕 | 在游戏中进入一局，保持在战斗/地图屏幕 |
| reward 全为 0 | 奖励函数未触发 | 检查 screen_type 是否为 COMBAT，打印 info 字典 |
| 选牌 match 奖励始终不触发 | `action_executed` 字段与解析逻辑不一致 | 确认执行动作与当前 API 命名一致（推荐 session/legacy 二选一），并检查 `train.py` 中选牌索引解析 |
| pg_loss 为 nan | 学习率太大或梯度爆炸 | 将 lr 从 3e-4 降到 1e-4，检查 max_grad_norm |
| LLM 调用报错 | Ollama 未运行 | 执行 ollama serve，确认监听 11434 端口 |
| 训练极慢（CPU） | 矩阵运算在 CPU | 设置 device: cuda 或减小 buffer_size |
| ImportError | 未激活环境或工作目录错误 | 先 **`conda activate`** 对应环境，在 **项目根目录** 运行；依赖是否已 **`pip install -r requirements.txt`** |

### 6.4 开启 LLM 辅助训练

验证纯 RL 部分正常后，按以下步骤开启 LLM 功能：

10. 修改 config：**`llm.enabled: true`**，设置 backend/model/api_key（该总开关开启后默认启用 card/relic/map/combat_opening 全部评估）
11. 确认 LLM 服务可用（Ollama 或 OpenAI/Anthropic）
12. 检查以下关键参数：  
   - `llm.confidence_threshold`（默认 0.55）  
   - `reward.card_weight` / `reward.relic_choice_weight` / `reward.map_route_weight` / `reward.combat_opening_weight`
   - `llm.combat_bias_steps`（开局建议影响前几步）
13. 重新运行训练，观察日志中是否出现选牌推荐与匹配奖励信息
14. 对比 `llm.enabled: false` 与 `true` 的平均楼层、胜率、奖励曲线，评估 LLM 综合评估收益

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

### 7.4 RL+LLM 待升级方案（建议按优先级迭代）

以下方案按照“先稳定、再变强、后泛化”的顺序设计。

#### P0：联训稳定性与观测能力（优先级最高）

- **统一实验配置快照**：保存 reward/llm 关键参数到每次 run 的配置快照，保证可复现对比
- **分模块指标上报**：在训练日志中分别记录 `LayerA~E`、`LLM_match`、`LLM_global` 的均值/方差
- **调用质量日志**：记录 LLM 推荐索引、置信度、实际选择、命中率（card/relic/map/opening 四类）
- **超时与降级策略**：LLM 失败或超时时强制降级为纯 RL，不阻塞 rollout

#### P1：奖励与建议耦合优化（直接影响效果）

- **置信度分段奖励**：将当前“高置信度命中奖励”改为分段函数（低/中/高置信度不同幅度）
- **类型感知加权**：对 `claim_reward` 的类型奖励与 LLM 建议做联合加权（如 relic > card > potion）
- **阶段调度联动**：将 `phase_schedule` 与 `llm_weight` 联动
  - 前期：较低 `llm_weight`，先学基础生存
  - 中期：逐步升高 `llm_weight`
  - 后期：按策略目标固定在稳定值
- **负反馈软化**：对于 LLM 高置信但策略未采纳的情况，使用轻惩罚并限制上限，避免训练不稳定

#### P2：提示词与知识检索升级（决定上限）

- **模板化 Prompt**：按节点类型拆分模板（card/relic/map/combat），减少上下文噪声
- **候选裁剪**：仅注入当前候选项相关知识（Top-K cards/relics/map routes）
- **轻量 RAG**：用 embeddings 检索 synergies/strategies，替代全量文本拼接
- **结构化输出校验**：对 LLM JSON 输出做 schema 校验 + 自动重试

#### P3：训练与评测闭环（用于最终迭代）

- **A/B 实验矩阵**：纯 RL vs RL+LLM；不同 `llm_weight`、不同 `phase_schedule` 组合
- **关键业务指标**：平均层数、Act1 胜率、精英击杀率、高价值遗物命中率、牌组质量指标
- **离线回放评估**：固定种子回放，比较不同策略在同局面下的动作质量
- **成本评估**：统计每局 LLM 调用次数、平均时延、token 成本（云端模型时）

### 7.5 LLM 接入后重点优化清单（Checklist）

- [ ] `llm.enabled=true` 时，不影响基础 rollout 速度与稳定性
- [ ] LLM 调用失败不会阻塞训练（有 fallback）
- [ ] card/relic/map/opening 四类建议均有可观测命中率日志
- [ ] `llm_weight` 与 `phase_schedule` 联动规则已配置并验证
- [ ] `claim_reward` 类型奖励与 LLM 建议在日志中可解释
- [ ] 至少完成 3 组 A/B 实验并输出结论

---

STS2-RL-Agent 项目分析文档 · PPO + Transformer + LLM Reward Shaping