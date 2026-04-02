# RL+LLM 方案总览与评估（4.2）

更新时间：2026-04-02
适用仓库：RL_slay the spire_sts2agent

## 1. 本文目标

本文回答以下问题：

1. 当前 RL+LLM 在代码中的真实工作方式是什么。
2. LLM 会在什么场景被调用，调用输入格式是什么。
3. LLM 建议是否有标准化输出、是否有缓存和失效机制。
4. RL 如何使用 LLM 建议（当前仅奖励引导，还是可扩展到动作概率引导）。
5. 是否可以在不从头训练的前提下，做“奖励 + 概率”双引导。

## 2. 先说结论

1. 当前实现中，LLM 主要用于奖励塑形（reward shaping），不是直接替代策略网络选动作。
2. LLM 在奖励、地图、遗物、删牌、商店、战斗开局等关键节点会被调用或刷新建议。
3. LLM 输出是标准化 JSON，并带有边界检查和失败回退。
4. 已有缓存与失效机制（call interval + TTL + action 后 invalidate）。
5. 不需要从头训练。可以基于现有 checkpoint 渐进切到 RL+LLM。
6. 可以扩展为“概率 + 奖励”双引导，但建议采用保守混合策略，避免策略崩坏。

## 3. 当前调用链路（从训练循环看）

### 3.1 训练循环中 LLM 触发点

在 train.py 中，LLM 的建议主要在以下节点触发（进入对应 screen 或特定子阶段时）：

1. 选卡奖励：evaluate_card_reward
2. 地图选择：evaluate_map_route
3. 遗物选择：evaluate_relic_choice
4. 删牌选择：evaluate_card_remove
5. 商店购买：evaluate_shop_purchase
6. 战斗开局：evaluate_combat_opening

这些建议会被记录日志（llm_card_advice、llm_map_advice、llm_shop_advice 等），并在 reward_shaper 中被读取用于匹配奖励。

### 3.2 RL 的动作仍由策略网络生成

当前动作执行主路径仍是：

1. 用 observation + action mask 调用 policy.get_action
2. 得到 action_id
3. 执行动作并拿到下一状态
4. reward_shaper.shape 合成最终训练奖励

即：LLM 不直接决定 action_id，默认只影响奖励信号。

## 4. LLM 调用分类与上下文

LLM 目前可以分为两类调用：

1. 全局路线评估（global advice）
2. 场景化决策评估（card/relic/map/remove/shop/combat opening）

### 4.1 全局路线评估（_query_llm_global）

输入上下文主要包括：

1. 角色、楼层、HP、金币
2. 当前牌组摘要
3. 关键牌的 Codex 数据检索结果
4. 协同条目（synergies）
5. 角色策略段落（strategies）

输出标准化字段：

1. deck_route
2. route_score
3. key_synergies
4. reward_shaping
5. reasoning

### 4.2 奖励卡评估（evaluate_card_reward）

输入上下文主要包括：

1. 当前牌组、遗物、楼层、HP
2. 候选奖励卡（含 Codex 数据）
3. 已知协同文本
4. 角色流派策略文本

输出标准化字段：

1. recommended_index（-1 表示跳过）
2. confidence
3. reasoning
4. deck_route_after
5. key_combo

### 4.3 商店评估（evaluate_shop_purchase）

输入上下文主要包括：

1. 楼层、金币
2. 当前牌组摘要
3. shop.cards / shop.relics / shop.potions 的可购条目
4. 是否可删牌与删牌价格

输出标准化字段：

1. recommended_action（buy_card / buy_relic / buy_potion / remove_card_at_shop / proceed / NONE）
2. option_index
3. confidence
4. reasoning

说明：这与“在商店把可购物品 + 当前牌组/遗物/药水综合交给 LLM 决策”的需求一致，代码中已经按该思路实现。

### 4.4 其他场景

1. 遗物选择：recommended_index + confidence + reasoning
2. 地图选择：recommended_option_index + route_value_scores
3. 删牌选择：recommended_index + confidence
4. 战斗开局：threat_level / priority_action / opening_card_sequence / target

## 5. 输出标准化与健壮性

### 5.1 标准化

LLM 系统提示明确要求只返回 JSON；每个场景有固定 schema。

### 5.2 健壮解析

通过统一 JSON 解析器处理：

1. 兼容 markdown 代码块包裹
2. 尝试从文本中提取第一个 JSON 对象
3. 失败时走异常回退

### 5.3 边界保护

对 index、action 枚举等进行边界校验：

1. 索引越界会钳制或回退
2. 非法动作会回退到 NONE
3. 失败时返回低置信度与默认建议

## 6. 缓存与失效机制

当前已有三层“快返回”机制：

1. 调用步频控制：call_interval_steps
2. 时间缓存：cache_ttl
3. 动作后失效：执行对应动作后 invalidate_*_recommendation

效果：

1. 减少重复请求与 API 延迟
2. 避免旧建议长期污染新状态
3. 在关键屏幕（如奖励选卡）可强制刷新

## 7. RL 目前如何参考 LLM 建议

当前是“奖励参考”，不是“动作直接参考”：

1. RewardShaper 读取 LLM 最近建议（如 card/relic/map/shop/opening）。
2. 如果 agent 行为与建议匹配，给正向 bonus；不匹配给惩罚或零增益。
3. 叠加 route bonus（llm_weight * llm_route）。

这让策略网络通过 PPO 在长期回报上“更愿意”靠近 LLM 建议。

## 8. 你提的关键问题：可否做概率引导

答案：可以，而且建议做“保守混合”，不要硬替换策略。

### 8.1 推荐实现（不破坏现有框架）

在 action 采样前增加一个 advice bias 项：

1. LLM 返回 recommended_action / index / confidence
2. 映射到 action_id
3. 对该 action logit 增加偏置：bias = alpha * confidence
4. alpha 从小到大渐进（例如 0.0 -> 0.4）

再与 action_mask 叠加，保持非法动作永远不可选。

### 8.2 与奖励引导协同

建议“双通道”并行：

1. 通道A：现有 reward shaping（稳定）
2. 通道B：logit 轻偏置（加速探索）

并设置保护：

1. 仅在 confidence >= 阈值时加偏置
2. 仅在关键 screen 启用（REWARD / MAP / SHOP / CARD_SELECTION）
3. COMBAT 中仅轻度启用 opening sequence 引导

### 8.3 风险与规避

风险：

1. 过强偏置导致策略塌缩到“模仿 LLM”
2. LLM 错判时放大错误动作

规避：

1. 小权重起步
2. 分阶段升权重
3. 保留 entropy regularization
4. 监控 KL、entropy、max_floor、rollout_avg_ep_r

## 9. 迁移建议（从当前训练继续，不从头）

你现在可直接从当前 checkpoint 继续，不需要重训。

推荐三阶段：

1. 阶段1（5k步）：仅开小奖励权重（无概率偏置）
2. 阶段2（5k-10k步）：开启小概率偏置（alpha <= 0.15）
3. 阶段3（后续）：按指标逐步增加到 alpha 0.25-0.40

观察指标：

1. rollout_avg_ep_r 是否上升
2. max_floor 是否提升
3. entropy 是否过快下坠
4. no_progress / recoverable_error 是否增多

## 10. 本轮对 discard_potion 的策略修正

已按需求调整训练策略：

1. 不再永久禁止 discard_potion
2. 当检测到药水槽已满时，自动放行并触发一次 discard_potion
3. 若仅有 discard_potion 且槽未满，仍保持等待态，避免无意义丢药

该改动用于减少“槽位满但无法拿新药水”的僵持状态。

## 11. 下一步可执行清单

1. 跑到 20k 步，记录纯 RL 基线窗口指标
2. 从同一 checkpoint 开 RL+LLM 小权重分支
3. 对比 5k-10k 步窗口后决定是否转主线
4. 若转主线，再引入概率偏置分支（alpha 渐进）
