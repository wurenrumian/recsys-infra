---
title: Conditional Memory via Scalable Lookup - A New Axis of Sparsity for Large Language Models
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-C, Engram, conditional-memory, n-gram, hash-lookup, MoE, sparsity-allocation]
related: [docs/papers/nestpipe/index.md]
zotero_key: A8BIBZTJ
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Xin Cheng、Rui Tian、Wangding Zeng、Damai Dai、Qinyu Chen、Bingxuan Wang、Zhenda Xie、Kezhao Huang、Xingkai Yu、Chengqi Deng、Shangyan Zhou、Chenggang Zhao、Zhewen Hao、Yukun Li、Han Zhang、Zhengyan Zhang、Yixuan Wei、M.Y Xu、Zhang Huishuai、Zhao Dongyan、Liang Wenfeng（**事实**，正文作者块）。
- 单位：**DeepSeek**（+ 北大等，**推断**：作者群与 `deepseek-ai/Engram` 仓库、tokenizer 用 DeepSeek-v3）。
- 年份：2026（arXiv v1 2026-01-12，v2 2026-07-12）。
- 会议：arXiv preprint，`https://arxiv.org/abs/2601.07372`（**事实**）。
- 代码：`https://github.com/deepseek-ai/Engram`（**事实**，摘要）。
- 取到的正文：`https://arxiv.org/html/2601.07372`（**事实**）。

## 一句话

MoE 之外再开一条稀疏轴：用哈希查 N-gram 表当下一个"内存层"，查得比算得快。

## 背景与问题

### 论点：语言建模有两类子任务

这是全篇的立论基石，而且说得比多数论文更硬（**事实**，§1）：

> "language modeling entails two qualitatively different sub-tasks: compositional reasoning and knowledge retrieval. While the former demands deep, dynamic computation, a substantial portion of text—such as named entities, idioms, and formulaic phrases—is **static and locally recoverable**."

翻译成系统语言：**组合推理**需要深的、动态的计算，适合用参数（MoE 专家）来承载；**知识检索**的大量文本内容是**静态的、局部的、可查的**，用查找比用计算划算得多。而现状是（**事实**，§1）："Transformers lack a native primitive for knowledge lookup, forcing them to inefficiently simulate retrieval through computation." —— **没有查表原语，只能拿算力去"模拟"查表**，这是巨大浪费。一个具体的例子（**事实**，§6.1 + Table 3）：要识别实体 "Diana, Princess of Wales"，LLM 必须消耗**多层**注意力 + FFN 一层层把特征组合出来（引 Li & Subramani 2025、Ghandeharioun et al. 2024 的 PatchScope 分析），而这些信息本可以直接查表得到。

### 目标：给稀疏性开第二条轴

MoE 是**条件计算**（稀疏地激活参数来处理动态逻辑），Engram 提出的是**条件内存**（用稀疏的**查找**操作检索静态 embedding）（**事实**）。两条轴的分工很清楚（**事实**，§1）：

| | MoE（条件计算） | Engram（条件内存） |
|---|---|---|
| 稀疏性来源 | 路由选专家 | 哈希选槽位 |
| 承载什么 | 动态逻辑、组合推理 | 静态知识、局部模式 |
| 单 token 代价 | 随激活专家数变 | **O(1)**（常数次查表） |
| 参数与计算 | 耦合（参数多→激活多） | **解耦**（表可以 100B，激活仍是常数） |
| 寻址 | 运行时动态（依赖 hidden state） | **确定性**（只依赖 token ID） |

最后一行（确定性 vs 动态）不是建模差异，是**系统差异**——它决定了能不能预取，是本篇后半段所有效率论证的根源。

## 核心做法

### 1. Engram 模块：三个阶段

```
输入序列 X + 第 ℓ 层的 hidden state H^(ℓ)
        │
        ├─► 阶段 1：N-gram 提取 + 哈希查表 → 静态记忆向量 e_t
        │      （tokenizer 压缩 → suffix N-gram → K 个头 × 多个阶 的哈希 → 拼接）
        │
        ├─► 阶段 2：上下文感知门控 → ṽ_t = α_t · v_t
        │      （当前 hidden state 当 Query，检索到的记忆当 Key/Value）
        │
        ├─► 阶段 3：轻量深度可分离卷积 → Y（扩大感受野 + 非线性）
        │
        └─► 残差：H^(ℓ) ← H^(ℓ) + Y → 再接标准 Attention 与 MoE
```

（**事实**，§2）

**阶段 1 的两个细节。** (a) **词表投影**：标准 subword tokenizer 追求无损重建，语义等价的写法会拿到不相交 ID（原文例子 `Apple` vs `␣apple`），稀释 N-gram 语义密度；解法是预计算映射 `x'_t = P(x_t)` 把原始 ID 映到规范 ID（**事实**）。(b) **多头哈希**：给所有 N-gram 建表不可行（组合爆炸），故用哈希（引 Tito Svenstrup et al. 2017）（**事实**）：

```
z_{t,n,k} = φ_{n,k}(g_{t,n})           # 第 n 阶、第 k 个头的哈希索引
e_t = ‖_{n=2..N} ‖_{k=1..K} E_{n,k}[z_{t,n,k}]   # 查表并拼接所有阶、所有头
```

`φ` 是轻量的**乘法-XOR 哈希**；每头有独立的、**素数大小** `M_{n,k}` 的表来降碰撞（**事实**）。这个结构与推荐系统的**多哈希嵌入表**几乎同构（**推断**：Track B 的 CAFE+ 等在做同样的"多哈希降碰撞"）——Engram 本质上是把推荐系统的哈希嵌入技法搬进 LLM 当一层。

**阶段 2：门控（与"简单平均"的关键差别）。** 检索到的 `e_t` 是**上下文无关**的先验，会有碰撞与多义性带来的噪声（**事实**），故用当前 hidden state 做 Query 去调制：`k_t = W_K e_t`、`v_t = W_V e_t`、`α_t = σ(RMSNorm(h_t)ᵀ·RMSNorm(k_t)/√d)`、`ṽ_t = α_t·v_t`。语义解释（**事实**）：检索到的记忆若与当前上下文矛盾，`α_t → 0`，噪声被抑制——这是"用动态状态剪枝静态记忆"的机制。

**阶段 3：短卷积。** `Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ`，kernel size = 4，dilation = 最大 N-gram 阶（**事实**）。

### 2. 与多分支 backbone 的适配

Engram 与拓扑无关，但默认 backbone 是**多分支**架构（mHC 系），故做了参数共享（**事实**，§2.4）：**共享**一张稀疏嵌入表 + 一个 `W_V`；**不共享** M 个分支各自的 `W_K^(m)`，即 `α_t^(m) = σ(RMSNorm(h_t^(m))ᵀ RMSNorm(W_K^(m) e_t)/√d)`、`u_t^(m) = α_t^(m)·(W_V e_t)`。这么切的原因是（**事实**）：所有线性投影（1 个 `W_V` + M 个 `W_K`）**可融合成一个稠密 FP8 矩阵乘**，所以"多分支门控"在系统层几乎没有额外成本。

### 3. 系统层：确定性寻址 → 可以预取（NestPipe 最关心的一节）

这一节是本篇与 NestPipe 的核心接口（**事实**，§2.5）。**训练**走标准模型并行——表分片到各 GPU，前向 All-to-All 收集激活行、反向分发梯度，总内存容量随加速器数线性扩展。**推理**则因为寻址确定性可以预取：索引在 forward 之前就算得出来（只依赖 token ID），于是可以从**主机内存**经 PCIe 异步取 embedding，与**前面几层的计算重叠**、把延迟藏掉。

关键在于：**Engram 放在哪一层，决定了有多少计算可以拿去藏延迟**（**事实**）。放得越深，可用来重叠的计算窗口越长；但建模上"越早介入越能卸载早期层的局部模式重建"。**这是一个建模与系统的显式 trade-off**，论文 §6.2 的消融同时考虑两端。

还有一个统计性质被利用（**事实**）：自然语言 N-gram 服从 **Zipf 分布**——少部分模式占绝大多数访问，这直接支持**多级缓存层次**（GPU HBM / Host DRAM / 更慢的层）。

实测（**事实**，§6.4）：把**100B 参数**的 Engram 表**全部**放在主机内存里，吞吐损失**最高仅 2.8%**（8B backbone 上）。原因有两条：早期稠密块的计算强度足够大，能遮住检索延迟；**每步的有效通信量随"激活槽位数"增长，而不随表大小增长**。第二条是本质——这是 O(1) 查找而不是线性扫描，所以"表越大"几乎免费。

### 4. 分配律：这篇最有价值的研究产出

作者把"MoE 与 Engram 怎么分预算"定义成一个问题（**事实**，§3.1）：给定固定总参数 `P_tot` 与每 token 激活参数 `P_act`（即固定 FLOPs），定义分配比例 `ρ ∈ [0,1]` = 分给 MoE 专家的比例，则 `P_MoE^(sparse) = ρ·P_sparse`、`P_Engram = (1−ρ)·P_sparse`，其中 `P_sparse = P_tot − P_act` 是"免费"的非激活预算；`ρ=1` 是纯 MoE、`ρ=0` 是纯 Engram。

结果（**事实**，§3.1 + Fig.3 左）：验证 loss 对 ρ 呈 **U 形**（两个算力档 2e20、6e20 FLOPs 都是）；MoE 只占 **~40%** 时已能追平纯 MoE；**最优**在 ρ ≈ **75%–80%**（即把 20–25% 稀疏预算给 Engram），且跨两个算力档稳定；10B 档 Val loss 从 1.7248（ρ=100%）降到 **1.7109**（ρ≈80%），Δ = **0.0139**；全程稀疏率固定 `P_tot/P_act ≈ 10`。

U 形两端的解释（**事实**）：`ρ → 100%`（纯 MoE）**缺少专门存静态模式的容量**，只能靠深度和计算低效重建；`ρ → 0%`（纯 Engram）**失去条件计算能力**，需要动态、上下文相关推理的任务受损——"memory cannot replace computation in this regime"。这是对"查表万能论"的直接否定。

第二个实验是"无限内存区制"（**事实**，§3.2）：固定 MoE backbone（`P_tot ≈ 3B`、`P_act = 568M`，训 100B token），加 Engram 表并把槽位数从 `2.58e5` 扫到 `1.0e7`（最多约 **13B** 参数）。结果 Val loss 对槽位数呈**严格幂律**（log 空间线性）；对比 OverEncoding（N-gram 嵌入平均法），同样内存预算下 Engram 的缩放潜力大得多。**这两条合起来才是"新轴"的完整论证**：固定预算下有最优混合比，内存单独放大时收益又可预测地继续——Engram 既是"替代 MoE 的一部分"，也是"独立可扩展的一维"。

### 5. 主实验设置

四个模型，全部 **262B token**、**激活参数 3.8B 对齐**、同一数据课程、DeepSeek-v3 tokenizer（128k 词表）、30 个 Transformer block（**事实**，§4）：**Dense-4B**（基线，每 block 标准稠密 FFN）；**MoE-27B**（用 DeepSeekMoE 替换 FFN，**72 routed + 2 shared**、top-k=6，总参 26.7B）；**Engram-27B**（严格从 MoE-27B 派生：routed **72 → 55**，省下的参数给 **5.7B** Engram，ρ = **74.3%**，总参不变）；**Engram-40B**（同 backbone、同算力预算，Engram 放大到 **18.5B**，总参 39.5B）。

### 6. 机制分析：Engram ≈ 加深度

这是本篇最有意思的部分（**事实**，§6.1）。**LogitLens** 把每层中间 hidden state 用最终 LM Head 投影，算中间分布与最终分布的 **KL 散度**——Engram 两个变体的 KL 散度**系统性更小**，且**差距在早期 block 最明显**，说明特征组合"更快完成"。**CKA（Centered Kernel Alignment）** 算 Engram 层与 MoE 层的表示相似度矩阵 `S ∈ [0,1]^{L×L}`，再定义**软对齐指标** `a_j = Σ_{i ∈ I_j} S_{i,j}·i / Σ_{i ∈ I_j} S_{i,j}`（`I_j = argtop-k_i(S_{i,j})`，k = 5），即"Engram 第 j 层在 MoE 里对应的有效深度"。结果（**事实**）：`a_j > j` 在很宽的层区间成立——例如 **Engram 第 5 层的表示最贴近 MoE 第 8 层**。结论（原文）："by bypassing early-stage feature composition via explicit lookups, Engram is functionally equivalent to increasing the model's depth."

**一个反向证据很有说服力**（**事实**，§6.3 + Fig.6）：推理时**直接把 Engram 输出抑制掉**（保留 backbone），出现**功能二分**——事实性知识类基准**崩溃**、只保留 **29%–44%**（TriviaQA 掉到 29%），而阅读理解类基准**基本保留**（由 backbone 撑住）。这证明 Engram 确实是"知识库"，不是"通用容量"。

### 7. 放置策略的 trade-off

单层插入 vs 分层插入（**事实**，§6.2）：

| 配置 | Val loss |
|---|---|
| 3B MoE 基线 | 1.808 |
| 单模块放 **Layer 2**（最优单层位置） | **1.770** |
| 拆成两个模块（同样 1.6B 预算，减小 `d_mem`）放 **Layer 2 + Layer 6** | **1.768** |
| 参考配置（`{2,3}`-gram，Layer 2 与 6） | 1.768（Δ = **0.04** vs 基线） |

为什么 Layer 2 最优（**事实**）：放 Layer 1 太早（门控精度差，hidden state 还没聚合足够全局上下文），放深了又失去"卸载底层局部聚合"的价值。**一轮注意力已足够提供有意义的上下文来做门控**。分层插入更好则因为它"调和了 trade-off"——早期介入 + 后期用更强的上下文做门控，且还有一个**系统上的好处**：更好地利用内存层次（不同层级的表可放在不同存储层）。

## 关键数字

| 指标 | 数值 | 等级 |
|---|---|---|
| Engram 最大规模 / 预训练 token | **27B**（40B 变体也测了）/ **262B** | 事实 |
| 激活参数对齐 | 3.8B | 事实 |
| MMLU | **+3.4**（摘要）/ **+3.0**（§4 正文，对 MoE-27B） | 事实（两处不一致） |
| CMMLU / MMLU-Pro | **+4.0** / +1.8 | 事实 |
| BBH / ARC-Challenge | **+5.0** / **+3.7** | 事实 |
| HumanEval / MATH | **+3.0** / **+2.4** | 事实 |
| Multi-Query NIAH / Variable Tracking | **84.2 → 97.0** / 77.0 → **89.0** | 事实 |
| 100B 表全放主机内存的吞吐损失 | 最高 **2.8%** | 事实 |
| 稀疏预算最优分配 / 10B 档 loss 改善 | ρ ≈ **75%–80%** 给 MoE / 1.7248 → **1.7109** | 事实 |
| 等价深度（Engram 第 5 层 ≈ MoE 第 8 层） | 差 ~3 层 | 事实 |
| 抑制 Engram 后知识类基准保留率 | **29%–44%** | 事实 |
| 长上下文：用 **82%** 算力（41k vs 50k 步） | 打平基线 LongPPL，且 RULER 更好 | 事实 |

**最值得注意的数字是"100B 表只掉 2.8% 吞吐"**——它把"参数规模"与"系统成本"解耦了，这正是 NestPipe 这条线关心的核心命题。

## 为什么值得读

- **它给"稀疏"开了一条新轴，而且系统含义与 MoE 完全不同**。MoE 的稀疏是"激活哪个专家"，参数与计算仍耦合（专家多 → 路由面大 → 通信涨）；Engram 的稀疏是"查哪个槽位"，**O(1) 查找**，参数与计算彻底解耦。这个区别在系统层是根本性的：MoE 越大通信越多，Engram 越大通信几乎不变。
- **与 C-02 HSTU / C-03 MTGR 的"嵌入表"是同一个物理对象，但用法不同**。推荐系统的嵌入表是"按 ID 查特征"，Engram 是"按 N-gram 哈希查知识"。两者都是**大规模稀疏查表 + 确定性寻址**，因此可共用系统基建（分片、预取、缓存层次）（**推断**）——这是 Track C 与 Track B 的直接连接点。
- **与 C-06 DMT 的"拓扑感知"互补**：DMT 优化"查表请求怎么走网络"，Engram 优化"查表能不能不走网络"（放主机内存 + 预取）。**两条路合起来才是完整的嵌入系统设计**（**推断**）。
- **与 C-13 Comet 的 MoE 视角形成对照**：Comet 说 MoE 的通信占 47%、要重叠；Engram 说"别用 MoE 做知识检索，用查表"，从根本上减少需要路由的东西。**一个是优化 MoE，一个是质疑 MoE 的适用范围。**
- **两个可复用的产出**：U 形分配律把"稀疏预算怎么花"变成可测量的曲线，且最优点跨两个算力档稳定（ρ ≈ 75–80%）；"Engram ≈ 加 3 层深度"这个发现把"查表"从"便宜的近似"提升为"与深度等价的容量形式"，且用 CKA 软对齐做定量、方法干净。
- **对 NestPipe 的独特价值**：这是本 track 里唯一明确写出"**确定性寻址 → 可预取 → 通信量与表大小无关**"的论文。这句话基本就是"嵌入表该怎么做大规模存储"的第一性原理（Track B 的多篇都在用这个原理，但很少有人这么直白地表述）。

## 局限

1. **只有语言模型**，没有推荐系统的实验。虽然机制（N-gram 哈希查表）与推荐系统的特征交叉很像，但"用户的 N-gram"是什么、怎么构造，论文完全没碰（**推断**）。
2. **哈希碰撞的代价没有独立量化**。论文说用 K 个头 + 素数表降碰撞，但没有"碰撞率 vs 性能"的曲线（**待核实**）——而这是哈希嵌入方案的关键指标。
3. **`d_mem` 与槽位数的分配未系统研究**。分层插入实验里"减小 `d_mem`"是个关键动作，但没有"总预算固定下，维度 vs 槽位数怎么分"的规律（**待核实**）。
4. **摘要与正文的数字不一致**：MMLU 在摘要写 **+3.4**、§4 正文写 **+3.0**（**事实**，两处原文并存），可能一个是 vs Dense、一个是 vs MoE。引用时必须注明比较对象。
5. **100B 表放主机内存的实验是在 "dense backbone" 上做的**。论文明确说为了避开 Expert Parallel 混淆通信模式才这么选（**事实**），也就是说**真实 MoE + EP + Engram 预取**的组合场景没测（**待核实**）。
6. **Engram 对"多语言/多领域"的影响未拆**。除 CMMLU / C-Eval 有中文基准，其余是一个混语料的大锅（**推断**）。
7. **训练侧的 All-to-All 并未优化**。论文说训练用标准模型并行 + All-to-All（**事实**），即**训练时 Engram 表的分片通信问题与传统嵌入表一样存在**，只有推理受益于预取。这点容易被误读成"Engram 省通信"——省的是推理（**推断**）。
8. **N-gram 阶数只试到 4-gram**，且 4-gram 在固定 1.6B 预算下略差（稀释了高频 2/3-gram 的容量）（**事实**）。更大预算下高阶是否更好，论文说 "do not rule out"（**待核实**）。

## 证据与待核实

| 结论 | 等级 | 依据 |
|---|---|---|
| 作者、DeepSeek 关联、arXiv 编号、代码仓库 | 事实 | 摘要 + 正文作者块 + 摘要链接 |
| 三阶段架构（N-gram 哈希 → 门控 → 卷积）、词表投影、多头哈希 | 事实 | §2.1–2.4 |
| U 形分配律、最优 ρ ≈ 75–80%、10B 档 Δ=0.0139 | 事实 | §3.1 + Fig.3 |
| 无限内存区制的幂律 | 事实 | §3.2 |
| 27B/40B、262B token、3.8B 激活参数 | 事实 | §4 + Table 1 |
| 各项基准提升（含 MMLU 的 3.4 vs 3.0 分歧） | 事实（数值）/ **待核实**（口径） | 摘要 vs §4 |
| CKA 软对齐、等价深度 ~3 层 | 事实 | §6.1 |
| 抑制 Engram 后知识类掉到 29–44% | 事实 | §6.3 + Fig.6 |
| 放置策略（Layer 2 最优、2+6 更好） | 事实 | §6.2 |
| 100B 表放主机内存只掉 2.8% | 事实 | §6.4 + Table 4 |
| "训练侧通信同样省" | **推断（否定）** | 论文说训练仍用标准 All-to-All |
| 迁移到推荐系统 | **推断** | 论文未涉及 |
| 哈希碰撞率的定量曲线 | **待核实** | 未给 |
| MoE + EP + Engram 预取的联合场景 | **待核实** | 只在 dense backbone 上测 |
| 是否发表于正式会议 | **待核实** | arXiv abs 无 venue |

没拿到的东西：正式发表版（若有）；MMLU 数字口径的确认；附录（哈希实现细节、更多基准的完整表格）；100B 表实验的完整延迟分解。

## 是否值得读原文

**必读**。

理由：这是本 track 里唯一一篇"给稀疏性开新轴"的论文，而且它的系统含义（确定性寻址 → O(1) 查找 → 通信量与表大小解耦 → 可预取）直接对应 NestPipe 的核心命题。必读 §2（模块三阶段）、§3（U 形分配律 + 幂律）、§6.1（等价深度的机制分析）与 §6.4（100B 表放主机内存的系统实验）。§4 的基准表可当数字字典查。
