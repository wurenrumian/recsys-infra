---
title: "Climber: Toward Efficient Scaling Laws for Large Recommendation Models"
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-A, scaling-law, generative-recommendation, inference-latency, kv-cache]
related: [docs/papers/nestpipe/index.md]
zotero_key: ABZSE7KJ
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Songpei Xu、Shijia Wang、Da Guo、Xianwen Guo、Qiang Xiao、Bin Huang、Guanlin Wu、Chuanjiang Luo（网易云音乐，全部作者同一单位）。
- 年份与会议：2025 年，CIKM '25（第 34 届 ACM 信息与知识管理会议，Seoul），DOI `10.1145/3746252.3761561`（**事实**：Zotero 元数据 + CrossRef + arXiv HTML 首页版权页）。
- 公开全文：`https://arxiv.org/abs/2502.09888`，正文取到 `https://arxiv.org/html/2502.09888`（**事实**）。

## 一句话

把长序列拆成多尺度子序列 + 自适应温度，让 Transformer 的 scaling 曲线变陡同时不爆线上延迟。

## 背景与问题

作者的主张很尖锐：**Transformer 在推荐系统上 scaling 得不理想，不是因为算力不够，而是因为架构与推荐特征结构不兼容**（**事实**：摘要第一条）。

两个具体的不兼容点：

1. **多源数据异构**。用户的"生命周期序列"里混着点击、点赞、分享、播放等不同行为，还有多个场景（首页推荐、歌单、搜索）的数据。直接拼成一条序列喂一个 Transformer，注意力分布被强行拉平，多场景多行为的差异被抹掉。
2. **线上延迟硬约束**。推荐推理预算是**几十毫秒**（tens of milliseconds），而序列长度和层数一涨，这个预算立刻被击穿。NLP 那边可以靠批量换吞吐，推荐这边是延迟敏感的单请求。

用作者的话说，工业部署里"单纯靠扩资源来扩模型"这条路"provably inefficient"（**事实**）。

所以 Climber 的目标不是"把模型做大"，而是**让曲线本身更陡**——在小算力区间就超过 DLRM，把 crossover point 往左推。

## 核心做法

Climber 由两部分咬合而成：**为高效 scaling 设计的模型架构** + **与架构共同设计的加速手段**。两者不是可选项，缺一半就不成立。

### 一、模型侧：三处结构改动

```
用户生命周期序列 S（长，可达万级）
   │
   ├─[抽取策略 1: 业务驱动 e.g. 点击]→ S1 ─→ ATL block 1 ─┐
   ├─[抽取策略 2: 业务驱动 e.g. 点赞]→ S2 ─→ ATL block 2 ─┤
   ├─[抽取策略 3: 模型过滤]        → S3 ─→ ATL block 3 ─┼→ 拼接 → ATL → sigmoid 逐位门控 → 融合向量
   └─[抽取策略 m]                  → Sm ─→ ATL block m ─┘
                                                          │
                                              候选 item 特征 ─→ 注意力交互 → 打分
```

**（1）Multi-Scale Sequence Extraction（MSE，多尺度序列抽取）**

把一条长序列按不同策略重组成 m 条等长子序列。复杂度从 O(L²) 降到 O(L²/m)（串行时），每段由独立 Transformer block 处理，**全并行时复杂度只取决于最长子序列长度**（**事实**：论文 3.1.2 的复杂度推导）。

这带来一个很实用的性质：**渐进式扩容**。序列长度从短往长加时，训练效率不会掉——短序列段和长序列段可以并行。作者说实践中 m 取一个小常数（正文里公式被剥离，具体数值**待核实**）。

**（2）Multi-Scenario Scale-Up：Adaptive Transformer Layer（ATL）**

标准 Transformer 里 softmax 前除以 √d 是个**固定**缩放因子。HSTU 用固定温度系数，Climber 改成**每个 block 的每一层各自学一个温度系数**，同时引入相对注意力偏置（含位置 p 与时间 t 信息）。理由：不同抽取策略产生的子序列分布差异极大，单一温度无法同时适配（**事实**：3.1.3 明确对比 HSTU 的固定温度）。

温度函数 τ = f(抽取策略, 场景)，是个可学习的映射。

**（3）Multi-Interest Scale-Up：逐位门控融合**

分段处理后各段之间没有交互，这是 MSE 的直接副作用。Climber 把各 block 的输出向量拼接，过一个新的 ATL，再经 sigmoid 做**bit-wise gating**，最后过一层网络。相当于用门控决定"这次预测该听哪个兴趣侧面"。

### 二、部署侧：两阶段加速

**离线训练**：把原始"单用户单 item"日志**压缩成"单用户多 item"**记录。一条记录里放一个用户的历史 + 多个候选 item，用 full-visible mask 让每个候选看完整历史，再用**对角掩码**做候选之间的隔离（防止信息泄漏）。

这个压缩直接带来 **5.15× 训练加速**（**事实**：3.2 原文 "providing 5.15 acceleration for model training"）。

**在线服务**：沿用 M-FALCON 的思路——先用用户特征算出多层 KV cache，再去特征服务取候选 item 特征，最后只算候选与 KV 的注意力。因为训练与服务用同一套"单用户多 item"数据形态，KV cache 可以直接复用。另加算子融合（embedding lookup + attention 合成单 kernel）与 FlashAttention 分块。

### 三、三者的耦合关系

| 组件 | 换来什么 | 代价 |
|---|---|---|
| MSE 分段 | 复杂度 ↓ O(L²)→O(L²/m)，可渐进加长 | 段间失联，需门控补回 |
| ATL 自适应温度 | 异构场景/行为可共存 | 每层多一组参数，温度函数需训练 |
| 逐位门控 | 补回段间交互 | 额外一层 ATL 计算 |
| 单用户多 item + KV cache | 训练 5.15×，推理 2.92–14.38× | 数据管线要改写，掩码设计变复杂 |

## 关键数字

| 指标 | 数值 | 来源 |
|---|---|---|
| 训练吞吐提升 | **5.15×**（单用户多 item 批处理所致） | **事实**，摘要 + 3.2 |
| 线上综合指标提升 | **+12.19%**（最大档 Climber 620 vs DLRM） | **事实**，4.5 + 摘要 |
| 线上中间档提升 | +7.78%（Climber 479 档） | **事实**，4.5 |
| 推理单请求延迟 | 比 DLRM **快 2.92–14.38×**（随序列长度/层数变化） | **事实**，4.5 |
| 可部署复杂度 | 同等推理预算下可部署 **100×** 更复杂的模型 | **事实**，4.5 原文 |
| 小算力区间表现 | FLOPs 低于某个阈值时 Climber 指标**负增长**，crossover 点比 Transformer 左移 | **事实**，4.5 + 4.3 |
| 规模 | 网易云音乐，日服务数千万用户 | **事实**，摘要 |

作者自称这是"首个公开记录的、受控模型 scaling 能带来线上指标持续增长（+12.19%）的框架"（**事实**：摘要原文 "first publicly documented framework where controlled model scaling drives continuous online metric growth"）。这句话要打折看——"公开记录的"是个限定词，不代表业界没有。

## 为什么值得读

在 Track A 里，Climber 是**唯一一篇把"线上延迟"当作一等约束写进设计目标**的文章。其它 16 篇（A-03 到 A-15）基本都在优化"训练吞吐"，默认推理侧另有人管；Climber 反过来：先定推理预算，再倒推训练侧该怎么做。

与同 track 的关系：

- **A-01（Scaling Law of Large Sequential RecSys）是它的直接对手**。A-01 用纯 ID + 标准 transformer 得出"幂律成立、扩就完事"；Climber 说"幂律成立但斜率不够，标准 transformer 在推荐上效率低于 DLRM，必须改架构"。两篇的分歧点不在数据，在**是否接受标准 transformer 作为 baseline**。做技术选型时这两篇必须对着读。
- **A-17（LLM-Based RecSys 的并行优化）是下游**：Climber 解决"单模型结构效率"，A-17 解决"模型已经很大之后多卡怎么切"。
- **C-02（HSTU）是它的参照物**：Climber 明确说自己的自适应温度是相对 HSTU 固定温度的改进，并把 HSTU 当 related work 的核心。这篇可以看作"HSTU 的工业改良版 + 延迟工程"。
- **C-05（Bending the Scaling Law Curve）主题最接近**：都在谈"让曲线变弯"。Climber 给的是架构层面的弯法，C-05 给的是别的层面。

它也是全 track 里**唯一给出"负收益区间"**的文章——明确承认 FLOPs 低时打不过 DLRM。这种诚实度在同类论文里少见，做决策时有用：说明它不是免费的，是有适用区间的。

## 局限

- **单公司、单场景验证**。全部数据来自网易云音乐，音乐推荐的行为模式（播放/收藏/分享）与电商、短视频、广告差异很大。"多尺度抽取策略"在音乐上有效，因为行为类型天然可分；在只有一两种行为的场景里，MSE 的分段收益会大幅缩水（**推断**）。
- **m 的取值、温度函数的具体形式、门控的实现细节在 HTML 版里公式被剥离**，无法复现（**待核实**）。
- **"5.15× 训练加速"归因模糊**。它来自数据形态压缩（单用户单 item → 单用户多 item），本质是**样本量变少**而不是算子变快。这会让等效 epoch 定义发生变化，跨论文比较加速比时不可直接对比（**推断**）。
- **crossover 点与具体阈值未在可读文本中给出数字**（4.3 节里 FLOPs 的数值都是公式占位），所以"什么时候该用 Climber、什么时候该继续用 DLRM"这个最关键的操作性问题实际上没答（**待核实**）。
- **没有讨论训练侧的多卡扩展**。全篇聚焦单模型效率，序列再长、层再多之后跨卡怎么切完全没提——这正是 A-17 的空档。
- 线上 A/B 只报了综合指标提升，**没有分场景/分用户群的拆解**，无法判断是否只是头部用户受益（**推断**）。

## 证据与待核实

- 证据等级：**fulltext**。正文取自 `https://arxiv.org/html/2502.09888`（HTML 版），章节完整（Introduction / Related Work / Method / Experiments / Conclusion）。
- **事实**：作者与单位、会议与 DOI、三处结构改动的机制描述、ATL 与 HSTU 固定温度的对比、单用户多 item 的掩码设计、5.15× 训练加速、+12.19% / +7.78% 线上提升、2.92–14.38× 延迟改善、100× 复杂度空间、FLOPs 低区间负收益。
- **待核实**：
  1. 抽取策略数量 m 的实际取值（正文 "In our practical application, m = …" 的数值被公式剥离）；
  2. MSE 复杂度的完整表达式与 O(L²/m) 的推导假设；
  3. ATL 中温度函数 f(·) 的具体形式与相对注意力偏置的构造；
  4. 4.3 节里 crossover 的 FLOPs 具体数值；
  5. 离线实验用了哪些数据集、baseline 有哪些（4.1 节在可读文本中只剩标题）；
  6. 5.15× 加速的基线定义（是相对"单用户单 item"训练还是相对其他系统）。
- Zotero 内无 PDF 附件（集合内 50 条均无，属预期）。

## 是否值得读原文

**必读**——如果你的目标是"在延迟约束下扩推荐模型"，这篇是目前唯一同时给出架构改动、部署改动和线上数字的完整案例。重点读 3.1（三处结构改动的公式）与 3.2（部署与 KV cache 复用），以及 4.3/4.5（crossover 与线上曲线）；这两部分恰好是 HTML 版丢失最多的，必须拿 PDF 才能拿到可复现的细节。
