---
title: Scaling Law of Large Sequential Recommendation Models
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-A, scaling-law, sequential-recommendation, transformer, lstm-free]
related: [docs/papers/nestpipe/index.md]
zotero_key: RJYMAGIN
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Gaowei Zhang、Yupeng Hou、Hongyu Lu、Yu Chen、Wayne Xin Zhao、Ji-Rong Wen（中国人民大学高瓴人工智能学院 / 腾讯）。
- 年份与会议：2024 年，RecSys '24（第 18 届 ACM 推荐系统会议），DOI `10.1145/3640457.3688129`（**事实**：Zotero 元数据 + CrossRef）。
- 公开全文：`https://arxiv.org/abs/2311.11351`，正文取到 HTML 版 `https://arxiv.org/html/2311.11351`（**事实**）。

## 一句话

ID-only 序列推荐模型从 98.3K 扩到 0.8B 参数时，scaling law 依然成立，且小模型能预测大模型。 

## 背景与问题

NLP 与 CV 已经证明 loss 与参数量、数据量呈幂律关系，但推荐系统这边只有两类零散证据：一是富特征 CTR 模型（DLRM 谱系），二是基于文本的推荐模型。**既没有系统研究纯 ID 序列推荐模型**，也说不清推荐领域特有的数据稀疏/噪声会不会让幂律失效。

作者选的设定非常克制：把用户交互历史写成按时间排序的 item ID 序列，**完全不引入任何 side information（item 文本、图像、类目）**。理由是要验证"仅从用户行为本身"能否撑起 scaling——如果加了文本特征，涨点到底来自语言模型还是行为建模就分不清了（**事实**：论文原文明确说明不做 tokenization、item 集合直接当词表）。

难点有两个：

1. **训练不稳定**。大模型在推荐任务上很容易训崩，作者说"经验上发现大规模 transformer 在推荐任务上难以训练"（**事实**）。
2. **数据受限**。最大的模型 0.8B 参数，而可用数据集只有 18.5M 条交互——参数量比 token 数还多一个量级。NLP 里"data-constrained scaling"是有专门形态的，推荐要比它更极端。

## 核心做法

模型本身没有新东西：**decoder-only transformer，因果掩码，结构等价于 SASRec**，用 HuggingFace 实现、RecBole 训练。真正的贡献是"把规模做上去的两条工程策略 + 三个拟合结论"。

架构与数据的配置（**事实**）：

| 项 | 取值 |
|---|---|
| 骨干 | decoder-only Transformer（SASRec 式因果结构） |
| 词表 | item ID 集合，不做 tokenization |
| 位置编码 | 可学习 |
| 最大序列长度 | 50（超截断、短补齐） |
| 训练目标 | 下一 item 的交叉熵 |
| 参数量扫描 | 98.3K → 0.78M → 9.44M → 75.5M → 0.8B（非嵌入参数） |
| 数据量扫描 | 1.8M / 9.2M / 18.5M 交互，全局 batch 固定 |

数据集（**事实**，来自论文 Table II）：MovieLens-20M 有 138,493 用户 / 26,427 item / 18,476,840 交互；Amazon(mix) 有 367,710 用户 / 240,320 item / 21,787,957 交互。

两条稳定训练的招式：

- **Layer-wise Adaptive Dropout（逐层自适应 dropout）**：不同深度的层给不同 dropout 强度，缓解深层梯度问题。
- **Switching Optimizer Strategy（切换优化器）**：训练过程中换优化器（**待核实**：具体切换点与两个优化器的搭配，正文里被公式/段落错位吃掉了，只能确认"切换"这一事实）。

数据流没什么可画的——单塔前向、next-item 交叉熵、反向传播。值得画的是**"要花的算力往哪儿投"的决策树**：

```
给定算力预算
  ├─ 加宽（width/embedding dim）      → 收益差，甚至出现 embedding collapse
  ├─ 加深（layer 数）+ 加数据          → 幂律收益，作者推荐的路径
  └─ 改模型形状（aspect ratio）        → 影响弱，不值得大规模搜超参
```

## 关键数字

- 规模上探到 **0.8B** 非嵌入参数，是"推荐领域当时公开的最大 ID-based 序列模型"（**事实**：摘要与正文）。
- **用小模型（<100M）的 loss 成功预测 0.8B 模型的 test loss**，预测点与实测点"高度吻合"（**事实**：Figure 2(a) 的描述）。这是本文最有价值的结论——意味着扩规模前可以先算账。
- LSRM（最大档）总参数是默认 SASRec 的 **8000 倍**、默认 Caser 的 **1000 倍**，在 HR@5 / NDCG@5 / NDCG@10 上"几乎两倍"优于 SASRec、GRU4Rec、FMLP、Caser（**事实**：Table V 描述）。
- 只加宽不加深的变体性能**明显下降**——作者归因为 embedding collapse（**事实**）。
- 幂律指数 α：作者称"比语言模型的 α 相对更大"，即推荐领域 loss 随容量下降更快；但**具体 α 数值与 β、L∞ 未能从 HTML 版提取（公式被剥离），需要核对 PDF**（**待核实**）。
- 数据重复（data repetition）在极度受限时能带来额外收益（**事实**：第 IV-C 节标题与结论）。

## 为什么值得读

在 NestPipe 这条线上，A-01 是**唯一一篇"先问值不值得扩，再谈怎么扩"**的文章，其它 16 篇默认"扩是既定方向，只解决扩不动的工程问题"。它的位置是**上游论证**：

- 与 A-02（Climber）是**对照+反例**：A-01 说"ID-only transformer 的幂律和 NLP 差不多甚至更好"，Climber 说"Transformer 在推荐上 scaling 不理想，因为多源异构特征结构不匹配，得改架构"。两篇都做实验，但 A-01 把特征剥干净只留行为，Climber 恰恰把多场景/多行为当作必须处理的输入。**读这两篇必须一起读**，否则会误以为"推荐 scaling 已解决"或"必须换架构"。
- 与 A-16（RecIS）、A-04（Persia）是**互补**：那两篇是系统侧，回答"0.8B 之后怎么再往上 100 倍"，本篇回答"往上是不是还有收益"。
- 与 Track C 的 C-05（Bending the Scaling Law Curve）方向相同但立场相反：C-05 主张要"弯曲线"（用更少算力拿更多收益），A-01 给出的是**未弯的基线幂律**，是判断"弯了多少"的参照系。

它也是全 track 里唯一给出**可预测性**（predictable scaling）实证的一篇——这是做容量规划时最直接可用的东西。

## 局限

- **只有两个数据集，且都是"玩具规模"**。MovieLens-20M 和 Amazon 与工业级序列推荐（十亿用户、百亿行为）差好几个数量级，"0.8B 就 data-constrained"这个结论在真实工业数据上可能完全不成立（**推断**）。
- **纯 ID、无 side info 是刻意简化，也是最大偏差**。真实线上系统几乎不可能不用 item 文本/图像/类目；一旦引入，幂律的斜率、拐点都会变。作者自己承认这是为了"隔离变量"，但这也意味着结论不能直接搬到生产（**推断**）。
- **只测了 loss 与少数离线指标**，没有线上 A/B。0.8B 模型在线上能否扛住延迟约束（几十毫秒）根本没讨论——而这正是 A-02（Climber）的立论起点。
- **NDCG 在 0.8B 处出现"低回报区"**，作者推测是数据量限制，但没有给出"需要多少数据才能突破"的定量答案（**事实**：论文自述 speculation）。
- 两条稳定训练策略只给了名字，切换点、dropout 表的构造方式都需要查 PDF 才能复现（**待核实**）。

## 证据与待核实

- 证据等级：**fulltext**。正文取自 `https://arxiv.org/html/2311.11351`（HTML 版，LaTeXML 生成），章节结构完整（Introduction / Related Work / Experimental Setup / Scaling Laws / Benefits / Conclusion 全部取到）。
- **事实**：作者、年份、会议、DOI、参数量扫描范围、两数据集统计、四条主要结论、8k×/1k× 规模对比、双倍指标提升、embedding collapse 现象、data repetition 收益。
- **待核实**：
  1. 拟合公式里的 α、β、L∞ 具体数值（HTML 中公式被剥离成空占位）；
  2. Switching Optimizer 的两个优化器分别是什么、何时切换；
  3. Layer-wise Adaptive Dropout 的逐层系数如何确定；
  4. Table IV（容量扫描配置表）的逐档 dim/depth/head 配置；
  5. 五个下游任务（long-tail / cold-start / robustness / multi-domain / trajectory）各自的涨点幅度——正文只给了定性结论"大幅提升"。
- 未取到 PDF 附件（NestPipe 集合内 50 条均无附件，属预期）。

## 是否值得读原文

**挑章节读**——第 IV 节（Scaling Laws）和第 III-B 节（Training Techniques）是真正的方法与证据所在，两节加起来足够支撑"要不要扩、怎么扩"的决策；第 V 节的五个下游任务结论可以直接引用摘要，不必逐表读。如果要做可预测性外推（fitting + 预测更大模型），必须读原文的公式与拟合细节。
