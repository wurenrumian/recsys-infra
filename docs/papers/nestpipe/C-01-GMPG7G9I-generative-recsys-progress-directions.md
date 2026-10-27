---
title: Generative Recommendation Models - Progress and Directions
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-C, generative-recommendation, survey, tutorial, semantic-id]
related: [docs/papers/nestpipe/index.md]
zotero_key: GMPG7G9I
zotero_collection: 9I82QPTL
evidence: abstract-only
---

## 出处

- 作者：Yupeng Hou、An Zhang、Leheng Sheng、Zhengyi Yang、Xiang Wang、Tat-Seng Chua、Julian McAuley（**事实**，Zotero 元数据；Crossref 给出单位：UCSD / NUS / USTC）。
- 年份与会议：**The Web Conference 2025（WWW '25）**，出版日期 2025-05-08，ACM 出版，**页码 13–16**（**事实**：Crossref 的 `page` 与 `published-print` 字段；这是 tutorial 摘要型的短篇）。
- DOI：`10.1145/3701716.3715856`（`https://doi.org/10.1145/3701716.3715856`）；arXiv 无对应条目（**事实**，检索 arXiv 未命中同名条目）。
- 许可：CC-BY 4.0（**事实**，Crossref `license`）。
- 类型：**tutorial / 综述**，不是系统实现论文。这一点决定了本篇的读法。

## 一句话

把生成式推荐（LLM 式、语义 ID、扩散式）的路线图讲清楚，是这条线的地图而非新方法。

## 背景与问题

推荐系统长期是**判别式**范式：给定 (user, item)，输出一个打分或概率，再按分数排序（**事实**，摘要："predicting whether items should be retrieved"）。这个范式的能力上限来自两点：

1. **用户是被动的**。用户只能在系统给出的候选里挑，不能主动用自然语言描述需求，也谈不上多轮交互（**事实**，摘要 "Users can only passively browse the recommended items rather than actively express their needs"）。
2. **表达力被特征工程框死**。判别式模型的容量靠堆交叉特征和 MLP，跟 LLM 那套"用算力换能力"的缩放路径不是一回事（**推断**：这是摘要里 "expressive capabilities are limited" 的直白解读，也是 Track C 其它 15 篇的共同动机）。

近两年 LLM 起来后，学界出现了两条改造路线（**事实**，摘要原文）：
- 把**预训练生成模型对齐到用户行为**上（LLM4Rec 路线）；
- 或者**直接在生成式框架里重新设计推荐模型**（HSTU / 语义 ID 那条线）。

两条路线都让系统能收发"更像人的内容"——自然语言、图像，乃至更多模态（**事实**）。问题在于：这两条线各自的代价、适用边界、还差什么，散在几十篇论文里，没有统一坐标系。这篇 tutorial 就是来给坐标系的。

## 核心做法

这不是系统论文，没有架构图和数据流。它的"做法"是**分类学**。按摘要明确点到的四类内容组织（**事实**）：

| 类别 | 一句话定位 | 与本 track 的对应 |
|---|---|---|
| 基于 LLM 的生成式推荐 | 用预训练语言模型做 backbone 或做 item 表示 | 与 C-02 HSTU 的"非 LLM 生成式"路线形成对照 |
| 基于语义 ID（semantic ID） | 把 item 离散成 token 序列，把推荐变成 next-token | C-03 MTGR、C-05 ULTRA-HSTU 的工业落地都在这条线上 |
| 基于扩散模型 | 把生成过程从自回归换成去噪 | 本 track 未覆盖，属待补 |
| 其它 | — | — |

tutorial 的叙事骨架（**推断**，由摘要"overview → challenges → open questions → future directions"四段式推出）：

```
判别式范式的能力边界
        │
        ├── 路线 A：对齐预训练生成模型（LLM4Rec）
        │       代价：item 要变成语言、延迟高、ID 词表与 LLM 词表不对齐
        │
        ├── 路线 B：在生成式框架里重设计（HSTU 家族）
        │       代价：扔掉交叉特征、要重写训练系统
        │
        └── 共同瓶颈
                ├── 语义 ID 怎么构造（tokenizer 设计）
                ├── 训练系统（长序列、KV/激活、embedding 规模）
                └── 推理系统（自回归解码 vs 打分，QPS 差几个量级）
```

最后一格（推理成本）是这篇与本 track 后半段（C-08…C-14）真正的接缝：**生成式推荐把排序从"一次前向"变成"逐步解码"，推理系统的账要重算**（**推断**）。

## 关键数字

- 无。tutorial 论文不给实验数字（**事实**：摘要中不含任何指标）。
- 若要引用具体加速比/规模，请改用 C-02（65.8% NDCG、5.3–15.2×、1.5T 参数）或 C-03（65× FLOPs、1.22%/1.31% 线上提升），那些是原文数字。

## 为什么值得读

- **它是这条 track 的索引页**。C-02 到 C-06 全是"生成式推荐"的具体实现，各自动了系统的一层（模型结构 / 特征重排 / 稀疏注意力 / 拓扑），但没人告诉你这些改动在整体版图里占哪一格。这篇给的就是格子。
- **与 C-02 的关系是"地图 vs 领土"**：HSTU 是路线 B 的原型实现，本篇是路线 A/B 的对照说明。先读本篇再读 C-02，能省掉"为什么 HSTU 要扔掉交叉特征"这个最常见的困惑——而 C-03 MTGR 恰好证明了扔掉交叉特征的代价（**事实**，MTGR 摘要："scaling up cannot compensate for it at all"）。
- **与 C-07 KuaiRand 的关系是"评测 vs 建模"**：生成式推荐的离线评测该用什么数据，本篇把它列为 open question（**推断**），KuaiRand 给了一个带随机曝光的答案。
- **反例价值**：tutorial 类论文天然乐观。C-05 ULTRA-HSTU 的实测显示，跨注意力（很多 LLM4Rec 工作依赖的东西）在长序列上**打不过**自注意力（**事实**，ULTRA-HSTU 原文 "STCA ... performs worse than ULTRA-HSTU due to lacking the power from self-attention"）。读本篇时要带着这个反证。

## 局限

1. **没有系统层论证**。它谈"建模进展"，不谈"这套模型在 GPU 集群上跑起来要多少显存、多少通信"。NestPipe 关心的那一层（存储层级、通信、并行、量化、调度）在这篇里基本缺位（**推断**，依据摘要覆盖面判断）。
2. **tutorial 的时效性极短**。2025 年 5 月的分类，到 2026 年已经漏掉了一整批工业系统（MTGR、ULTRA-HSTU、MTGenRec 等都是其后出现的）（**事实**：Zotero 中这些条目的日期晚于本篇）。
3. **四类划分的边界模糊**。语义 ID 与 LLM-based 常被同一篇论文同时使用，分类更多是叙事需要（**推断**）。
4. **证据等级最低的一篇**：本文件只有摘要级证据，正文（tutorial 幻灯片/讲稿）未取到。

## 证据与待核实

| 结论 | 等级 | 依据 |
|---|---|---|
| 作者、年份、DOI、出版社 | 事实 | Zotero 元数据 |
| WWW '25、页码 13–16、CC-BY 许可 | 事实 | Crossref `page` / `published-print` / `license` |
| 判别式 vs 生成式的范式对立 | 事实 | 摘要原文 |
| 覆盖 LLM / 语义 ID / 扩散三类方法 | 事实 | 摘要原文 |
| 推理成本是生成式推荐的真实瓶颈 | 推断 | 领域常识 + 本 track C-08…C-14 的整体主题 |
| 具体分类树、各路线代价对比 | **待核实** | 未取到正文/幻灯片 |

没拿到的东西：ACM DL 全文（`https://doi.org/10.1145/3701716.3715856` 返回 403 Cloudflare 拦截，**事实**：实测）；tutorial 幻灯片的可下载版本。

## 是否值得读原文

**知道结论即可**。

理由：这是一篇定位型 tutorial，摘要已经把"两条路线 + 四类方法 + 挑战与方向"讲完了；真正的方法细节都在它引用的原始论文里，而那些原始论文本 track 大多单独覆盖了（C-02/C-03/C-04/C-05）。除非你要写综述或做选题，否则读摘要 + 本表足够。若要做选题，去找它的 slides（比正文更密）。
