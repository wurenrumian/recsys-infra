---
title: "EVStore: Storage and Caching Capabilities for Scaling Embedding Tables in Deep Recommendation Systems"
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-B, embedding-lookup, caching, storage-hierarchy, inference]
related: [docs/papers/nestpipe/index.md]
zotero_key: KP82AUTM
zotero_collection: 9I82QPTL
evidence: abstract-only
---

## 出处

- 作者：Daniar H. Kurniawan、Ruipu Wang、Kahfi S. Zulkifli、Fandi A. Wiranata、John Bent、Ymir Vigfusson、Haryadi S. Gunawi（University of Chicago / 北京工业大学 / 万隆理工 / Seagate / Emory）。
- 年份与会议：2023 年，ASPLOS '23（第 28 届 ACM 体系结构支撑会议，Vol. 2），页码 381 起。
- DOI：`10.1145/3575693.3575718`。
- 代码：`https://github.com/ucare-uchicago/ev-store-dlrm`（据 ASPLOS 论文页与二手阅读笔记，**事实**）。

## 一句话

给推荐推理加三层嵌入缓存，DRAM 不够时用近似换内存，最多省 94% 内存。

## 背景与问题

推荐模型推理一次要查**多张**嵌入表，只要其中一次查询慢，整个请求就被拖慢（**事实**：摘要与二手笔记都说"multiple EV table lookups, if any memory access is slow, the whole inference request is slow"）。

开源 DLRM（如 Facebook DLRM）把完整嵌入表放在 DRAM 里，**内存耗尽就直接失败**，没有向下一层存储（SSD/远端）回落的机制。而工业模型的表规模是 TB 级，全塞 DRAM 在成本上不可行。

现有缓存思路的问题在于：它们把嵌入当普通 KV 缓存来管，没有利用两件事——
1. **结构规律**：一次请求的查询序列在推理侧是可预期的（**推断**：来自摘要 "structural regularity in inference operations"）；
2. **领域内允许近似**：推荐打分对单个嵌入向量的轻微偏差不敏感，而系统层完全可以拿这一点换内存。

## 核心做法

EVStore 是插进 DLRM 里的**三层 EV 表查询系统**，L1→L2→L3 逐层降级：

| 层 | 名字 | 存什么 | 换什么 |
|---|---|---|---|
| L1 | EVCache | 完整的 fp32 嵌入，缓存替换算法（扩展了多种替换策略） | 命中即返回，最快 |
| L2 | EVMix | **低精度**嵌入（如 fp8） | 少占内存，精度损失微小 |
| L3 | EVProx | **key→key 映射**：把一个冷门 key 代理到嵌入值相近的另一个 key | 不存自己的向量，只存指针；精度损失取决于相似度阈值 |

数据流（**事实**，据二手阅读笔记的层级描述整理）：

```
请求的 key 集合
   │
   ├─ L1 EVCache 命中 ──→ 直接返回 fp32 向量
   ├─ L1 miss，L2 命中 ──→ 返回低精度向量（解量化）
   └─ L1/L2 miss ──→ L3 EVProx：key 重映射到代理 key
                        ├─ 代理 key 在 L1/L2 → 取其向量
                        └─ 都没有 → 回落到后端存储（DRAM 之外）
```

L3 的 key 映射表是**离线预处理**建立的：用欧氏距离与余弦距离做相似度统计，为每个 key 找替代 key。论文给出的运维规则是"当 L3 命中率显著下降时重做重映射"（**事实**，二手笔记转述）。

代价结构很清楚：**L1 花内存换延迟，L2 花精度换内存，L3 花精度换内存且额外花一次查表**。三层是替代关系而非叠加——同一份内存预算要在"多放几条完整向量"和"多放几条代理指针"之间分配。

## 关键数字

- 平均延迟降低 **23%**，p90 延迟降低 **27%**，吞吐**提升到 4 倍**，精度只掉 **0.2%**（**事实**：ASPLOS 官方摘要原文 "up to 23% and 27% reduction on the average and p90 latency while quadrupling throughput at 0.2% loss in accuracy"）。
- DRS 内存占用最多降低 **94%**，代价是"minor cost of accuracy"（**事实**：同摘要末句）。
- 评测数据：Criteo 1TB Click Logs，13 个稠密特征 + 26 个稀疏类别特征（26 张 EV 表），所有表维度统一为 36，共 1560 亿特征值、超 8 亿唯一属性值（**事实**，二手阅读笔记）。

## 为什么值得读

在 NestPipe 这条"嵌入表规模爆炸后系统怎么重设计"的线上，EVStore 站的是**推理侧、存储层级**这一格：

- 与 B-09（Distributed Hierarchical GPU PS）是**镜像关系**：B-09 解决"训练时 10TB 参数放不进 GPU/内存"，用 HBM→DRAM→SSD 三层 + 4 级流水；EVStore 解决"推理时查询延迟"，用缓存三层 + 近似。两者都把嵌入当"分层存储问题"，但一个在训练侧优化吞吐，一个在推理侧优化尾延迟。
- 与 B-15（HypeReca）是**互补**：HypeReca 把分布式内存 KV 数据库当嵌入服务抽象，追求训练侧多节点扩展；EVStore 是单机内的缓存层级，追求把单次查询压下去。
- 它是 Track B 里少数**明确用"精度"换"内存"**的推理侧工作，和 B-03（CAFE+）、B-10（MPE）、B-11（DQRM）这些量化/压缩派是同一动机不同落点：那几篇压模型本身，EVStore 压"存什么"。
- 若你的系统只关心训练吞吐，这篇可以直接跳过；若涉及在线推理的 P99 或边缘部署，它是必读的一手参考。

## 局限

- **只覆盖推理**。训练侧的梯度更新、参数同步全部不在射程内（**推断**：论文通篇讲 lookup 与 inference request）。
- **依赖近似**。L3 的 key 代理本质是"用别的 key 的向量冒充"，0.2% 的精度损失在论文的 Criteo 设置里可接受，但换到对单条特征极敏感的任务（如出价、风控）未必成立；这个边界论文没展开。
- **重映射是离线批处理**，且触发条件是"命中率显著下降"——这是个滞后信号，意味着分布漂移期间会有一段性能退化窗口（**推断**）。
- 评测表维度统一为 36，工业上表维度常混杂（32/128/512），多层缓存的分层收益在不同维度分布下会不会变，**待核实**。
- 未取到正文，三层之间的内存分配比例、L3 相似度阈值怎么定，**待核实**。

## 证据与待核实

证据等级：**abstract-only**。已确认的来源：
- Zotero 元数据（作者/年份/DOI/会议）——**事实**。
- ASPLOS '23 官方摘要片段（来自会议论文集目录页与 SIGPLAN OpenTOC 页）——**事实**。
- `paper.lingyunyang.com` 的二手阅读笔记（三层结构与 Criteo 实验设置）——**事实级转述**，不是原文。

没拿到：正文 PDF（`https://ucare.cs.uchicago.edu/pdf/asplos23-EVStore.pdf` 存在但本次未解析）。待核实清单：
1. 三层各自的容量比例与替换策略的具体变体；
2. L3 相似度阈值、重映射的触发指标与周期；
3. "quadrupling throughput" 的基线是什么（原始 DLRM？还是已有缓存方案？）；
4. 94% 内存降低对应的精度损失具体数值。

## 是否值得读原文

**挑章节读**——如果你的工作涉及在线推理的尾延迟或嵌入表 offload，读三层设计与 L3 相似度构造那两节就够；训练系统方向的话，结论（近似换内存、最多 94%）已经能支撑决策，不必读全文。
