---
title: "Hierarchical Training: Scaling Deep Recommendation Models on Large CPU Clusters"
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-A, cpu-cluster, model-parallelism, data-parallelism, operator-placement]
related: [docs/papers/nestpipe/index.md]
zotero_key: HP588YPM
zotero_collection: 9I82QPTL
evidence: abstract-only
---

## 出处

- 作者：Yuzhen Huang、Xiaohan Wei、Xing Wang、Jiyan Yang、Bor-Yiing Su、Shivam Bharuka、Dhruv Choudhary、Zewei Jiang、Hai Zheng、Jack Langman（Meta / Facebook，**事实**：Meta Research 页面列全部作者且标注单位 Facebook, Menlo Park）。
- 年份与会议：2021 年，KDD '21（第 27 届 ACM SIGKDD），页码 **3050–3058**，DOI `10.1145/3447548.3467084`（**事实**：Zotero 元数据 + CrossRef 确认页码与 container）。
- 可访问页面：Meta Research 出版页 `https://research.facebook.com/publications/hierarchical-training-scaling-deep-recommendation-models-on-large-cpu-clusters`（摘要全文可取，**事实**）；ACM DL 有"Free access"标记但 PDF 直链返回 403（本次未取到正文）。

## 一句话

CPU 集群上把神经网络部分按算子切开、组内既做数据并行又做模型并行，提速 1.9× 不掉点。

## 背景与问题

2019–2021 年 Meta 的推荐模型在 CPU 集群上训（不是 GPU——这是理解本篇的关键前提）。模型越来越复杂、训练数据越来越多，需要**在集群上横向扩**；但一扩就掉点，或者根本扩不动。

作者说"提升可扩展性而不牺牲模型质量是有挑战的"（**事实**：摘要原话 "improving the scalability without sacrificing the model quality is challenging"），并且明确做了"对大规模 CPU 集群上现有训练架构的可扩展性瓶颈的深入分析"（**事实**）。

**具体瓶颈是什么，摘要没有展开**（**待核实**）。但从论文的解法可以反推问题形状（**推断**）：

- 如果纯数据并行：每个 worker 都要存一份完整的稠密参数副本，模型一大，副本内存 × worker 数就成了墙；而且每步全量 AllReduce 的通信量随副本数线性涨。
- 如果纯模型并行：每个 worker 上的算子变小，算力利用率掉下来，且层间激活要跨网络传。
- 嵌入层通常另有专门的并行方案（按表切），所以**真正的麻烦在"神经网络部分"**——这正好对应摘要里那句"exploits both data parallelism and model parallelism **for the neural network part of the model within a group**"。

## 核心做法

Hierarchical Training（下文简称 HT）的核心是把**并行维度做成两层嵌套**，而不是在全局只选一种。

### 一、拓扑：两层并行

```
                全局：N 个 worker（CPU 节点）
   ┌──────────┬──────────┬──────────┬──────────┐
   │  Group 0 │  Group 1 │  Group 2 │   …      │   组间：数据并行
   │  (g 个)  │  (g 个)  │  (g 个)  │          │   （各组吃不同数据分片）
   └────┬─────┴────┬─────┴────┬─────┴──────────┘
        │          │          │
   ┌────▼────┐┌────▼────┐┌────▼────┐
   │组内：    ││组内：    ││组内：    │            组内：数据并行 + 模型并行
   │DP + MP   ││DP + MP   ││DP + MP   │             （稠密参数既切又复制）
   └──────────┘└──────────┘└──────────┘
```

摘要的原话是"within a group"同时用 DP 和 MP。**组内怎么同时用这两种**，摘要没说清（**待核实**）。一种自洽的读法（**推断**）是：把神经网络切成若干层/算子块，相邻块放到不同 worker 上（MP，解决单副本内存），同时组内多个 worker 冗余持有同一块的副本做数据并行（DP，解决吞吐）；这样每组能独立完成一次完整前反向，组间再用数据并行合并。

### 二、实现：两层设计

论文自述的实现是一个"two-layer design"（**事实**：摘要明确）：

| 层 | 名字 | 职责 |
|---|---|---|
| 上层 | **Tagging System**（标记系统） | 决定**算子放哪儿**——即每个算子该被切、被复制，还是两者兼有 |
| 下层 | **Net Transformation System**（网络变换系统） | 把训练计划**物化**（materialize）：按上面的标记真正改写计算图，插入通信、拆分/复制参数 |

这个分层很有意思：**决策与执行分离**。tagging 只产出"位置决策"这种声明式的东西，net transformation 负责把它编译成可执行的图。这样同一套 tagging 结果可以落到不同形态的训练栈上——摘要说"integrate hierarchical training into existing training stack"（**事实**）。

**tagging 怎么决定位置，摘要没有给算法**（**待核实**）。一份第三方摘要站点（`oplclaw.com`）称是"规则式标记 + 迭代贪心精化"，但这是二手，**不作为事实**。

### 三、三个附加优化

摘要列了三类（**事实**：都只到名字级）：

1. **模型架构优化**（model architecture optimization）；
2. **通信压缩**（communication compression）；
3. **各类系统级改进**（system-level improvements）。

三者各自做了什么、贡献多少，**全部待核实**。通信压缩这一项在 HT 里应该是必需的——组间数据并行的 AllReduce 需要压，否则通信吃掉并行收益。

## 关键数字

| 指标 | 数值 | 来源 |
|---|---|---|
| 端到端加速 | **1.9×**（"without model quality drop"） | **事实**：摘要原句 |
| 集群规模 | "massive scale"、"large CPU clusters"——**具体节点数摘要未给** | **待核实** |
| 组规模 | 二手来源称 gang size 100–200 节点 | **待核实**（第三方摘要站，非原文） |
| 单节点核心数 | 未取到 | **待核实** |
| 模型参数量 | 未取到 | **待核实** |

注意：**1.9× 是本 track 里最"温和"的一个加速比**（对比 A-04 的 7.12×、A-06 的 40×、A-10 的 5.6×）。这不是因为 HT 弱，而是因为它的基线是"已经优化过的 CPU 训练栈"，不是"没优化的原始系统"。做横向对比时这个口径差异必须记住（**推断**）。

## 为什么值得读

在 Track A 的 17 篇里，A-05 是**唯一一篇基线完全在 CPU 上**的文章，这个位置不可替代。

它的价值有三层：

**1. 它是"并行维度嵌套"思路的早期形态。** A-03（2D Sparse Parallelism）用"组内 MP + 组间 DP"处理嵌入表，A-06（Neo）用 4D 并行处理嵌入算子——两者都是"在多个维度上同时切"。HT 在 2021 年就已经在稠密网络上做了同样的事，只是维度只有两维、且在内层是"DP 和 MP 混用"而非"分组"。**A-03 / A-06 可以看作 HT 思路在嵌入表上的重演与复杂化。**

**2. 它是 Persia（A-04）的直接前驱。** 两篇同一个作者群的核心成员（Yujian Yang、Bor-Yiing Su 等），同一个机构，同一条演化线：HT 在纯 CPU 集群上做混合并行 → Persia 把同样的"混合"抬升到算法与硬件两层（嵌入 CPU 异步 + 稠密 GPU 同步）→ 规模从"大 CPU 集群"到"100T 参数"。**看 A-05 → A-04 能看出一条完整的四年演进。**

**3. 它给出了"决策与执行分离"的工程范式。** tagging（声明位置） + net transformation（物化计划）这个两层结构，在后面的系统里反复出现：TorchRec（A-08）的 sharder + planner 是同一个模式的另一份实现（planner 自动生成分片方案，sharder 执行）。**如果你要设计自己的训练系统，这个"先声明后编译"的结构比任何单个优化都值得抄。**

与同 track 的关系梳理：

| 篇目 | 关系 |
|---|---|
| A-04 Persia | **演进后继**：CPU 集群混合并行 → CPU+GPU 异构混合训练 |
| A-06 Neo | **同期平行**：Meta 的另一条路，全 GPU + 4D 并行 + 专用硬件 |
| A-03 2D Sparse | **思路同源**：组内 MP + 组间 DP，但对象是嵌入表 |
| A-08 TorchRec | **模式同构**：sharder + planner 与 tagging + net transformation 结构相同 |
| A-16 RecIS | **反例**：RecIS 明确说"CPU 异构架构已经过时"，HT 正是那个架构的代表 |

## 局限

- **基线是 CPU，结论不能外推到 GPU**。1.9× 的加速在有 GPU 的场景下不成立——把稠密网络搬到 GPU 的收益远超 1.9×（**推断**）。这篇的结论只在"必须/只能跑 CPU"的约束下有效。
- **只有摘要可读，机制细节全空**。tagging 的决策算法、net transformation 的具体改写规则、三个优化的具体内容，全部**待核实**。这意味着这篇的"可复用价值"目前只能停在"知道有这么个结构"，不能停在"会实现"。
- **加速比 1.9× 在 2021 年也不算亮眼**，尤其对比同期 Neo 的 40×。合理的解释是基线不同（HT 的基线已经很优化），但这需要正文来证实，摘要里看不出来（**推断**）。
- **纯 CPU 集群这个前提在 2026 年基本消失**。连 RecIS（2025）都说"把嵌入留在 CPU 的异构架构已经过时"（**事实**：RecIS 2.2.2）。所以这篇的现实价值主要是**历史脉络 + 设计模式**，而不是可落地的方案（**推断**）。
- **没有线上质量指标**。只说 "without model quality drop"，但没给具体的 NE/AUC 差异数值（**推断**：摘要层面未提）。
- 与 A-04 相比，**没有开源**。Persia 有 GitHub，HT 没有——想复现只能靠正文。

## 证据与待核实

- 证据等级：**abstract-only**。已确认来源：
  - Zotero 元数据（10 位作者、2021-08-14、DOI、Publisher ACM）——**事实**；
  - CrossRef（container = KDD '21、页码范围 3050–3058、年份 2021）——**事实**；
  - Meta Research 官方出版页的完整摘要——**事实**（本次直接抓取）；
  - ACM DL 页面标记 "Free access" 但 PDF 直链返回 403（Cloudflare 挑战页），本次未能取到正文——**事实**（技术限制）。
- **事实**：两层并行结构（within a group 同时 DP+MP）、two-layer 实现（tagging system + net transformation system）、三类附加优化（模型架构优化 / 通信压缩 / 系统级改进）、1.9× 加速且不掉点、10 位作者、KDD '21、页码 3050–3058。
- **待核实**：
  1. 具体瓶颈分析（"in-depth analysis of scalability bottleneck" 的结论是什么）；
  2. tagging system 的决策算法（规则式？贪心？有成本模型吗？）；
  3. net transformation system 如何改写计算图、插入哪些通信原语；
  4. 组内 DP 与 MP 的具体配比与切分粒度；
  5. 实验集群规模、模型参数量、基线系统名称；
  6. 三类优化各自的贡献分解（ablation）；
  7. 1.9× 的度量对象（吞吐？到固定质量的端到端时间？）；
  8. 二手来源提到的"gang size 100–200 节点"与"规则式标记 + 迭代贪心精化"是否属实。
- 未尝试绕过 ACM 的访问限制（合规考虑），也未向仓库下载 PDF。

## 是否值得读原文

**挑章节读**——如果你在做"多维度并行 + 算子放置"的系统设计，tagging system 与 net transformation system 这两节值得精读，它们是"声明式并行规划"最早的一手描述之一；如果只是想知道 CPU 集群这条路怎么走，摘要的结论（1.9×、组内 DP+MP、两层架构）已经够用，而且这条路在 2026 年基本没有落地价值，不必花时间读全文。
