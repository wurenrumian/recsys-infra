---
title: "Persia: An Open, Hybrid System Scaling Deep Learning-based Recommenders up to 100 Trillion Parameters"
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-A, hybrid-training, parameter-server, asynchronous-sgd, open-source]
related: [docs/papers/nestpipe/index.md]
zotero_key: 892HASSZ
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Xiangru Lian、Binhang Yuan、Xuefeng Zhu、Yulong Wang、Yongjun He、Honghuan Wu、Lei Sun、Haodong Lyu、Chengjun Liu、Xing Dong、Yiqiao Liao、Mingnan Luo、Congfei Zhang、Jingru Xie、Haonan Li、Lei Chen、Renjie Huang、Jianying Lin、Chengchun Shu、Xuezhong Qiu、Zhishan Liu、Dongying Kong、Lei Yuan、Hai Yu、Sen Yang、Ce Zhang、Ji Liu（快手 + ETH Zurich 等，共 27 人）。
- 年份与会议：2022 年，KDD '22（第 28 届 ACM SIGKDD），DOI `10.1145/3534678.3539070`（**事实**：Zotero 元数据 + CrossRef 确认 container 为 KDD '22）。
- 公开全文：`https://arxiv.org/abs/2111.05897`，正文取到 `https://arxiv.org/html/2111.05897`（**事实**）。
- 开源：`https://github.com/PersiaML/Persia`（**事实**：摘要与结论均明示）。

## 一句话

嵌入层走 CPU 参数服务器异步更新、稠密层走 GPU AllReduce 同步更新，两套机制拼出百 T 参数。

## 背景与问题

2021 年那条"模型规模每跳一档质量就跳一档"的曲线（Google 2016 的 1B → Facebook 2022 的 12T）让作者判断"100 万亿参数的时代就在眼前"（**事实**：引言）。真要做，卡在一个结构性的异质问题上：

**嵌入层占模型 99.99% 以上参数，极其吃内存；稠密神经网络部分越做越深，极其吃算力。**（**事实**：摘要原话 "the model's embedding layer could include more than 99.99% of the total model size, which is extremely memory-intensive; while the rest neural network is increasingly computation-intensive"）

用统一的方式对待这两半，两边都不讨好：

- **全同步**：五步（准备嵌入 → 前向 → 反向 → 同步参数 → 更新嵌入）必须串行，硬件利用率低，扩不上去。
- **全异步**：硬件效率最好，但稠密网络的更新重叠严重，staleness 直接把收敛搞坏。

作者还列了现有系统的具体不足：直接套通用 PS（如 TensorFlow 的）会把嵌入和 NN 参数都放 CPU PS、前向反向都放 GPU worker——"far from efficient and even impossible for deployment"，因为模型装不进 GPU RAM，而且把嵌入和 NN 当同一类东西会产生大量无谓网络流量（**事实**：4.1 节）。XDL 把嵌入的前反向也放到 model server 上、Baidu AIBox 用协同定位 PS + 缓存策略，都是对的方向但规模不够（**事实**：4.1 节的文献综述）。

## 核心做法

Persia 的核心是一个判断：**训练算法的同步语义应该按层的性质分别设计**，然后系统结构照着这个分裂去搭。

### 一、混合训练算法

分界线就是"稀疏 vs 稠密"：

| | 嵌入层 | 稠密 NN |
|---|---|---|
| 参数占比 | >99.99% | <0.01% |
| 访问模式 | 每步只碰极少数行（稀疏） | 每步碰全部参数（稠密） |
| 更新重叠 | 几乎不重叠 ⇒ staleness 无害 | 高度重叠 ⇒ staleness 有害 |
| 同步方式 | **异步**（PS 语义） | **同步**（AllReduce 语义） |

这个分法的理论依据是作者引的两条 ML 社区共识（**事实**，3.1 节）：

- 异步更新在**稀疏访问**下高效：单步只改一小部分参数，覆盖写很少，即使发生也几乎不引入偏差。
- staleness **限制**异步 SGD 的可扩展性与收敛：更新重叠严重（稠密网络）时，偏差会拖垮收敛。

结论：**异步给嵌入，同步给 NN**。作者说这个混合算法"converges as fast as vanilla SGD"（**事实**：结论），并在第 5 节给了 Assumption 1 / Theorem 1 的理论支撑（**具体定理形式待核实**，正文公式被剥离）。

### 二、系统结构：四个可独立伸缩的模块

```
                ┌──────────────┐
   Hadoop/Kafka │  Data Loader │
                └──────┬───────┘
                       │ ① 只发 ID 类特征
                       ▼
        ┌──────────────────────────────┐
        │  Embedding Workers（CPU 节点）│  ② 生成唯一 sample ID，本地缓存 ID 特征
        └──────────────┬───────────────┘
                       │ ③ 把 sample ID 回给 data loader
                       ▼
              sample ID 关联 Non-ID 特征与 label
                       │ ④ 只发 Non-ID 特征 + label
                       ▼
        ┌──────────────────────────────┐
        │  NN Workers（GPU 节点）       │  ⑤ 前向/反向，AllReduce 同步
        └──────────────┬───────────────┘
                       │ ⑥ 嵌入梯度
                       ▼
              Embedding PS（CPU 节点，异步更新）
```

拆解这个流程的关键设计（**事实**，4.1 节）：

- **数据在入口就被切成两路**。ID 类特征发嵌入 worker，Non-ID 特征 + label 发 NN worker。中间用"唯一 sample ID"（实现上就是拿第一个字节编码嵌入 worker 的 rank）把两半拼回去。
- **嵌入 worker 跑在 CPU 节点上，与嵌入 PS 之间是 PS 范式**（pull 参数、push 梯度）。
- **NN worker 跑在 GPU 节点上，之间是 AllReduce 范式**。
- 四个模块（data loader / embedding PS / embedding workers / NN workers）**各自可动态伸缩**，这是"支持 100T"的前提——不同规模只需要调不同模块的副本数。

这个设计解决了一个具体问题：如果嵌入和 NN 共用一套通信原语，那么"嵌入的稀疏 pull"和"NN 的稠密 AllReduce"会在同一张网上抢带宽，而且两者对同步语义的要求正好相反。

### 三、为填平异步/同步缝隙做的工程

第 4.2 节列了一串（**事实**：小节标题齐全，细节部分需查 PDF）：

- **Fill the Async/Sync Gap**：异步嵌入和同步 NN 在同一个 step 里的时序对不齐，需要显式补偿。
- **Persia Memory Management**：为百 T 规模的嵌入参数做内存管理。
- **Communication Optimization**：通信优化。
- **Fault Tolerance**：容错——百 T 规模下 checkpoint 本身是分钟级的，不容错没法跑。

## 关键数字

| 指标 | 数值 | 来源 |
|---|---|---|
| 最大规模 | **100 万亿（100T）参数** | **事实**，摘要 + 结论，且在 Google Cloud 上跑通 |
| 相对 XDL 加速 | **7.12×**（结论段与 Figure 6 描述一致） | **事实** |
| Taobao-Ad 上相对 XDL 同步 / 异步 | **7.12× / 8.4×** | **事实**，6.1 节 |
| Taobao-Ad 上相对 PaddlePaddle | **1.71×** | **事实**，6.1 节 |
| 生产集 Kwai-Video | 嵌入 2T 参数 / 稠密 34M 参数，30 亿条记录 | **事实**，Table 1 |
| 合成扩容序列 | Criteo-Syn 1–5：6.25T / 12.5T / 25T / 50T / 100T | **事实**，Table 1 |
| 基线系统跑不动的证据 | XDL 在 Kwai-Video 上**跑若干小时仍未收敛** | **事实**，6.1 节 |
| 异步 vs 混合的上界 | 100T 时异步模式比混合再快 **1.2×**，但收敛性没保障 | **事实**，6.3 节（数值为 1.2，来自公式 alttext） |
| 集群配置 | 8×8 GPU workers | **事实**，6.1 节 |

## 为什么值得读

Persia 在 Track A 里是**"用异构硬件 + 异构同步语义"这一派的代表作**，而且它有一个别人没有的属性：**开源、可复现**（GitHub 上有 GCP 部署说明）。Track A 的 17 篇里，真正把百 T 规模跑通并放出代码的，只有这一篇。

与同 track 各篇的关系：

- **A-06（Neo / SW-HW co-design）是它的同期对手**。两边都冲 12T→更大规模，但路线相反：Persia 用 CPU 扛嵌入、GPU 扛稠密、两套同步机制；Neo 全部塞进 GPU，靠 4D 并行 + ZionEX 专用网络。Neo 做到 12T / 1.7M QPS / 40× 加速，Persia 做到 100T / 7.12×。**Neo 胜在单位算力的吞吐，Persia 胜在规模上限和开放性。**
- **A-05（Hierarchical Training）是它的前身/同门**。A-05 在纯 CPU 集群上做"组内同时用数据并行和模型并行"的两层架构，提速 1.9×；Persia 把同一思路搬到 CPU+GPU 异构上，并加上算法层的混合同步。看这两篇能看出 Meta 与快手/KDD 两条独立演化路径的收敛。
- **A-03（2D Sparse Parallelism）是相反方向**。A-03 全程 GPU、不走 CPU PS，靠"组内 MP + 组间 DP"压缩通信；Persia 恰恰把嵌入推进 CPU。两者的分歧本质是"嵌入该不该离开 GPU"——A-11（ScratchPipe）、A-09（PICASSO）、A-16（RecIS）在往"全部上 GPU"走，Persia 则是"CPU 仍然是嵌入的家"。这条分歧是 Track A 最值得跟的一条线。
- **A-12（Hotline）是它的直接改进者**：Hotline 批评混合 CPU-GPU 模式的 CPU↔GPU 传输时间，用"热门嵌入上 GPU、冷门留 CPU"把传输压掉，正好补 Persia 这条路线最大的短板。
- 与 **A-17（LLM-Based RecSys 并行优化）互补**：A-17 处理的是"LLM 化之后稠密部分变成主瓶颈"的场景，而 Persia 的分界线（稀疏异步 / 稠密同步）在那种模型上会让 NN 那一侧变得极重，A-17 是它的后继问题。

## 局限

- **异步嵌入的收敛保证是"实践上足够"，不是"理论上等价"**。Theorem 1 给的是在 Assumption 1 下的收敛性，但 Assumption 1 的具体内容（有界 staleness？有界梯度？）在可读文本中被剥离。工业界最关心的"异步到底会不会掉点、掉多少"这篇没有给系统的消融（**待核实**）。
- **CPU 那一半的算力被浪费了**。嵌入 lookup 本身是内存带宽型操作，放 CPU 上意味着 CPU 的 DRAM 带宽成为硬瓶颈——这正是 A-12（Hotline）和 A-11（ScratchPipe）共同攻击的点。Persia 的做法是把"内存容量"最大化，代价是"内存带宽"最小化（**推断**）。
- **同质集群假设**。所有嵌入 worker 同构、所有 NN worker 同构，靠数量伸缩。真实集群里机器代次混杂，这个假设很贵（**推断**）。
- **规模数字的对照口径可疑**。100T 是在合成的 Criteo-Syn 上做的（随机 ID 特征），不是真实模型；真实生产集只到 2T（Kwai-Video）。"能扩到 100T"和"100T 有用"是两件事，论文混着说（**推断**：Table 1 显示 100T 那一档是 Criteo-Syn 5、稠密参数只有 12M）。
- **2022 年的系统，硬件假设已经过时**。当时 A100 40/80GB 是顶配，现在单机 8×H100/H200 的 HBM 容量已经从"装不下"变成"能装下不少"，"嵌入必须上 CPU"的前提正在被削弱——A-16（RecIS）明确说了这一点（"this architecture is now outdated given current hardware trends"）（**事实**：RecIS 2.2.2 节）。
- 论文没讨论**推理侧**。训练到 100T 之后怎么服务，完全在射程外（**推断**）。

## 证据与待核实

- 证据等级：**fulltext**。正文取自 `https://arxiv.org/html/2111.05897`（HTML 版，完整 8 节 + 附录），含 Algorithm 1、Assumption 1、Theorem 1、Table 1、Figure 3–9 的文字描述。
- **事实**：27 位作者、KDD '22 与 DOI、开源仓库地址、嵌入层占比 >99.99%、两条混合算法的理论依据（异步适稀疏 / staleness 害稠密）、四模块结构与数据分发流程的六步、PS 与 AllReduce 的分工、7.12× / 8.4× / 1.71× 三组加速比、100T 在 GCP 跑通、Table 1 的模型规模、XDL 在 Kwai-Video 上跑不收敛、异步比混合再快 1.2×。
- **待核实**：
  1. Assumption 1 与 Theorem 1 的完整陈述与证明条件（HTML 中公式被剥离，只剩 "Assumption 1." / "Remark 1." / "Theorem 1." 的编号）；
  2. Algorithm 1（嵌入 worker 的异步更新伪码）的具体步骤；
  3. 4.2 节四个工程优化的具体机制（只取到小节标题）；
  4. 7.12× 的度量口径（到固定 AUC 的端到端时间，还是固定迭代数的吞吐）；
  5. Theorem 1 是否要求有界 staleness，以及实测 staleness 分布。
- Zotero 内无 PDF 附件（集合内 50 条均无，属预期）。开源代码存在但本次未阅读，实现细节未核对。

## 是否值得读原文

**必读**——这是 Track A 里唯一"百 T 规模 + 开源 + 有理论"三条都占的一篇，做超大嵌入系统选型绕不过去。重点读第 3 节（混合算法的推导与两半的分界线在哪）和第 4 节（四模块的数据流与 async/sync 缝隙怎么填）；第 5 节的理论如果只需要"能收敛"这个结论可以跳过，但如果要在自己的场景里改异步窗口大小，必须读 Assumption 1 的约束。第 6 节的 Table 1 值得单独看一眼——它会让你重新审视"100T"这个标题。
