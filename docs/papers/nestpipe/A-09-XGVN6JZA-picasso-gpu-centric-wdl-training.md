---
title: "PICASSO: Unleashing the Potential of GPU-centric Training for Wide-and-deep Recommender Systems"
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-A, gpu-utilization, kernel-packing, hybridhash, operator-fusion]
related: [docs/papers/nestpipe/index.md]
zotero_key: XGVN6JZA
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Yuanxing Zhang、Langshi Chen、Siran Yang、Man Yuan、Huimin Yi、Jie Zhang、Jiamang Wang、Jianbo Dong、Yunlong Xu、Yue Song、Yong Li、Di Zhang、Wei Lin、Lin Qu、Bo Zheng（阿里巴巴，**推断**：从作者群、XDL 对比基线、Alibaba/EFLOPS 集群名可合理判断；正文未逐条列单位）。
- 年份与会议：2022 年，ICDE 2022（IEEE 第 38 届数据工程国际会议），页码 **3453–3466**，DOI `10.1109/ICDE53745.2022.00324`（**事实**：Zotero + CrossRef）。
- 公开全文：`https://arxiv.org/abs/2204.04903`，正文取到 `https://arxiv.org/html/2204.04903`（**事实**）。
- 生产状态：2020 年 10 月起部署在阿里内部集群，服务检索、广告竞价、推荐、搜索排序（**事实**：第 V 节）。

## 一句话

WDL 模型把 GPU 用废了，靠算子打包、交错执行和分层哈希缓存放回硬件利用率。

## 背景与问题

PICASSO 的起点是一个很具体的观测：**GPU 在推荐训练里严重闲置，拿不到 CV/NLP 那种吞吐提升**（**事实**：摘要）。

原因被拆成两条（**事实**：摘要）：

1. **输入特征域太多**——WDL 模型有**上千个输入特征域**，产生大量碎片化、内存密集的算子；
2. **特征交互子模块太多**——多个交互子模块（GRU、Transformer、DIN…）产生大量小尺寸计算 kernel。

论文做了很细的 profiling，把 WDL 负载分成三类，各自的瓶颈位置不同（**事实**，第 II-D 节，V100 集群实测）：

| 负载类型 | 代表模型 | 瓶颈 | 实测占比 |
|---|---|---|---|
| **I/O & 内存密集** | W&D | 数据传输 + 嵌入查询，且 I/O 未被重叠 | 即使优化后，暴露的 I/O 与内存访问仍占 **~20%** 墙钟 |
| **通信密集** | CAN（DIN + DLRM 衍生） | 拉取远端高维嵌入 + 高阶交叉特征的频繁参数交换 | MP 模式 **~60%**，PS 模式 **~70%** |
| **计算密集** | MMoE 变体（71 个专家） | 算术运算 | **~50%** |

关键的数据分布观测（**事实**，第 II-B 节）：类别特征 ID 的频率分布高度倾斜，**按频率降序排列后，平均 20% 的 ID 覆盖 70% 的训练数据，最高可达 99%**。这是 HybridHash 的立论基础。

对底层执行的分析更关键（**事实**，第 II-D 节）：把 WDL 各层投影到硬件资源上，会发现**每一时刻训练只被一种硬件资源卡住，其它资源在闲置**。同步 GPU 训练的资源使用是"间歇性"的（sporadic）。

以嵌入层为例，它由这些算子串成：Unique（去冗余 ID）→ Partition（分本地/远端）→ Gather（查本地表）→ Shuffle（拉远端嵌入）→ Stitch（拼接）→ SegmentReduction（按段池化）。**每个算子被一种硬件资源卡住**（Shuffle 卡在跨节点带宽上），而它们只能顺序执行。

## 核心做法

PICASSO 的三招都是围绕"让被不同资源卡住的算子重叠起来"这个目标。

### 一、混合训练策略（Hybrid Distributed Training）

按层的性质分策略（**事实**，第 III-A 节）：

| 层 | 策略 | 通信原语 |
|---|---|---|
| 嵌入层 | **MP**（参数切到所有 executor，存在分层内存子系统里） | **AllToAllv** |
| 特征交互层 + MLP | **DP**（参数复制到所有 executor） | **Allreduce** |

这是标准做法（HugeCTR、Persia、Neo 都一样）。PICASSO 的贡献不在这里。

### 二、Packing：把上千个碎算子压成几个

分两层（**事实**，第 III-B 节）：

**（1）D-Packing（数据打包）**

把不同特征域喂进**同一个算子**的类别 ID 合并成一个 packed ID tensor，一次 launch 处理完。这直接对上 GPU 的 SIMD 范式，同时把 launch 开销从"上千次"降到"几次"。

**但不能无脑全打**——论文指出一个坑：工业系统常用 hashmap 实现嵌入表，**百万级并发查询会撞 hashmap 的低层锁**，吞吐崩溃。所以打包的约束是：

- **只打包嵌入维度相同的表**（维度不同无法进同一个 packed op）；
- 即使维度相同，如果一个 packed op 的并发查询太多，还要**继续切分**。

切分依据是一个成本估计（**事实**：式 1）：用"packed op 内所有表的参数总量 × ID 频率"估算执行成本，超过均值就拆。举例：所有维度 8 的表打成一个 op，而维度 32 的表则拆成 4 个 packed op、每个装四分之一。

算子数量的效果（**事实**：Figure 7 描述）：**不打包时查 3 张表的嵌入需要 18 个算子；D-packing + K-packing 后只剩 4 个**。

**（2）K-Packing（kernel 打包）**

这里论文做了一个关键的反对判断（**事实**，第 III-B 节）：

- **反对"巨型 kernel"**：把整个嵌入层融成一个 CUDA kernel 会丧失"被不同硬件资源卡住的算子互相交错"的机会——而这正是 PICASSO 想要的。
- **反对"编译式代码生成"**（如 XLA）：它依赖静态输入输出形状来推断 kernel 尺寸，而**类别特征 ID 带来动态形状**，直接废掉编译优化的效率。

所以 K-Packing 的做法是：按硬件资源使用把 kernel 分三类（**计算密集 / 内存密集 / 通信密集**），**只在同一类内部融合**，跨类保留交错机会。

融合后的两个关键 kernel：**Unique&Partition**（消除冗余 ID 的内存访问与通信）、**Shuffle&Stitch**（把 shuffle 的输出直接 stitch，省掉显式 stitch kernel）。

### 三、Interleaving：让不同资源的算子并行

**（1）D-Interleaving（数据交错）**

针对大 batch 与 GPU 显存容量的矛盾（**事实**，第 III-C 节）。大 batch 对精度和吞吐都有利，但中间张量（feature map）占用与 batch 成正比，32GB 的 V100 很容易 OOM。

做法是**微批（micro-batch）切分 + 交错**，两个方向都可切入：

```
方向 A（从 MLP 切入，省显存）
  特征交互层输出 ──┬─→ 微批1 ─→ MLP ─┐
                   ├─→ 微批2 ─→ MLP ─┼─→ 合并
                   └─→ 微批3 ─→ MLP ─┘
  效果：MLP 的峰值显存被摊平

方向 B（从嵌入层切入，省开销）
  类别 ID ──┬─→ 微批1 ─→ 嵌入层 + 后续 ─┐
            ├─→ 微批2 ─→ 嵌入层 + 后续 ─┼─→ 合并
            └─→ 微批3 ─→ 嵌入层 + 后续 ─┘
```

微批大小由式 2 估算：用"算子的主导硬件资源上界 ÷ 单条数据在该资源上的成本"。但嵌入层算子形状动态，解析值算不出来，**只能在 warm-up 迭代里实测**。

**（2）K-Interleaving（kernel 交错）**

Packing 之后算子少了，但**不同 packed op 仍然抢同一种资源**——比如所有 Shuffle&Stitch 同时 launch，以太网带宽立刻成为瓶颈。

K-Interleaving 在 packed op 组之间**建立控制依赖**，使一个"交错组"内不会同时被多种资源卡住。组容量由式 3 估算（处理参数量为度量）。此外允许用户设置 **preset excluded embedding**：某些嵌入的输出不与其它嵌入拼接时，可以提前跑它的下游。

### 四、Caching：HybridHash

针对两级存储各自的短板（**事实**，第 III-D 节）：

- DRAM 容量大但**带宽受限**；
- GPU 显存带宽高但**容量受限**。

HybridHash 的设计选择很有意思：**把 hashmap（稀疏结构）放在 Cold-storage（DRAM）上，把 Hot-storage（GPU 显存）只当 scratchpad 用**。这跟常见的"GPU hashmap"方案相反——论文的理由是**GPU 显存是昂贵资源，不能浪费**（**事实**：原文 "we consider Hot-storage as an expensive resource, which shall avoid any waste of its capacity"）。

```
warm-up 若干迭代：统计每个 ID 的查询频率
        │
        ▼
每 flush_iters 次迭代：把 top-K 高频嵌入从 DRAM 拷进 GPU 显存
        │
        ▼
稳态：查询先查 Hot-storage
        ├─ 命中 → 直接返回（多数查询走这条）
        └─ 未命中 → 走 Cold-storage（DRAM hashmap）
        │
        ▼
更新：Hot-storage 里的条目被更新；未命中的冷条目在 DRAM 更新
```

K 由 Hot-storage 容量决定。有一个自适应行为：如果 warm-up 发现 GPU 显存**远大于**全部嵌入表，就全放 GPU 显存。也可扩展到 Intel 持久内存 / SSD 做多级。

## 关键数字

| 指标 | 数值 | 来源 |
|---|---|---|
| 硬件利用率提升 | **一个数量级**（"by an order of magnitude"） | **事实**，摘要 |
| 端到端吞吐提升 | **最高 6×**（多种工业推荐模型） | **事实**，摘要 |
| 生产环境收益 | 同样硬件预算下，每日训练任务墙钟**平均缩短 7 小时** | **事实**，摘要 |
| 生产对比（vs XDL） | 平均任务墙钟 **8.6h → 1.4h**；GPU SM 利用率 **15% → 75%**；带宽 **1.412 Gbps → 6.851 Gbps** | **事实**，Table IX |
| 算子压缩 | 3 张表 lookup：**18 个算子 → 4 个** | **事实**，Figure 7 |
| 数据倾斜 | **20% 的 ID 覆盖平均 70%、最高 99%** 的训练数据 | **事实**，第 II-B 节 |
| 三类负载瓶颈占比 | I/O 内存 ~20%；通信 60%(MP)/70%(PS)；计算 ~50% | **事实**，第 II-D 节 |
| 评测模型参数 | Product-1 160B、Product-2 1T、Product-3 1T | **事实**，Table II |
| 测试集群 | Gn6e（8×V100-32GB, 724GB DDR4, 32Gbps TCP）、EFLOPS（1×V100S-PCIe, 512GB DDR4, 100Gbps RDMA） | **事实**，Table I |

## 为什么值得读

A-09 在 Track A 里的定位是**"利用率"视角**——别的文章问"怎么扩到更多卡"，它问"现有的卡用满了没有"。

与同 track 的关系：

- **A-06（Neo）在算子层与它正面呼应**。Neo 的 hybrid kernel fusion 同样把"多次 lookup 融成一个 kernel、反向+优化器融成一个 kernel"，效果是"每迭代只启两个 kernel"；PICASSO 的 K-Packing 是同一思路，但**多了一层"只同类融合、跨类交错"的克制**。两者的分歧很值得玩味：Neo 追求最少 kernel 数，PICASSO 认为**融合过头会丧失交错机会，反而更慢**。这是本 track 里唯一一处明确的技术对立。
- **A-12（Hotline）、A-11（ScratchPipe）、A-04（Persia）都在处理"热门/冷门分离"**，PICASSO 的 HybridHash 是其中最"朴素"的实现（不做预取、不做流水线，纯靠频率统计 + 周期刷新）。对比 A-11 的 RAW 依赖分析，能看出"工业够用"与"学术最优"的差距。
- **A-15（FusedRec）与它的 K-Packing 有共同的敌人**：碎片化的嵌入传输导致的带宽浪费。FusedRec 说"去重让消息变小，把带宽用废了"，用融合发送解决；PICASSO 说"上千个小算子抢同一种资源"，用打包+交错解决。**两者是同一个问题的两个层次（通信层 vs 算子层）**。
- **A-16（RecIS）是它的四年后继**。RecIS 的"600+ 特征变换算子融成 3 个"与 PICASSO 的"D-Packing + K-Packing"是同一件事的下一代实现（并且 RecIS 在 PyTorch 上做，PICASSO 在 TensorFlow 上做）。
- **A-05（Hierarchical Training）与它同年（2021 KDD vs 2022 ICDE）**，代表 Meta 与阿里对同一问题的两条答案：Meta 去改并行结构，阿里去改算子层。PICASSO 能拿 6× 说明**在很多场景下先别急着加卡，先把卡用满更划算**。

最值得记的是那句"每一时刻训练只被一种硬件资源卡住"——这是一个**可迁移的诊断方法**：把负载投影到硬件资源轴上，看闲置的是哪些。这个分析框架比 PICASSO 的任何单个优化都更有价值。

## 局限

- **强绑定 TensorFlow**。PICASSO 实现在 TF 上（**事实**：第 V 节 "We implement PICASSO on top of Tensorflow"），所有优化都依赖 TF 的图结构（K-Packing 的"按资源分类融合"需要图信息）。迁移到 PyTorch 的 eager 模式后，这套分析是否还成立、收益还剩多少，论文没讨论（**推断**）。
- **6× 的基线不明**。摘要说"相对 SOTA 基线"，但具体是哪几个（TF-PS？PyTorch 手调？）需要看第 IV 节，而 8.6h→1.4h 那组是相对 **XDL**——XDL 是阿里自家的老一代框架，这个对比同样宽松（**推断**）。
- **"一个数量级"的表述含糊**。"increases the hardware utilization by an order of magnitude" 指的是 SM 利用率从 15% → 75%（5×，勉强算一个数量级）还是别的指标，摘要没说清（**事实**：这句话本身含糊）。
- **HybridHash 的统计有 warm-up 代价**。前若干迭代要做频率统计，这段时间的缓存是"瞎的"。论文说统计在 warm-up 里做，但没给 warm-up 占总训练时长的比例（**待核实**）。
- **D-Packing 的切分依赖 warm-up 实测**。式 1 需要 ID 频率，式 2/3 需要算子成本，两者都"从 warm-up 迭代获得"。这意味着 PICASSO 在**短训练任务**上可能得不偿失（warm-up 都占大头）。论文没讨论这个边界（**推断**）。
- **生产数据是内部的**。Product-1/2/3 三款模型只给了参数量，没有结构描述，也没给与公开数据集（Criteo/Alibaba）的对应关系（**事实**：Table II 只列统计量）。
- **没有线上业务指标**。全文只在系统层（吞吐、墙钟）论证，没有 AUC/CTR 的线上结果。"7 小时"这个数字是墙钟，不是业务收益（**推断**）。

## 证据与待核实

- 证据等级：**fulltext**。正文取自 `https://arxiv.org/html/2204.04903`（完整 IEEE 版 HTML），章节完整（I Introduction / II Implication from Analyzing Workloads / III System Design / IV Experiment / V Deployment / VI Related / VII Conclusion），含 Algorithm 1（HybridHash）、Table I / II / IX、式 1–3。
- **事实**：15 位作者、ICDE 2022 与页码 3453–3466、三类负载的瓶颈占比与实测数字、20% ID 覆盖 70%–99% 数据、D-Packing 的 hashmap 锁约束与维度约束、K-Packing 反对巨型 kernel 与编译式生成的论证、Unique&Partition 与 Shuffle&Stitch 两个融合 kernel、D-Interleaving 的两个切入方向、K-Interleaving 的控制依赖与 preset excluded embedding、HybridHash 的冷热分工与周期刷新、18→4 算子、6× / 数量级 / 7 小时、Table IX 的生产对比（8.6h→1.4h、15%→75%、1.412→6.851 Gbps）、2020 年 10 月起在阿里内部集群部署。
- **待核实**：
  1. 作者单位（阿里是推断，正文未见逐条列出）；
  2. 式 1/2/3 的完整表达式与其中的系数；
  3. Algorithm 1（HybridHash）第 9–26 行的具体伪码（HTML 中行号保留但内容被剥离）；
  4. 6× 的具体基线（第 IV-B 节的基准系统列表在可读文本中只剩开头）；
  5. warm-up 迭代数、flush_iters 的默认取值；
  6. Product-1/2/3 的模型结构；
  7. "an order of magnitude" 具体对应哪个指标。
- Zotero 内无 PDF 附件（集合内 50 条均无，属预期）。

## 是否值得读原文

**必读**——不是因为它的优化最先进（四年后大部分已被 RecIS 之类取代），而是因为**第 II 节的负载分析是全 track 最好的一份问题诊断**：三类负载、资源轴投影、"每时刻只被一种资源卡住"这个洞察，可以直接搬到任何 WDL 类模型的性能分析上。第 III-B 节关于"为什么不该融合成巨型 kernel"的论证也值得精读——它是本 track 里唯一一篇反方向的技术判断。
