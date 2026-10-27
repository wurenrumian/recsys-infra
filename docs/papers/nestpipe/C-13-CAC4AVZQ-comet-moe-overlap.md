---
title: Comet - Fine-grained Computation-communication Overlapping for Mixture-of-Experts
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-C, MoE, communication-overlap, thread-block-specialization, GroupGEMM, shared-tensor]
related: [docs/papers/nestpipe/index.md]
zotero_key: CAC4AVZQ
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Shulai Zhang、Ningxin Zheng、Haibin Lin、Ziheng Jiang、Wenlei Bao、Chengquan Jiang、Qi Hou、Weihao Cui、Size Zheng、Li-Wen Chang、Quan Chen、Xin Liu（**事实**，Zotero + 正文）。
- 单位：上海交通大学 + ByteDance（**推断**：作者组合与 "adopted in the production environment of clusters with ten-thousand-scale of GPUs" 的表述）。
- 年份：2025（arXiv v1 2025-02-27）。
- 会议：arXiv preprint，`https://arxiv.org/abs/2502.19811`（**事实**）。
- 取到的正文：`https://arxiv.org/html/2502.19811v1`（**事实**）。

## 一句话

MoE 的通信和计算粒度对不上，别在 kernel 外做流水，塞进一个 kernel 里按线程块分。

## 背景与问题

### MoE 的通信账

MoE 用稀疏激活把容量撑到万亿级而不增加单 token 计算（**事实**）。代价是**专家必须摊到不同 GPU**（单卡装不下全部专家），于是每个 MoE 层都要在设备间大量搬 token。量化一下（**事实**，§1）：用 Megatron-LM 在 8×H100 上跑主流 MoE 模型，**设备间通信平均占整个模型执行时间的 47%**——"接近一半的时间在搬数据"。

### 现有解法为什么不够

主流做法是**流水化**：把输入数据切成块，让通信块 2 与计算块 1 重叠（**事实**，Fig.1(b)）。这个粗粒度重叠确实比不重叠好，但有两个结构性低效（**事实**，§1）。**低效 1：切块降低了专家计算效率**——切得越小，每个专家分到的数据块越小，GEMM 的 tile 利用率越低；设不切时专家计算时间是 `t`，切成 2 块后是 `t₁ + t₂`，则 `t₁ + t₂ > t`，即**重叠的收益被算力损失吃掉一部分**。**低效 2：首尾无法重叠**——收到 chunk 1 的数据、发走 chunk 2 的结果，这两段是**必然暴露**的（没有可重叠的计算），所以粗粒度重叠有不可避免的"头尾空洞"。

还有一个更根本的问题（**事实**，§1）：**MoE 的负载是动态的**。gate 在运行时决定路由，各专家输入形状每步都不同 → 各 GPU 的通信量与计算量都在变。而现有实现把通信与计算**封成不同 kernel、跑在不同 stream 上**，导致无法精确控制硬件资源、kernel 性能不确定、**重叠对不齐**（chunk 1 的计算与 chunk 2 的接收错位）。

### 粒度错配：全篇的核心观察

这是最本质的一层（**事实**，§2.2.1）：**通信的最小单位是 token**（一个 token 是一个要搬的数据项），**计算的最小单位是 tile**（如 128×128 的 GEMM tile）。一个 tile 需要哪些 token 是**运行时由 gate 决定的**，而这些 token **随机散落在多个设备上**。后果：**一个计算 tile 必须等它需要的所有 token 都到齐才能开始**；如果通信是粗粒度的（按 chunk），tile 的等待时间就被这个"凑齐"过程拉长。

天真的解法是"让通信也细到 token 级"，但（**事实**，§2.2.1）：

> "Remote I/O operations between GPUs exhibit significantly higher latency compared to local GPU memory access. Therefore, executing numerous fine-grained read and write operations on remote data tokens **within computation thread blocks** can block subsequent computational tasks."

也就是说：**细粒度 I/O 会阻塞计算线程块**，在 Hopper 上尤其明显。

**两条洞察**（**事实**，§2.2）：

> 洞察 1：解决"计算与通信的粒度错配"，是实现高效重叠的关键。
> 洞察 2：资源分配必须**在 kernel 内运行时自适应**，才能做到无缝重叠。

## 核心做法

### 先看清结构：两个生产者-消费者流水

Comet 的分析起点是把 MoE 层拆成两条流水（**事实**，§3）：**layer0** 是通信-计算流水——`dispatch`（跨 GPU 传 token，producer）→ GroupGEMM（专家第一层 FFN，consumer）；**layer1** 是计算-通信流水——GroupGEMM（专家第二层 FFN，producer）→ `undispatch` + topk 归约（combine，consumer）。两条流水的 producer 与 consumer 之间都隔着同一个共享 buffer，即 **shared tensor**，其全局尺寸为 `(M × topk, N)`（M 是 token 数，**事实**，Fig.4）。它既是 producer 的输出 buffer 也是 consumer 的输入 buffer——**重叠只能在这里发生**。

### 机制 1：shared tensor 的分解与重排

**能不能重叠，取决于 producer 与 consumer 是否在处理 shared tensor 中相互独立的数据**（**事实**，§4.1），于是关键是找"沿哪个维度切，consumer 仍然能独立处理"。两条流水方向相反，切的维度也不同（**事实**）：**layer0**（通信→计算，consumer 是 GroupGEMM）沿 **M（token）** 维切，因为 token 之间在 M 维上独立；**layer1**（计算→通信，consumer 是 top-K 归约）**只能沿 N** 维，因为归约沿 M 维做、token 之间在 M 维强耦合。而 N 维对 layer0 不能切（**事实**）："the computation of a GEMM tile involves multiplication and reduction along the token embedding dimension"——沿 N 切会破坏 tile 内的归约。

最细粒度是"一根行/列"，但算力效率极低（GEMM 需要 tile 才有高利用率），所以分解之后必须**重排成 tile**（**事实**，§4.1），两条原则：① 子张量要**对齐原有计算 tile 粒度**；② 调度要**优先安排 consumer 能立刻用上的部分**。

具体做法：**layer0（沿 M 切）**按 **token 来源 rank 排序**（**事实**，Fig.5），GroupGEMM 的 tile 计算顺序设计成**尽量少依赖远端数据**——从只含本地 token 的 tile 开始算，同时远端 token 还在传：

```
本地 token 到齐 ──► [tile_A: 全本地] 开始算！（无需等待）
远端陆续到   ──► [tile_B: 本地+远端] ──► [tile_C: 全远端]
        ↑ 通信与计算在 tile 粒度上自然重叠
```

**layer1（沿 N 切）**让 GroupGEMM **按列执行**（而不是按专家顺序逐个算），于是**前 `T_N` 列算完就能立刻开始归约与通信**，不必等所有专家（**事实**，Fig.6）：

```
不重排：  专家0 全算完 → 专家1 全算完 → 专家2 全算完 → 才能归约
重排后：  列0-31 → 归约传输 → 列32-63 → 归约传输 → 列64-95 → …
```

### 机制 2：线程块特化（thread block specialization）

分解+重排之后，重叠变得规则了。但要让通信延迟真正被藏住，**通信与计算的耗时必须匹配**——否则一边等一边（**事实**，§4.2）。"天真做法"是把整个流水封进**同构线程块**、把通信 I/O 塞进 GEMM 的 prologue 或 epilogue（论文叫 **vertical fusion**），它不行：线程块并发执行导致重叠不规则 → 通信与计算延迟**不确定** → 无法平衡；而且 token 级细粒度 I/O 会显著降低 kernel 效率（Hopper 尤甚）。

**Comet 的做法：通信线程块与计算线程块物理隔离。**

```
一个融合 kernel 内，n 个线程块分成两类：
  n_p 个 producer（通信）：读 consumer warp 写进 global memory 的结果
                          → 做 top-K 归约 → 写本地或发往远端
  n_c 个 consumer（计算）：用**和未融合时完全相同**的 CUTLASS 实现
                          → 不改 GEMM 代码，不受通信干扰
```

三个关键收益（**事实**）：① **GEMM 代码零改动**（复用 CUTLASS），算力效率不降；② 通信对计算的影响被**隔离**（"Comet isolates the impact of communication on computation performance"）；③ 通过调整 `n_p / n_c` 的**数量比例**可以精确平衡两侧延迟，减少重叠气泡。

最优的 `n_p / n_c` **不是固定的**（**事实**，§4.2）：实测受输入形状、序列长度、并行策略影响，要按场景测。为什么不把通信 warp 与计算 warp 放进**同一个**线程块（能省数据搬运）——论文试过，两个原因不行（**事实**）：warp 数量有限导致通信 warp 吃不满带宽；通信 warp 会**干扰**同块内的计算 warp。

### 3. 集成与测试

集成进 **Megatron-LM**，验证了多种并行策略（**事实**，§1）。测试平台：NVIDIA **H100** 与 **L20**（**事实**）。

## 关键数字

| 指标 | 数值 | 等级 |
|---|---|---|
| 单个 MoE 层加速 | **1.96×** | 事实 |
| 端到端加速（平均） | **1.71×** | 事实 |
| MoE 层通信占执行时间（基线，8×H100，Megatron-LM） | **47%** | 事实 |
| 生产部署规模 | **万卡级 GPU 集群** | 事实 |
| 生产节省 | **数百万 GPU 小时** | 事实 |
| 测试模型 | Mixtral-8x7B、Qwen2-MoE、Phi3-MoE 等 | 事实 |
| 测试硬件 | H100、L20 | 事实 |

**1.96× vs 1.71×** 的差距值得注意：单层加速明显高于端到端。说明 MoE 层之外的部分（注意力、其它算子）没有被优化，也说明 **MoE 层的通信确实是主要瓶颈**（否则单层优化不会带来 1.71× 端到端）。

"数百万 GPU 小时"是个有意思的间接证据：按 1.71× 折算，如果原来用了约 500 万 GPU 小时，优化后省下约 200 万。这个量级说明它确实在超大规模上跑着（**推断**）。

## 为什么值得读

- **它把"重叠"这件事做到了最细的粒度**。本 track 里 **C-12 ODC** 做通信**原语**级重叠（去掉同步屏障）、**C-10 OctoPipe** 做**调度图**级重叠（重排指令）、**C-13 Comet** 做**kernel 内线程块**级重叠。三者是同一件事的三个层级且**粒度递减**；Comet 是最细的一层，也是唯一直接操作 GPU 硬件资源（线程块、SM）的一篇。
- **"粒度错配"这个诊断框架非常通用**。它不只是 MoE 的问题：任何"生产者/消费者粒度不同"的流水（如 C-02 HSTU 的 embedding lookup vs GEMM、C-03 MTGR 的 dispatch stream vs compute stream）都可以用这个框架分析。**这是一个可以反复使用的分析工具**。
- **与 C-14 GBA、C-12 ODC 共享同一个"动态负载 + 静态调度"的困境**：Comet 在 kernel 内运行时自适应分配线程块，GBA 在 token 层控制 staleness，ODC 在 minibatch 层放宽屏障——三者都是"**让调度跟着负载动**"。
- **对 NestPipe 的价值**：① MoE 正在进入推荐模型（C-03 MTGR 的 baseline 里就有 MoE-4experts；C-15 Engram 讨论 MoE 与条件记忆的分配），Comet 的方法可直接迁移到推荐 MoE（**推断**）；② **线程块特化**与"通信-计算分离"的思路，本质上和 C-06 DMT 的"把全局 AlltoAll 拆成局部洗牌"同源——都是"用更细、局部性更好的操作替代全局大操作"；③ 它是本 track 里**唯一有"万卡生产 + 百万 GPU 小时"证据**的论文（C-03 是"主流量"、C-05 是"数十亿用户"，但都没这么具体的成本数字）。
- **1.96× 单层加速的含金量**：在已经高度优化的 Megatron-LM + CUTLASS 上再拿近 2×，且不改 GEMM 代码（靠线程块隔离），这是很硬的工程结果。

## 局限

1. **强绑定 Hopper**。线程块特化、warp specialization、SM 数量约束（"Each SM only accommodate one thread block"）、CUTLASS 实现都与 Hopper 深度耦合，换架构要重做（**推断**）。
2. **`n_p / n_c` 需要离线测**。最优点受输入形状、序列长、并行策略影响，要实测。这意味着**动态负载下这个"自适应"其实是"预先 profile 出的静态比例"**，不是真正的运行时自适应（**推断**）——论文说 "adaptive workload assignment"，机制上是"按 profiling 结果配置线程块数"。
3. **只优化 MoE 层**。端到端 1.71× vs 单层 1.96× 的差距说明其它部分未动；若模型里 MoE 占比更低，收益会更小（**推断**）。
4. **未涉及专家并行的拓扑优化**。它不改 token 路由（gate 不变）、不改专家放置，所以与"拓扑感知的专家放置"（类似 C-06 DMT）正交、可叠加（**推断**）。
5. **完整实验表未取到**。本文件取得的正文数值集中在 §1 与 §4，Fig.8 之后的逐模型/逐序列长数据抽取不全（**待核实**）。
6. **"数百万 GPU 小时"不可验证**。自报数字，未给计算方式（**待核实**）。
7. **未讨论与 TP/EP 混用时的交互**。论文说 "verified the capability with various parallel strategies"，但没给 EP+TP+PP 全开的场景数据（**待核实**）。

## 证据与待核实

| 结论 | 等级 | 依据 |
|---|---|---|
| 作者、arXiv 编号、2025 年 | 事实 | Zotero + abs |
| 47% 通信占比（8×H100 / Megatron-LM） | 事实 | §1 + Fig.1(a) |
| shared tensor 分解维度（layer0 沿 M、layer1 沿 N） | 事实 | §4.1 |
| 线程块特化、vertical fusion 的失效分析 | 事实 | §4.2 |
| 1.96× / 1.71× / 万卡 / 百万 GPU 小时 | 事实 | 摘要 + §1 |
| 集成 Megatron-LM、测 H100 + L20 | 事实 | §1、§5 |
| "adaptive" 实为 profile 出的静态比例 | 推断 | 由 §4.2 的调参描述推出 |
| 百万 GPU 小时的计算方式 | **待核实** | 未给 |
| 与 EP+TP+PP 全开的交互 | **待核实** | 未给详细数据 |
| 是否发表于正式会议 | **待核实** | arXiv abs 无 venue |

没拿到的东西：正式发表版（若有，可能是 MLSys/OSDI 系）；Fig.8 之后的完整实验数据；Comet 开源代码（摘要说"will open-source"）。

## 是否值得读原文

**挑章节读**。

理由：核心在 §4.1（shared tensor 的分解维度怎么选——这是全篇最有迁移价值的部分，尤其"layer0 沿 M、layer1 沿 N"这个非对称结论）和 §4.2（线程块特化的动机与设计）。§2.2.1 的"粒度错配"论述值得一读，是个好用的分析框架。若你不写 kernel，§4.2 的 CUTLASS/warp 细节可跳过，只记"通信与计算线程块隔离 + 调数量比"这个结论。
