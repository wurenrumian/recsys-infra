---
title: OctoPipe - Reducing Pipeline Bubbles for Heterogeneous Models via Co-Optimizing Partitioning, Placement, and Scheduling
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-C, pipeline-parallelism, heterogeneous-model, bubble, co-optimization, SC26]
related: [docs/papers/nestpipe/index.md]
zotero_key: UFFU6KVA
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Jihu Guo、Tenghui Ma、Wei Gao、Peng Sun、Xun Chen、Jiaxing Li、Zhisheng Ye、Yuyang Jin、Dahua Lin（**事实**，arXiv abs 页）。
- 单位：复旦大学 & 上海人工智能实验室、香港科技大学、香港中文大学 & 商汤（**事实**，正文 affiliation 行；但注意见下"版本差异"）。
- 年份与会议：arXiv v1 2025-09-28，v3 2026-09-02，**已被 SC '26 接收**（**事实**：arXiv abs 页 comments 写 "14 pages, 13 Figures; Accepted by SC'26"）。
- arXiv：`https://arxiv.org/abs/2509.23722`。
- 代码量：约 **14,000 行** Python（**事实**，§6 实现段）。

> **版本差异提醒（重要）**：本次取到的 `https://arxiv.org/html/2509.23722v1` 的 `<title>` 是 **"AdaPtis: Reducing Pipeline Bubbles with Adaptive Pipeline Parallelism on Heterogeneous Models"**，正文全篇自称 **AdaPtis**，而 arXiv 最新版（v3）与 Zotero 记录的标题都是 **OctoPipe**。两份标题、会议信息（v1 HTML 写 PPoPP 2026，v3 abs 写 SC'26）都不一致。合理判断：**这是同一工作的改名与改投稿**（**推断**）。本文件按 Zotero/最新版的 **OctoPipe** 命名，但正文机制描述来自 v1 的 AdaPtis 文本；引用时请注意这一点。

## 一句话

异构模型的气泡要治，得把切分、放置、调度三件事一起调，不能只调一样。

## 背景与问题

### 模型越来越不"齐整"

经典 PP（GPipe / 1F1B / interleaved）都隐含假设：**模型是 N 个同构 Transformer 层的堆叠**，于是"每层切一组、均分到各卡、按固定顺序跑"就够了。现代 LLM 把这个假设打碎了（**事实**，§1）：**Gemma** 词表极大（**256K** vs LLaMA-2 的 32K）→ 输出层是个巨块；**DeepSeek** 是 FFN 层与 MoE 层混排；**Nemotron-H** 是自注意力（SA）与 Mamba 层混排。异构 → 各层耗时差异大 → **气泡加剧**（**事实**，Fig.1：I-1F1B、ZB、Mist 在 Gemma/DeepSeek/Nemotron-H 上的气泡率都高于在 LLaMA-2 上）。

### 为什么现有方法不够

PP 有三个可优化的环节（**事实**，§2）：**模型切分**（模型切成若干 stage，每个 stage 是一组连续的层）、**模型放置**（stage 映射到设备，决定设备间数据依赖）、**工作负载调度**（按什么顺序跑 F / B / W，其中 F=前向、B=输入梯度、W=参数梯度，后两者是反传拆分）。现有方法各自只优化**一个**（**事实**，Table 2 的分类）：S-1F1B（均分 / 顺序映射 / 固定）、I-1F1B（均分 / 虚拟 stage 交错 / 固定）、ZB（均分 / 顺序 / 自适应拆 B-W）、Mist（**自适应**切分 / 顺序 / 固定）、Tessel（均分 / 顺序 / 自动生成 JSSP）。

后果（**事实**，§1）："prior partially adaptive PP methods often deliver only marginal improvements and **sometimes even degrade efficiency** on heterogeneous models"。实测：I-1F1B 在 Nemotron-H (Large) 上吞吐**下降最多 22%**；ZB 相对 S-1F1B 只有 **1.02×**。

### 三件难事

论文把共优化的障碍列得很清楚（**事实**，§1）：① **性能建模复杂**——三个环节互相影响，任一处改动会改变其它两处的代价；② **搜索空间组合爆炸**——切分方案数随层数增长、放置方案数随 stage 数增长、调度方案数随 microbatch 数增长（Fig.4 画了爆炸曲线）；③ **执行顺序不规整**——共优化产生的调度往往不是规则模式，需要一个能跑任意顺序、且**不能死锁**的执行器。

## 核心做法

三个部件，对应三个障碍。

### 1. Pipeline Performance Model：把"执行"变成可计算的

一个细粒度模拟器，输入训练设置 + profiling 数据 + pipeline 配置，输出**每台设备**的执行时间 `T_d` 和显存代价 `M_d`，且这些输出里**显式包含** `BubbleTime(d)` 和 `OverlapTime(d)`（**事实**，§4.2）。

三步计算（**事实**）：**Step 1 给定切分** —— 每个 stage 的算力/显存 = 该 stage 内各层 profiling 代价之和（`C_s`、`M_s`）；**Step 2 给定放置** —— 每台设备的代价 = 其上所有 stage 之和 + 通信（`C_d`、`M_d`）；**Step 3 给定调度** —— 模拟每台设备的执行行为，识别空闲时间的位置与成因得 `BubbleTime(d)`、识别通信可与计算重叠的机会得 `OverlapTime(d)`，于是

```
T_d ← C_d + BubbleTime(d) − OverlapTime(d)
```

**优化目标**是 `min max_d T_d`，受 `M_d ≤ M_d^capacity` 约束（**事实**）——即"最小化最慢设备的耗时"，受每卡显存上限约束。论文报告这个模型的**平均吞吐预测误差 2.12%**（Nemotron-H、SeqLen=4K 上）（**事实**）。

### 2. Pipeline Generator：在爆炸空间里贪心迭代

面对组合爆炸，做法是**不穷举**（**事实**，§4.3）：先从代表性 baseline 起步（切分用 S-1F1B / Mist 的策略，放置用 S-1F1B / I-1F1B / Hanayo，调度用 S-1F1B / ZB），用 Performance Model 估算并剪掉差的；再从较优起点迭代——找出 bottleneck（贡献 `BubbleTime` 最大的 phase），针对性地调切分/放置/调度中的**一项**，重新估算并保留改进，直到收敛。

三个调节动作（**事实**，Fig.6）：

**① 调切分**：前人只均衡 `C_d`。OctoPipe 把**通信代价和内存约束**一起纳入（所以切出来的分组不一定等大，但总耗时更均衡）。

**② 调放置**：**置换每个 stage 内各层的设备分配**，把相邻层放到**不同**设备上。效果是"有效 stage 数变多"，从而减少气泡（这正是 I-1F1B 的虚拟 stage 思路，但这里是自动搜索出来的、可以是非均匀的）。

**③ 调调度**（三条策略）：

| 策略 | 做法 | 为什么有效 |
|---|---|---|
| 填充气泡 | 显存约束内**先推 F**，再 B，最后 W | `W`（参数梯度）**无跨设备依赖**（只依赖本设备的 B），最适合塞气泡 |
| 重叠感知 | 避免把有依赖的任务连续排；**刻意延迟**某些计算以造出通信重叠 | 通信可与不依赖它的计算并行 |
| 防 OOM | 记录调度中的潜在 OOM 时刻，**提前**推进最新 B/W 释放显存 | 满足显存约束 |

### 3. Pipeline Executor：能跑任意顺序且不死锁

把计算和通信抽象成指令，然后编排（**事实**，§4.4）。指令集包括计算指令（F/B/W）与通信指令（`S_F|B` 发送、`R_F|B` 接收、`W_F|B` 等待）。

生成分两步（**事实**）：

```
Step 1  按调度结果生成计算指令列表（保持每台设备上的原始顺序，维持数据依赖）
Step 2  插入通信指令：
         若某计算需要别的设备的结果 → 在它之前插 R_F|B
         对应地，发送方在产生结果后插 S_F|B
```

**死锁问题**很具体（**事实**，§4.4 举例）：`S_F|B` 和 `R_F|B` 必须在收发两侧同步执行，顺序不对就互等。

```
GPU1: ... C_F → S_F ...        ↑ 两边都在等对方的 R，互锁
GPU2: ... C_B → S_B ...
```

解法：**遍历每台设备上所有通信指令，检查每个 `S` 与其对应 `R` 的先后关系，发现有死锁风险就重排**。

**重叠优化**（**事实**）：`R_B` 若被放在紧邻 `C_B` 之前就无法重叠（`C_B` 依赖它收到的数据）。执行器会**向前搜索插入点**——比如挪到 `C_F` 之前（两者无依赖），于是 `C_F` 执行时 `R_B` 可异步进行。

## 关键数字

| 指标 | 数值 | 等级 |
|---|---|---|
| 平均加速 vs **S-1F1B** | 最高 **1.34×** | 事实 |
| 平均加速 vs **I-1F1B** | 最高 **1.42×** | 事实 |
| 平均加速 vs **ZB** | 最高 **1.34×** | 事实 |
| 平均加速 vs **Mist** | 最高 **1.20×** | 事实 |
| 长序列下的加速（Nemotron-H Large, 1K–32K） | 1.54× / 2.14× / 1.51× / 1.27×（对 S-1F1B / I-1F1B / ZB / Mist） | 事实 |
| 摘要口径 | **1.09–1.49×**（跨异构配置与集群规模） | 事实 |
| 性能模型预测误差 | 平均 **2.12%** | 事实 |
| 测试集群 | **128× H800**，节点内 NVLink，节点间 InfiniBand | 事实 |
| PP size 扫描 | P = 4、8、16 | 事实 |
| 代码量 | ~14,000 行 Python | 事实 |
| I-1F1B 在 Nemotron-H Large 上的退化 | **−22%** | 事实 |
| ZB 相对 S-1F1B | 仅 **1.02×** | 事实 |

**注意摘要口径与正文口径不同**：摘要是 1.09–1.49×（对 "state-of-the-art PP approaches" 整体），正文是分基线的最优值（1.20–1.42×）。引用时用摘要区间更稳。

## 为什么值得读

- **它把 PP 优化从"设计一个调度模式"变成"搜索一个调度方案"**。S-1F1B / ZB / I-1F1B 都是人手设计的固定模式；OctoPipe 的立场是"模式应该随模型形状自适应"。这是方法论层面的转变，比某个具体加速比更重要。
- **与 C-08 SlimPipe 是"互补但互斥"的一对**。SlimPipe 假设**层内同构**（才能均匀切片），OctoPipe 恰恰处理**层间异构**——适用条件互斥：同构模型 + 超长上下文 → SlimPipe；异构模型 + 常规上下文 → OctoPipe。有趣的是，SlimPipe 的"输出层词表并行"其实就是在处理**词表这一层的异构性**，而 OctoPipe 把这类异构交给搜索器处理。两种思路的对比很有启发（**推断**）。
- **与 C-09 TD-Pipe 是"训练 vs 推理"的又一对照**，但异构源不同：OctoPipe 是**模型异构**，TD-Pipe 是**请求异构**。
- **与 C-12 ODC、C-13 Comet 同属"通信-计算重叠"这一族，但层次不同**：Comet 在 **kernel 内**（线程块级）、ODC 在 **通信原语**层（去掉同步屏障）、OctoPipe 在 **调度图**层（重排指令）。三篇合看能拼出"重叠"的三个粒度。
- **2.12% 的建模误差是被低估的贡献**。绝大多数"自动调优"类论文的性能模型误差在 5–10% 量级，2.12% 意味着搜索结果可信；它也让"改一处、估一次"的迭代策略在计算上可行。
- **对 NestPipe 的价值**：推荐模型比 LLM 更异构（embedding 层、交叉层、注意力层、MLP 混排），而现有推荐训练系统（TorchRec / C-03 MTGR 的框架）几乎都用固定调度。OctoPipe 的方法论——**把切分/放置/调度交给搜索**——在推荐场景的应用空间可能比在 LLM 场景更大（**推断**，论文未涉及推荐）。

## 局限

1. **只做训练**。推理的调度约束完全不同（无 B/W、有 KV cache、请求动态到达），方法不能直接搬（**推断**）。
2. **搜索本身的开销未充分披露**。论文声称比 ILP solver 快（Fig.13），但**没有给出绝对搜索时间**与"模型变化时要不要重搜"的成本（**待核实**）。
3. **依赖 profiling 质量**。Performance Model 的输入是 profile 出的每层代价；profile 不准，搜索就偏。论文未讨论 profile 的成本与噪声敏感性（**待核实**）。
4. **14,000 行代码 = 高移植门槛**。它是在 PyTorch 上独立实现的执行器，不是给 Megatron 打补丁，落地要替换现有框架的 PP 层（**推断**）。
5. **重算优化被排除**。论文明确说不与 AdaPipe / Mario 比较，因为"recomputation 与本文正交"（**事实**）——留了叠加空间，也意味着实际收益可能更大或更小。
6. **版本混乱影响引用**。v1 叫 AdaPtis、投 PPoPP 2026；v3 叫 OctoPipe、被 SC'26 接收，两版标题、会议、甚至可能的实验数字都不同（**事实**：本文件读的是 v1 HTML）。**引用时必须核对版本**。
7. **异构性覆盖有限**。只测 Gemma（词表大）、DeepSeek（FFN+MoE）、Nemotron-H（SA+Mamba）三类，更极端的异构（如 MoE + 多模态编码器）未覆盖（**推断**）。

## 证据与待核实

| 结论 | 等级 | 依据 |
|---|---|---|
| 作者、arXiv 编号、SC'26 接收 | 事实 | arXiv abs 页 |
| v1 HTML 标题为 "AdaPtis"、正文自称 AdaPtis | 事实 | 取到的 HTML |
| 三部件架构、三步性能模型、三条调度策略 | 事实 | v1 §4 |
| 1.34×/1.42×/1.34×/1.20×、2.12%、128×H800 | 事实 | v1 §6 |
| 摘要 1.09–1.49× | 事实 | arXiv abs（v3） |
| "AdaPtis 与 OctoPipe 是同一工作" | **推断** | 作者/摘要/机制一致，仅名称与投稿变更 |
| 搜索时间绝对值 | **待核实** | 只有相对对比 |
| profile 成本与敏感性 | **待核实** | 未讨论 |
| v1 与 v3 实验数字是否一致 | **待核实** | 只读了 v1 HTML |

没拿到的东西：v3 的 HTML/PDF（SC'26 版，可能有更新的实验与更正的措辞）；正式版 PDF。

## 是否值得读原文

**挑章节读**。

理由：如果你的工作是"给现有 PP 换调度"，§4.2（性能模型三步）和 §4.3（三条调度策略：F/B/W 顺序、重叠感知、防 OOM）是可直接借用的设计。§4.4 的死锁检测与指令重排是工程难点，值得对着 Figure 7 看。**但一定要读 v3（SC'26 正式版）而不是 v1**——本文件基于 v1，名称与部分数字可能与最终版有出入。
