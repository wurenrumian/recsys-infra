---
title: The Evolution of Embedding Table Optimization and Multi-Epoch Training in Pinterest Ads Conversion
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-B, multi-epoch-overfitting, sparse-optimizer, learning-rate, pinterest]
related: [docs/papers/nestpipe/index.md]
zotero_key: L2VVL9ZT
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Andrew Qiu、Shubham Barhate、Hin Wai Lui、Runze Su、Rafael Rios Müller、Kungang Li、Ling Leng、Han Sun、Shayan Ehsani、Zhifang Liu（Pinterest Inc. / UC Irvine）。
- 年份：2025 年 5 月 8 日，arXiv:2505.05605。文中标注投稿会议为 **KDD 2025**（31st ACM SIGKDD，2025-08-03 至 08-07，Toronto），但 DOI 字段仍是占位符 `XXXXXXX.XXXXXXX`，**是否被接收待核实**。
- Zotero 记为 preprint，`https://arxiv.org/abs/2505.05605`。

## 一句话

给嵌入表配 50× 学习率、给低频行降学习率，多 epoch 过拟合基本能治。

## 背景与问题

这是本 track 里**唯一一篇纯工业实践报告**：不讲新架构、不讲新硬件，讲"我们在 Pinterest 广告转化模型上踩过的坑和试出来的招"。

问题的特殊性在于**多任务 + 转化预测**（**事实**，摘要与 §4）：

| 目标 | 相对点击的正样本密度 |
|---|---|
| p(click) | 1.00 |
| p(good click) | 0.15 |
| p(add-to-cart \| view, no click) | 0.03 |
| p(add-to-cart \| click) | 0.007 |
| p(checkout \| view, no click) | 0.01 |
| **p(checkout \| click)** | **0.002** |

最稀疏的目标（`p(checkout|click)`）正样本密度只有点击的 **1/500**。这就是"标签极度稀疏"的量化含义。

嵌入表训练困难的四个原因（**事实**，摘要）：
1. **梯度稀疏**：每个 batch 只有一小部分行被更新 → 收敛慢；
2. **基数高**：Campaign ID 这类特征有上百万唯一值；
3. **ID 分布不均**：50% 的频次由 **0.74%** 的 ID 贡献（**事实**，Figure 1 的说明）；
4. **标签极度稀疏**：见上表。

这四个因素叠加导致两个病：
- **收敛慢**；
- **第一 epoch 之后过拟合**（"one-epoch phenomenon"，测试性能在新 epoch 开始时"sharp, unrecoverable drops"）。

已有做法的不足（**事实**，§1）：
- **降基数**（哈希、过滤低频 ID）：损害性能，因为"稀释了嵌入的粒度"；
- **全局手段**（降学习率、嵌入重初始化 MEDA、各种正则）：**不区分行的高低频**，"partially mitigating overfitting ... often do not differentiate between infrequent and frequent rows"。

这就是本文的两个着手点。

## 核心做法

### 两个技术，一快一稳

```
┌──────────────────────────────────────────────────────────────┐
│ Sparse Optimizer：给嵌入表一个更高的 layer-specific 学习率      │
│   目的：解决"收敛慢"                                           │
│   做法：Adam 之上，嵌入表参数的学习率 ×50，其他（稠密）参数不变 │
├──────────────────────────────────────────────────────────────┤
│ FAL（Frequency-Adaptive Learning Rate）：按行的频次缩放学习率   │
│   目的：解决"多 epoch 过拟合"                                  │
│   做法：低频行 → 学习率压低；高频行 → 基本不动                 │
└──────────────────────────────────────────────────────────────┘
```

这两个是**互补**的，甚至方向相反：一个整体加速（尤其对中高频行），一个专门压制低频行。

### ① Sparse Optimizer

**做法极简**（**事实**，§3.1）：在 Adam 之基础上，**给嵌入表一个更高的 layer-specific 学习率**，保持所有其他（稠密）参数的学习率不变。

调出来的倍数是 **50×**（**事实**，§5.1："tuned to have a 50x larger embedding table learning rate relative to the rest of the model"）。

**为什么能work**（**推断**）：梯度稀疏意味着每行更新次数少，等效于"学习率被稀释"。给分层学习率本质上是**补偿稀疏性带来的有效步长损失**。

**效果**（**事实**，§5.1）：
- 训练损失在所有时间点都更低。第一个 10000 迭代之后，`p(checkout|click)` 的训练损失至少低 **0.41%（0.0012）**，`p(checkout|view,no click)` 低 **0.69%（0.00026）**。
- 累计 AUC：相对"同训练天数的重训基线"高 **0.10%**；相对"有 4 个月持续训练优势的生产模型"，仍追平并超出 **0.017%**。

**已上线生产**（**事实**，§6.2："The Sparse Optimizer has already been launched to production models"）。

### ② FAL（核心创新）

**动机**（**事实**，§3.2）：多 epoch 过拟合的机制是
> "the **overfitting of infrequent rows** in the embedding layer, **followed by the rapid adaptation of downstream layers** to these biases"

即：**低频行先过拟合 → 下游层快速适应这些偏差**。所以低频行是元凶。

已有做法（整体降学习率）能缓解但代价是峰值性能下降。FAL 的思路是**只惩罚元凶**：

```
η_T[i] = η*_T · log(F_T[i] + 1) / max_j log(F_T[j] + 1)
```

其中：
- `F_T[i]`：表 T 第 i 行的累计频次；
- `η*_T`：表 T 的 layer-specific 学习率（来自 Sparse Optimizer）；
- `η_T[i]`：该行的实际学习率。

**关键设计选择：对数缩放（log scaling）而非线性**（**事实**，§3.2 与 Table 1）：

> "We scale row-wise learning rates by their **relative log frequency**, as to **prevent highly frequent rows from dominating the learning rate distribution**."

实证（**事实**，Table 1，10 天持续训练后相对 60 天 2-epoch 训练的累计 AUC 增益）：

| 缩放方式 | 累计 AUC 增益 |
|---|---|
| **Log Scaling（对照组）** | **0.00%** |
| Linear Scaling | **−0.13%** |

即：线性缩放**反而变差**。这是一个有价值的负面结果——直觉上行级缩放越"忠实"越好，但线性缩放会让极少数超高行占据学习率分布的高端，把绝大多数行的学习率压得太低。

**实现**（**事实**，Algorithm 1）：

```
F_T ← 0  （每张表一个长度 n 的向量）
for batch B:
    Forward Pass
    Backward Pass
    for each Embedding Table T:
        F_T ← F_T + f_T(B)                 # 累积本 batch 的频次
        α_T ← log(1 + F_T) / max(log(1 + F_T))
        ∇T ← ∇T · α_T                       # 梯度乘缩放系数
    Parameter Update
```

三个实现要点（**事实**，§3.2.1）：
- **不需要预计算**：边训练边累积，"offline testing 显示有无预计算性能差异可忽略"；
- **内存开销极小**：每表一个 32-bit 无符号整型张量。固定嵌入维度 32 时，开销相当于"把每张表的嵌入维度加 1"，即 **3.125%** 的嵌入表内存，**不影响其他层**；
- **不干扰梯度裁剪与动量**：FAL 可以理解为"每个嵌入的学习率调度"，与 Adam 的 momentum 项、gradient clipping 正交。

### ③ 与 MEDA 的对比

**MEDA**（Multi-Epoch learning with Data Augmentation，Fan et al. 2024）：在第二个 epoch 前**重新初始化嵌入表**。

对比结果（**事实**，§5.3，Aug–Dec 数据集）：

| 目标 | MEDA | FAL | 2-epoch 基线 |
|---|---|---|---|
| `p(add-to-cart\|click)`（次稀疏） | 优基线 **0.36%** | 优基线 **0.31%** | — |
| `p(checkout\|click)`（最稀疏） | 优基线 **0.27%** | **≈基线**（未能缓解） | — |

**结论**：FAL 在除最稀疏目标外均可比，**在最稀疏目标上不如 MEDA**。

### ④ 最重要的一个"反高潮"结论

**事实**（摘要与 §5.4）：

> "Although both MEDA and FAL achieve superior performance compared to a multi-epoch baseline on a validation set, these gains **vanish after several days of continual training as the baseline catches up** in performance. As such, we demonstrate how **solving multi-epoch overfitting may be unnecessary as long as fresh data is available** and overfitting is not too severe."

数据（**事实**，§5.4）：
- May–Aug 数据集：FAL 累计 AUC 增益 0.07%。2-epoch 基线初期因过拟合不如 1-epoch，但**持续训练 8 天后追平 FAL**。
- Aug–Dec 数据集：MEDA 初期最好，但**2 天持续训练后三者相近**。整体上 FAL 0.20%、MEDA 0.19%、2-epoch 基线 0.18%——**差距已经很小**。

这个结论对工业界极有价值：**如果你的训练流程是"批量训练 → 每日增量训练"，那多 epoch 过拟合不值得投入工程去治**，因为增量训练会用新数据纠正它。反之，如果是"纯多 epoch 在线学习"（没有新鲜数据来纠偏），那就必须治。

### ⑤ 另一个观察：过拟合程度与标签密度、数据量相关

**事实**：
- 标签密度高的目标（`p(click)` 1.00、`p(add-to-cart|view,no click)` 0.03）→ 过拟合轻，损失曲线几乎单调下降；
- 标签密度低的目标（`p(add-to-cart|click)` 0.007、`p(checkout|click)` 0.002）→ 过拟合重。例如 `p(add-to-cart|click)` 在 epoch 边界损失涨 **1.19%（8.80e-5）**，而 `p(add-to-cart|view,no click)` 反而降 **0.37%（1.11e-4）**。
- **数据集大 25% 时过拟合明显变轻**：Aug–Dec 数据集总损失只在 epoch 边界涨 **0.07%（3.66e-4）**，而 May–Aug 涨 **0.13%（6.82e-4）**；`p(add-to-cart|click)` 从 1.19% 降到 0.24%。

这条观察的价值：**过拟合的严重程度是可预测的**（由标签密度与数据量决定），不需要每次都做实验诊断。

## 关键数字

| 指标 | 数值 | 证据 |
|---|---|---|
| 模型总参数 | 2.98 亿（298M） | **事实**（§4.5） |
| 其中嵌入参数 | 约 **1.67 亿**（167M） | **事实** |
| 嵌入表数量 | 约 60 张（多数基数很小） | **事实** |
| 嵌入维度 | 全部固定 **32** | **事实** |
| 高基数特征哈希 | 二值哈希，最多 **21 bit** | **事实** |
| Sparse Optimizer 学习率倍数 | **50×** | **事实** |
| 基础学习率 | 0.00015（Adam） | **事实** |
| batch size | 2000 | **事实** |
| Adam 参数 | ε=0.00001, β=(0.9, 0.999) | **事实** |
| 批量训练数据跨度 | 110 天（2024-05 至 2024-08） | **事实** |
| FAL 内存开销 | 嵌入表的 **3.125%** | **事实** |
| 上线判定阈值 | 内部离线累计 AUC 门槛 **+0.10%** | **事实** |
| 频次分布 | 50% 频次由 **0.74%** 的 Campaign ID 贡献 | **事实** |

**关于 1.67 亿 / 2.98 亿 = 56%**：这个模型的嵌入参数只占一半多，远不是 B-09（HPS）那种 99% 的情形。所以 Pinterest 的问题是"训练动力学"，不是"装不下"。**这是理解本篇定位的关键。**

## 为什么值得读

Pinterest 这篇在 Track B 里是**唯一一篇把"嵌入表问题"从系统层拉回训练动力学层的**，它的价值主要在于打破其他篇的隐含假设：

### 它对 Track B 其他篇的三处挑战

**① 挑战 B-12（AdaEmbed）的"低频行不重要"。**
AdaEmbed 说低频行重要性低，直接剪掉。Pinterest 说低频行是**多 epoch 过拟合的元凶**——它们有很强的拟合能力（能记住噪声），所以"不重要"这个判断是错的：它们对训练损失贡献很大，只是对泛化有害。
- 但换个角度看，**这两者其实一致**：如果低频行的贡献主要是过拟合，那剪掉它们等于"消除过拟合源"，效果应该和 FAL 类似。而 AdaEmbed 观察到"剪完精度还涨了"——这正好是过拟合被缓解的表现。
- 所以更准确的说法是：**AdaEmbed 无意中做了 Pinterest 想做的事，只是手段更粗暴**（**推断**）。

**② 挑战 B-11（DQRM）的"INT4 正则收益"。**
DQRM 观察到 INT4 位宽越低精度越高，归因为"强正则"。Pinterest 的解释提供了另一个视角：**过拟合来自低频行的快速拟合**。量化降低了所有行的表示能力，等效于限制了低频行的拟合速度。两者是**同一机制的不同表述**（**推断**）。
- 一个可检验的推论：如果 DQRM 的正则收益在"大 25% 数据集 + 高标签密度"的场景下（即 Pinterest 观察到过拟合很轻的场景）会消失，那么两者的解释就统一了。**DQRM 论文没有做这个实验。**

**③ 挑战 B-09（HPS）、B-02（HET）等的"内存是主要矛盾"前提。**
Pinterest 的模型只有 298M 参数，1.67 亿嵌入参数按维度 32 算，约 **5.3 GB**——单张 40GB 卡都装得下。**这个规模下根本不需要层级存储或分布式缓存。** 它的瓶颈在收敛速度与过拟合。
- 这说明"嵌入表太大"不是普适问题。按规模分层：
  - ~GB 级（Pinterest）：问题是**训练动力学**（收敛、过拟合）；
  - ~10GB–1TB 级：问题是**存储层级**（B-09、B-01、B-15）；
  - ~TB 级以上：问题是**分布式通信**（B-02、B-04、B-14）。
  
  **这个分层是本 track 最有用的组织框架之一。**

### 它的方法论价值

- **一篇文章里同时给出正面结果与"其实没必要"的结论**。§7 最后一句："solving multi-epoch overfitting may be unnecessary as long as fresh data is available"。这种自我否定的结论在工业论文里很少见，但对工程决策极有价值——它避免你为一个在持续训练下会自动消失的问题投入工程。
- **"分层学习率"这个极简手段的收益被量化了**：50× 倍数、0.10% 累计 AUC、已上线。没有复杂机制，只有正确的诊断。
- **对数缩放 vs 线性缩放的对比**（Table 1）是一个反直觉的实证：更"忠实"的线性缩放反而掉 0.13%。这类小实验的价值往往高于主结果。

## 局限

- **只在 Pinterest 的广告转化模型上验证**。单一业务、单一模型架构（多任务 + 分层 ensemble + Transformer/Discretization 等），外推到其他推荐场景的可靠性未知。
- **最稀疏的目标 FAL 失效**。`p(checkout|click)`（正样本密度 0.002）上 FAL 与基线持平，MEDA 更好。所以 FAL 不是万灵药，**它的有效区间是"中等稀疏"**。
- **核心结论是"不需要治"**。这让 FAL 的实际价值打折扣——论文自己的数据显示，在有持续训练的场景下，FAL 的 0.07%–0.20% 累计 AUC 增益会在几天内被基线追平。虽然 0.20% 超过了他们内部的 +0.10% 上线门槛，但**长期收益不明**。
- **"多 epoch 过拟合"的定义依赖指标**。论文用测试损失与 AUC 度量，但"sharp, unrecoverable drops"在本文的设置里其实**很轻微**（总损失只涨 0.13%、0.07%）——远不如文献（Zhang et al. 2022）描述的严重。作者自己说这是因为"高特征多样性 + 强哈希 + 多任务带来的更多训练数据"让模型更鲁棒。**所以这篇的"过拟合"可能本身就比较轻，结论的适用范围要打问号。**
- **FAL 需要维护每行的频次张量**。3.125% 的开销在维度 32 时成立；如果嵌入维度是 128，开销降到 0.78%，但如果是维度 8，就涨到 12.5%（**推断**，按 32bit/维度算：4 bytes / (8×4 bytes) = 12.5%）。
- **需要累积频次，冷启动阶段缩放系数不准**。训练开始时所有 F 都接近 0，`log(1+F)/max(log(1+F))` 的比值噪声大。论文说"无预计算差异可忽略"，但没报冷启动阶段的具体表现（**待核实**）。
- **会议接收状态待核实**。文中 DOI 是占位符，arXiv 首页也没有 "Accepted by KDD" 的 comments 字段。

## 证据与待核实

证据等级：**fulltext**。取到正文的 URL：`https://arxiv.org/html/2505.05605v1`（含 §1–§7、Table 1–2、Algorithm 1、Figure 1–9 的完整文本与数值）。

事实清单：FAL 公式 `η_T[i] = η*_T · log(F_T[i]+1)/max_j log(F_T[j]+1)`、Algorithm 1、50× 倍数、3.125% 开销、298M/167M 参数、60 张表、维度 32、21 bit 哈希、0.74%/50% 频次分布、标签密度表、所有 AUC/损失数值、与 MEDA 的对比、持续训练后差距消失、上线阈值 +0.10%、Sparse Optimizer 已上线。

待核实：
1. **会议接收状态**（KDD 2025？DOI 仍是占位符）；
2. 模型的具体架构细节（Figure 2 只有高层图）；
3. FAL 在冷启动/训练早期（F 接近 0）的数值稳定性；
4. "多 epoch" 具体到几个 epoch（文中主要是 2-epoch 实验，是否测过更多）；
5. 是否在纯多 epoch（无持续训练）场景下验证过 FAL（论文只是**推测**这类场景更受益）。

## 是否值得读原文

**必读**——理由不是"它给了好方法"，而是**它给了正确的诊断框架**：
1. §4.1 的标签密度表 + §5.2 的"过拟合程度与标签密度、数据量相关"——这套对应关系可以直接用来**预判**你的模型会不会有多 epoch 过拟合，不需要做实验；
2. §5.4 的"持续训练后差距消失"——这条结论能帮你**避免为一个会自动消失的问题投入工程**；
3. §3.1 的 Sparse Optimizer（50× 分层学习率）——最简单的改动、最低的风险、已上线验证的收益。

§3.2 的 FAL 公式很短，可以直接抄；但先读 §5.3 的对比，确认你的最稀疏目标不在它失效的区间里。
