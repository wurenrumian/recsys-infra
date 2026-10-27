---
title: "DQRM: Deep Quantized Recommendation Models"
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-B, quantization, qat, int4, gradient-compression, edge]
related: [docs/papers/nestpipe/index.md]
zotero_key: Z56XJYUC
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Yang Zhou、Zhen Dong、Ellick Chan、Dhiraj Kalamkar、Diana Marculescu、Kurt Keutzer（CMU / UC Berkeley / Intel / UT Austin）。
- 年份：2024 年 10 月 26 日。Zotero 记为 preprint（`https://arxiv.org/abs/2410.20046`），arXiv 许可证 CC BY 4.0。
- 代码：`https://github.com/YangZhou08/Deep_Quantized_Recommendation_Model_DQRM`。
- 系统名：**DQRM**（Deep Quantized Recommendation Model）。

## 一句话

INT4 量化缩 8×，还顺带治好过拟合，精度不降反升。

## 背景与问题

DLRM 的模型大小 99% 以上是嵌入表，1TB+ 的表在训练和推理上都是严重的内存瓶颈（**事实**）。作者关心三个下游目标（**事实**，§1）：
1. 云上推理更高效；
2. 把推荐模型放到**边缘设备**（手机）；
3. 优化**分布式训练的数据并行通信**。

已有的量化工作分两类，都不够：
- **PTQ**（Guan et al. 2019 把嵌入表量化到 INT4；Deng et al. 2021 把整个模型量化到 4 bit）——**都会掉精度**；
- **LPT**（低精度训练，Zhang 2018 / Xu 2021 / Li 2022）——**以掉精度换训练内存**。

作者要的是：**推理阶段做到超低精度（INT4）且不掉精度**，同时**让 QAT 本身在推荐模型上跑得动**。

### 两个被发现的"额外"问题

**问题一：DLRM 严重过拟合。** "testing accuracy dropping steeply after the first one or two epochs"（**事实**）。这是 B-16（Pinterest）和大量文献都在讲的 one-epoch phenomenon。

**问题二：朴素 QAT 在 DLRM 上极慢。** 作者做了时间拆解（**事实**，Figure 1(c)）：
- 无量化时（单卡 GPU、Kaggle），反向传播占大头：8.5ms 里占 5.8ms；
- 加上 QAT 后，**前向变得显著更长**，主要耗时在"**找量化 scale**"（scale 计算）；
- 找 scale 占单卡 GPU 训练时间的 **1/3 以上**；在单卡 CPU 上训 Terabyte 时**占 97% 以上**。

这个观察很重要：**QAT 的开销不在算力，而在遍历巨大张量求 min/max**。嵌入表 99% 是参数，遍历它就是遍历几乎整个模型。

## 核心做法

### 三个技术点

```
┌────────────────────────────────────────────────────────────┐
│ ① 只量化"用到的"权重（Reduce Memory of Unused Weights）      │
│    朴素 QAT: 复制整表 → 量化 → 存低精度影子副本             │
│    DQRM   : 先做 embedding lookup 取出稀疏向量 → 只量化这些  │
│             实际被访问的条目数比全表小 3 个数量级            │
├────────────────────────────────────────────────────────────┤
│ ② 周期性更新 scale（Periodic Update）                        │
│    不每步都遍历全表求 min/max，每 N 步（200 或 1000）更新一次│
│    把一次遍历的开销摊到成百上千次迭代上                      │
├────────────────────────────────────────────────────────────┤
│ ③ 梯度量化 + 只对 MLP 做误差补偿                             │
│    在已有 specified sparsity 之上，再把梯度压到 INT8         │
│    误差补偿只给 MLP 层（嵌入表对梯度量化鲁棒得多）           │
└────────────────────────────────────────────────────────────┘
```

### ① 只量化用到的权重

这是全文最巧的一步。看 Figure 2 的对比：

```
朴素 QAT（图 2a）:
  embedding table (整表) ──copy──► 低精度副本 (整表)
  ↑ 问题：99% 的参数没被访问，也照样复制、照样量化

DQRM（图 2b）:
  1. 先执行 embedding lookup → 得到 batch 内实际用到的稀疏向量集合
  2. 只对这些向量做 copy + quantize
  ↑ 被访问的条目数通常比全表小 3 个数量级以上（取决于 batch size）
```

数据支持（**事实**）：实际被访问的条目数"usually more than three magnitudes smaller than the total number of embedding vectors in the tables on average depending on the training batch size"。

代价（**推断**）：这改变了计算顺序——必须先做 lookup 再做量化，所以量化不能作为一个"前置的图节点"，要融合进 embedding 算子。这也是为什么必须自己写实现（论文开源了）。

### ② 周期性更新 scale

把"求量化 scale"（遍历全表算 min/max）从**每步**改成**每 N 步**：
- Kaggle 用 N = 200；
- Terabyte 用 N = 1000。

作者称这个改动"不伤收敛"，甚至有实验显示**周期性更新能提升精度**（**事实**，§3.2 末："periodic updates can even boost model accuracy"），ablation 在附录 A.1。

直觉（**推断**）：scale 的抖动本身是噪声源，冻结它几百步反而有稳定化效果。

### ③ 梯度量化 + 选择性误差补偿

先说为什么需要：PyTorch 内置的 **specified sparsity**（`EmbeddingBag` 对 sparse 梯度的天然支持）已经把 all-reduce 通信从 **2.161 GB 压到 2.010 MB**（Kaggle，4×A5000），**且是无损的**（**事实**，Table 4）。但剩下的通信仍是大头。

作者在此基础上再压 INT8。朴素做会有两个坑：
- **朴素 INT8 量化梯度会显著损害收敛**（附录 A.2 的 ablation）；
- **对所有参数都做误差补偿会爆内存**——误差缓冲区和嵌入表一样大。

关键发现（**事实**，§3.3）："gradients in MLP layers are more sensitive to quantization than those in embedding tables while embedding tables are much more robust"。所以：
- **只对 MLP 层做误差补偿**；
- 嵌入表的梯度直接 INT8，不加补偿。

效果：通信再压 **≈4×**（0.509 MB vs 2.010 MB），精度损失 Kaggle 上 0.057% / ROC AUC 0.0012。

**一个诚实的负面结果**（**事实**，Table 4 脚注 3）：因为 PyTorch 的 sparse tensor all-reduce **不支持低精度算术**，加上稀疏张量合并（coalescing）的开销，**实际每迭代延迟反而从 61ms 涨到 110ms**。作者明确说："achieving speedup from communication reduction remains difficult"，并把它归为超出本文范围的工作。这条自我限制很有价值——很多论文会把通信量降低包装成速度提升。

### 量化粒度选择

- **嵌入表：per-table 量化**（不是 per-row）。理由（**事实**）：per-row 会存大量 FP32 scale，每步都要存取，开销大。
- **MLP：channel-wise 量化**。理由（**事实**）：MLP 对量化更敏感，channel-wise 几乎没有精度损失，而 INT4 的 matrix-wise 量化"badly converges"。

## 关键数字

| 指标 | 数值 | 证据 |
|---|---|---|
| Kaggle 模型大小 | FP32 **2.161 GB** → INT4 **0.270 GB**（**8×**） | **事实**（Table 3a） |
| Kaggle 测试精度 | FP32 78.923% → DQRM INT4 **79.071%**（+0.148%） | **事实** |
| Kaggle ROC AUC | FP32 0.8047 → DQRM **0.8073** | **事实** |
| Terabyte 模型大小 | FP32 **12.575 GB** → INT4 **1.572 GB**（**8×**） | **事实**（Table 3b） |
| Terabyte 测试精度 | FP32 81.165% → DQRM **81.210%**（+0.045%） | **事实** |
| 找 scale 的时间占比 | 单卡 GPU 1/3+；单卡 CPU（Terabyte）**97%+** | **事实**（Figure 1c） |
| 稀疏化后通信量 | 2.161 GB → 2.010 MB（无损） | **事实**（Table 4） |
| 再加 INT8 后 | 0.509 MB（≈4× 再压缩） | **事实** |
| 精度代价（梯度量化） | −0.057% acc，−0.0012 AUC | **事实** |

嵌入表量化对比（**事实**，Table 2，Kaggle）：

| 权重位宽 | 测试精度 | ROC AUC |
|---|---|---|
| FP32 | 78.923% | 0.8047 |
| INT16 | 78.928%（+0.005%） | 0.8046（−0.0001） |
| INT8 | 78.985%（+0.062%） | 0.8054（+0.0007） |
| INT4 | **79.092%（+0.169%）** | **0.8073（+0.0026）** |

**位宽越低精度越高**——这是全文最反直觉、也最有说服力的结果。

与其他 QAT 方法对比（**事实**，Table 3a，Kaggle INT4）：

| 方法 | 模型大小 | 测试精度 | AUC | 每迭代训练时间 |
|---|---|---|---|---|
| FP32 基线 | 2.161 GB | 78.923% | 0.8047 | 7 ms |
| Vanilla PTQ | 0.270 GB | 76.571% | 0.7675 | — |
| PACT | 0.270 GB | 无法收敛 | — | — |
| LSQ | 0.270 GB | 78.972% | 0.8051 | 25 ms |
| HAWQ | 0.270 GB | 79.040% | 0.8064 | 31 ms |
| **DQRM** | 0.270 GB | **79.071%** | **0.8073** | **22 ms** |

注意：**HAWQ 精度也不差（79.040%），但训练时间 31ms；DQRM 22ms。** 这个差距完全来自"只量化用到的权重 + 周期更新 scale"。在 Terabyte 上差距更悬殊：PACT 与 HAWQ **直接跑不完（>1000 ms/it）**，LSQ 42ms，DQRM 29ms。

## 为什么值得读

DQRM 在 Track B 的量化支线里有三个独特价值：

**① 它是"统一低精度"路线的代表，且发现了量化的正则副作用。**
B-10（MPE）走混合精度（每个组不同位宽），追求"在不掉精度的前提下最大化压缩"；DQRM 走统一 INT4，结果**精度反而涨了**。两者对"量化是什么"的理解不同：
- MPE：量化 = 有损压缩，要小心控制损失；
- DQRM：量化 = **强正则**，损失本身有益。

这个分歧有实证支撑（Table 2 的单调趋势：INT16→INT8→INT4 精度逐级上升），不是修辞。**在你会遇到多 epoch 训练的场景下，这个发现可能比压缩比更重要。**

**② 它明确区分了"训练加速"和"通信量降低"，并诚实地承认后者没兑现。**
Table 4 的脚注 3 说得很直白：延迟涨了，因为 PyTorch 的低精度稀疏 all-reduce 不支持。这种诚实在系统论文里不常见。读它可以学会一件事：**通信量 ↓ 4× ≠ 训练时间 ↓ 4×**。

**③ 与 B-09（HPS）构成"压缩 vs 层级存储"的完整辩论。**
HPS（百度，2020）在 §2 用真实生产数据论证"哈希压缩必然掉精度，所以我们做层级存储"。DQRM（Berkeley/Intel，2024）证明"INT4 量化不掉精度，还涨"。**后者直接动摇了前者的前提**。不过要注意两者的数据集不同（百度 10¹¹ 特征的广告主模型 vs Criteo 的 2600 万特征），规模差 4 个数量级——所以更准确的说法是：**压缩派在中等规模上已证明可行，在超大规模上仍未被证明**（**推断**）。

**④ 与 B-16（Pinterest）呼应同一现象（过拟合），但给的解法完全不同。**
- Pinterest：**频率自适应学习率**（FAL）+ 嵌入重初始化（MEDA），在**优化器层面**治过拟合；
- DQRM：**INT4 量化**，在**表示精度层面**治过拟合。

Pinterest 的结论是"若有过量新数据，多 epoch 过拟合不必治"；DQRM 的结论是"低精度本身就是治疗"。**两者可以叠，且 DQRM 的方案更便宜**（不需要额外状态）。这是一个值得注意的交叉点。

## 局限

- **"精度不降反升"依赖过拟合这个前提**。如果模型本身没过拟合（比如 Pinterest 那个 25% 更大数据集的情形，过拟合很轻），INT4 的正则收益就不存在，甚至可能变成损伤。作者在附录 A.6 测了"训练很多 epoch"，但没有测"本来就不过拟合"的场景。
- **只验证了 Criteo 的 Kaggle 与 Terabyte**。两个都是 2600 万特征级别的中等规模；10¹¹ 特征的工业模型上 INT4 是否还无损，**待核实**。
- **分布式实验的结论是负面的**。通信量降了 4×，但延迟涨了近 2×。所以要真正收益，还需要"低精度稀疏 all-reduce"的系统层工作——这部分论文明确留给未来。
- **边缘部署只在附录 C 提了一句**（"Demo Recommender on the Phone"），没有量化指标。所以"能上手机"这个卖点，证据很弱（**待核实**）。
- **per-table 量化的粒度问题**。一张表的所有行共享一组 scale，如果表内的权重范围差异大（长尾特征常见），低位宽会截断大值。作者说 per-row 开销大所以不做，但**没给 per-table 的相对精度损失数据**（**推断**：Table 2 的 INT4 结果是在 Criteo 的 26 张表上得到的，每张表内部比较同质）。
- **周期性更新 scale 的周期是手调的**（200 / 1000）。这个参数与 batch size、学习率、数据分布都有关，论文没给选取原则。
- **INT2 也测了**（附录），结论是"也治过拟合，但精度掉得更多"——说明这个方法有个下限。

## 证据与待核实

证据等级：**fulltext**。取到正文的 URL：`https://arxiv.org/html/2410.20046v1`（含 §1–§5、Figure 1–3、Table 1–4、附录 A 的部分 ablation 描述）。

事实清单：8× 压缩、Kaggle 79.071%/0.8073、Terabyte 81.210%、scale 时间占比 97%、Table 2 的单调趋势、Table 4 的通信数据与延迟涨幅、per-table/channel-wise 的粒度选择、只给 MLP 做误差补偿、N=200/1000。

待核实：
1. 附录 C 的边缘部署具体做了什么（只有一句标题）；
2. 附录 A.6 "训练很多 epoch" 的具体结果（摘要提到 INT4 收敛能持续 16 epoch）；
3. 附录 A.2 梯度量化位宽的完整扫描；
4. per-table vs per-row 量化的精度差（论文说 per-row 开销大，未报精度对比）；
5. 在不过拟合的设置下 INT4 是否仍无损。

## 是否值得读原文

**必读**——理由不是压缩比（8× 在 Track B 里不算高），而是**两个洞察**：
1. **QAT 在嵌入表上的真正瓶颈是"遍历全表找 scale"，不是算力**。§3.1–3.2 的两个技巧（只量化用到的、周期更新 scale）把 CPU 上 97% 的耗时抹掉了，这个诊断过程值得学。
2. **低精度可以当正则用**。Table 2 的单调趋势配上 Figure 3 的过拟合曲线，是"压缩改善泛化"最清晰的实证。

如果你只关心分布式训练，读 §3.3 与 Table 4——那里有一个诚实的负面结果（通信降 4×、延迟涨 2×），比正面结果更有信息量。
