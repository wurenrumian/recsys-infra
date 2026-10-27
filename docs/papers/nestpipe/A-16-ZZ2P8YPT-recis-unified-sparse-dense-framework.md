---
title: "RecIS: Sparse to Dense, A Unified Training Framework for Recommendation Models"
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-A, pytorch-framework, unified-sparse-dense, bandwidth-roofline, generative-recommendation]
related: [docs/papers/nestpipe/index.md]
zotero_key: ZZ2P8YPT
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Hua Zong、Qingtao Zeng、Zhengxiong Zhou、Zhihua Han、Zhensong Yan、Mingjie Liu、Hechen Sun、Jiawei Liu、Yiwen Hu、Qi Wang、YiHan Xian、Wenjie Guo、Houyuan Xiang、Zhiyuan Zeng、Xiangrong Sheng、Bencheng Yan、Nan Hu、Yuheng Huang、Jinqing Lian、Ziru Xu、Yan Zhang、Ju Huang、Siran Yang、Huimin Yi、Jiamang Wang、Pengjie Wang、Han Zhu、Jian Wu、Dan Ou、Jian Xu、Haihong Tang、Yuning Jiang、Bo Zheng、Lin Qu（**34 人，阿里巴巴**，**事实**：作者群与摘要均指 Alibaba）。
- 年份：2025 年 9 月（arXiv 预印本，`2509.20883`），**Zotero 中无 DOI、无会议字段**（**事实**：条目类型 preprint，Extra 仅 arXiv 编号）。
- 公开全文：`https://arxiv.org/abs/2509.20883`，正文取到 `https://arxiv.org/html/2509.20883`（**事实**）。
- 系统名：**RecIS**（Rec-ommendation I-ntelligent S-ystem）。

## 一句话

把 TensorFlow 那套稀疏训练搬到 PyTorch 上，用带宽 roofline 指导优化，稀疏稠密统一训。

## 背景与问题

RecIS 的问题不是"某个瓶颈"，而是**一个生态断层**（**事实**：第 1.3 节，讲得非常直白）：

| | 强在哪 | 弱在哪 |
|---|---|---|
| **TensorFlow** | 大规模嵌入表、分布式训练、生产稳定性成熟 | **静态图**，迭代慢，新稠密架构（大模型）实验困难 |
| **PyTorch** | 动态图、生态丰富、与大模型/多模态无缝集成 | **缺乏生产级的大规模稀疏建模原生支持** |

而现实是（**事实**）"几乎所有前沿大模型的首个官方或社区实现都基于 PyTorch"，所以 PyTorch 成了大模型的事实平台。但工业推荐系统的稀疏那一半还锁在 TensorFlow 里。

作者的判断很明确：**必须基于 PyTorch 建统一框架**，把稀疏组件从 TensorFlow 侧"移植"过来。

这个决定带来两类挑战（**事实**）：

- **建模挑战**：PyTorch 缺稀疏建模的原生支持；
- **系统挑战**：大模型这边 PyTorch 生态的 AI 基础设施很完整（高效计算、分布式），但**稀疏组件引入了新问题——IO 瓶颈、并发灾难（concurrency disasters）、内存带宽限制**。

## 核心做法

### 一、方法论：从算力 roofline 换成带宽 roofline

这是 RecIS 最有思想价值的一节（**事实**，第 1.4.2 节）。

标准 roofline 模型（Williams et al. 2009）的 x 轴是**算术强度（Arithmetic Intensity, AI）**，y 轴是算力，用来衡量 **MFU（Model FLOPS Utilization）**，适合**计算密集**负载（稠密层、LLM）。

但稀疏组件是典型的**内存带宽受限**：**unique、embedding lookup、embedding reduce 这些算子的 AI 都低于 1**，落在 roofline 图 x 轴最左边——**算力 roofline 对它们毫无解释力**。

于是作者提出**带宽 roofline**：

| | 算力 roofline | 带宽 roofline |
|---|---|---|
| x 轴 | 算术强度（Arithmetic Intensity） | **带宽强度（Bandwidth Intensity）** |
| y 轴 | 算力（FLOPS） | **带宽（Bandwidth）** |
| 核心指标 | MFU | **MBU（Model Bandwidth Utilization）** |
| 适用 | 稠密/计算密集 | **稀疏/内存带宽受限** |

作者的论断是：在这个模型里，稀疏算子因为**带宽强度高**，理论上全部落在 roofline 上——**说明"内存受限"就是它们的优化上限**。所以应该用这个模型来评估与优化稀疏组件。

**MBU 被提议为推荐系统的第一等指标，地位类比 LLM 里的 MFU**（**事实**：贡献 2 明确这么说）。

### 二、三堵墙 + 对应策略

框架的优化组织成"破三堵墙"（**事实**，第 1.4.2 节 + 第 2.2 节）：

```
        ┌──────────┐   ┌──────────┐   ┌──────────────┐
        │  IO 墙    │ → │ 内存墙    │ → │  计算墙       │
        │(样本读取) │   │(稀疏,带宽)│   │(稠密,算力)    │
        └──────────┘   └──────────┘   └──────────────┘
             │              │                │
        IO 优化          MBU 优化         MFU 优化
     (列存/多线程/     (全上 GPU/       (混合精度/
      CSR/GPU 批处理)   两层存储)        Fused kernel/ZeRO)
```

**（1）破 IO 墙**（**事实**，2.2.1 节）

推荐系统高度 IO 密集——要适应样本分布变化，必须持续用新样本训练，所以数据获取与预处理必须有极高 IO 效率。"只有 IO 充分优化，瓶颈才会前移到计算部分。"

五个具体优化：

- **列存格式（Columnar Format）**：按列存。好处有三——组装 batch 时**省掉从行拷贝的开销**；存储后端**零成本选列**（满足特征过滤）；**压缩率更好**（省容量、缓解网络带宽）。
- **高并发 + 异步**：多线程读分片数据。对比 PyTorch 基于 Python multiprocessing 的 DataLoader，**C++ 多线程性能更高、对其它 GPU kernel 启动的干扰更小**。异步设计让 IO 与训练计算重叠。
- **CSR 内存布局**：对比 TensorFlow 的 SparseTensor（COO），**RaggedTensor（CSR）对超长序列特征更高效**。
- **GPU 批处理**：极端 IO 需求时，用 GPU 做 batch 组装——充分利用 GPU 内存带宽，CPU→GPU 拷贝可以多线程预执行。

**（2）破内存墙**（**事实**，2.2.2 节）

核心动作是**把一切搬到 GPU**。作者对"旧架构"的批判很直接：

> 过去把内存密集算子放 CPU（为了省 HBM、为了异构并行）。**但这个架构在当前的硬件趋势下已经过时了**。单机多卡（8 卡以上）成为主流后，**GPU 的内存带宽比 CPU 高两个数量级**，同时 GPU HBM 容量还在涨。所以要充分利用新硬件的潜力，必须把高内存需求的任务全迁到 GPU。

**具体实现是一个两层存储架构**（**事实**）：

```
        ┌─────────────────────────────────────┐
        │  IDMap（第一层）                     │
        │  key = feature ID                    │
        │  value = offset                      │
        └──────────────┬──────────────────────┘
                       │ 一次 IDMap 查找 → 拿到 offset
                       ▼
        ┌─────────────────────────────────────┐
        │  Blocks（第二层）                    │
        │  连续的分片内存块                     │
        │  存嵌入参数（float）+ 优化器状态       │
        │  支持动态分片以灵活扩容                │
        └─────────────────────────────────────┘

前向：IDMap lookup → 拿到 offset → 直接从 Blocks 读向量
反向：用梯度 + 保留的前向 offset → 直接更新 Blocks 里的优化器状态与参数值
```

这个设计的巧妙处在于**反向不需要再查一次 IDMap**——前向拿到的 offset 直接复用。而且它是**key-value 语义**，所以支持"驱逐陈旧特征"（连续训练场景必需）。

**弹性**：HBM 够用时 IDMap 与 Blocks 全放 HBM；HBM 紧张时可把其中之一**灵活 offload 到 CPU 内存**。这保留了框架对不同硬件的适应性（**事实**）。

**负载均衡**：阿里电商场景里模型常有**上千个特征列**，每列对应稀疏参数，要**均匀分片**以避免热点（**事实**：小节标题与开头，具体算法**待核实**）。

**（3）破计算墙**（**事实**，2.2.3 节）

稠密部分直接复用 PyTorch 生态的成熟优化：

- **混合精度**：稀疏部分用 **FP32**，稠密注意力部分用 **FP16/BF16**；
- **Fused kernel**：FlashAttention、FusedSoftmaxCrossEntropy；
- **ZeRO**：把稠密模型状态（参数、梯度、优化器状态）分片到多卡，降低单卡 HBM 占用。

### 三、框架组件（**事实**，2.1 节）

| 组件 | 职责 | 关键细节 |
|---|---|---|
| **ColumnIO** | 分片读多列样本、组装 PyTorch Tensor | 支持嵌套 list（多值/序列特征）、按列聚合特征；数据源含**天级静态表**与**秒级实时流表**；支持 point-lookup join；生产环境直读阿里云 DFS / MaxCompute |
| **Feature Engine** | 训练期特征预处理 | hash 变换、数值分桶、序列截断/补齐、跨列特征交叉 |
| **Embedding Engine** | 可扩展、无冲突的 KV 嵌入表 | 连续训练时可驱逐陈旧特征；稀疏存储的高效计算与优化器更新 |
| **Saver** | 稀疏参数分片并行 Load/Save | **SafeTensors**，同一格式交付线上推理 |
| **Optimizer** | 嵌入优化器 | **SparseAdam / SparseAdamW**，与 TF 兼容 |
| **Pipelines** | 串成训练工作流 | 多阶段（训/测交错）、窗口式在线学习、多任务 |

**"无冲突嵌入（conflict-free embedding）"是 RecIS 相对 TorchRec 的核心差异化卖点**（**事实**：贡献 1 原文对比 TorchRec 时强调 "conflict-free embedding, efficient IO, and sparse processing"）。具体指什么机制**待核实**——推测是指 ID 到物理位置的映射不会因哈希碰撞而互相覆盖（**推断**）。

### 四、向后兼容

一个很实际的工程决定（**事实**）：**RecIS 能加载 TensorFlow 的 checkpoint 与优化器状态**，让迁移平滑——老系统训的权重与超参可以在新框架里对齐续训。

## 关键数字

### 算子级 MBU（**事实**，Table 1，H20 上）

| 算子 | TensorFlow | PyTorch | RecIS | 提升倍数（vs PT） |
|---|---|---|---|---|
| bucketize | 0.40% | 0.40% | **0.88%** | 2.2× |
| mod | 0.45% | 0.70% | **1.68%** | 2.4× |
| ids partition | — | 34.60% | **55.10%** | 1.6× |
| sequence tile | 2.43% | 4.58% | **18.25%** | 4.0× |
| reduce hard | 0.48% | 0.93% | **2.25%** | 2.4× |
| reduce easy | 1.38% | 2.75% | **13.75%** | **5.0×** |
| gather | 1.70% | 15.00% | **47.50%** | **3.2×** |
| scatter | — | 20.75% | **58.00%** | **2.8×** |

注意这些 MBU 的绝对值仍然很低（最高 58%），说明**稀疏算子离带宽 roofline 还有距离**——这也是为什么作者把它当作"SOTA 指标"来跟踪。同时注意 **bucketize/mod 只有 0.88%/1.68%**，论文自己承认："虽然融合提升了带宽利用，**实际带宽利用仍然很低**，因为融合 100 列特征带来了额外计算，这是 RecIS 未来要解决的问题"（**事实**，2.2.1 节的诚实自述）。

### 端到端（**事实**，3.2 节，32 / 64 H20 GPU）

| 模型 | 系统 | sparse | overall |
|---|---|---|---|
| MSE | TensorFlow | 713ms | 1042ms |
| MSE | PyTorch (TorchRec) | 303ms | 424ms |
| MSE | **RecIS** | **218ms** | **337ms** |
| LMA(16k) | TensorFlow | 1137ms | 1374ms |
| LMA(16k) | **RecIS** | **764ms** | **1050ms** |
| LMA(100k) | TensorFlow | **跑不起来** | — |
| LMA(100k) | **RecIS** | **259ms** | **588ms** |

换算：MSE 上 RecIS 的 overall 是 PyTorch 的 **79%**、TensorFlow 的 **32%**；稀疏部分时间是 PyTorch 的 **72%**、TensorFlow 的 **30%**（**事实**）。LMA(16k) 的 overall 是 TensorFlow 的 **76%**，稀疏部分是 **67%**（**事实**）。

### 三个规模化应用（**事实**，第 4 节，这部分是本 track 独有的"规模实证"）

| 维度 | 结果 |
|---|---|
| **稠密参数缩放**（生成式排序） | 相比 TorchRec，**单节点最大 batch size 提升 200%**，**总训练时间降 70%**；多机多卡近线性扩展；5000 万稠密参数的生成式排序模型已在主 feed 做 A/B，CTR 与转化**双升** |
| **用户序列缩放** | 生产环境部署了**长度 1M** 的行为序列模型（相对此前标准 **100 倍**提升）；相比 10k 序列长度，**CTR +4.8%**，训练成本**降 50%** |
| **模态缩放** | 用 target-aware embeddings 做知识迁移，多个工业场景离线与线上 A/B 均有稳定提升 |

序列缩放那节的工程数字特别具体（**事实**）：单 batch 可达 **6 GB**，假设 2 秒训完，则网络吞吐要 **24 GB/s**，峰值内存访问带宽 **100–200 GB/s**——"CX7 级网卡与 16 通道 DDR 都吃不下"，所以用 **ORC 压缩 + GPU 样本组装**把瓶颈从 IO 带宽搬走。嵌入侧单 batch 可达 **10 亿次 KV 查询、1 TB 嵌入内存访问**，用 **GPU Unique + GPU Hashtable** 各优化 10×。

### 总括

摘要与结论的官方口径是 **"up to 2× 训练吞吐"**（**事实**：贡献 3 与结论都这么说）。但第 4 节里出现了 200% batch size 提升、70% 训练时间降低（相对 TorchRec）、CTR +4.8% 等更强的数字——**口径不同，不能混用**（**推断**：2× 应该指框架层面的端到端吞吐，第 4 节是特定应用场景）。

## 为什么值得读

A-16 是 Track A 里**唯一一篇 2025 年的一手框架论文**，也是唯一一篇**真正把"推荐训练框架"的完整形态（IO + 稀疏 + 稠密 + 序列 + 模态）都讲全**的。

与同 track 的关系：

- **它是 A-08（TorchRec）的明确挑战者**。RecIS 在三个地方直接批评 TorchRec（**事实**）：
  1. **无冲突嵌入**：TorchRec 不支持（LMA 模型因此无法在 TorchRec 上跑——论文说"我们只在 TensorFlow 和 RecIS 上部署 LMA，因为这个模型需要 TorchRec 目前不支持的零冲突嵌入特征"）；
  2. **IO 与多机扩展性**：RecIS 说 TorchRec "在 IO 与多机扩展性上有性能退化"；
  3. **闭源与迭代速度**："closed architecture and slow iteration and open-source cycle"。
  
  **这是全 track 里对 TorchRec 最直接的负面评价**，值得记下来。
- **它是 A-09（PICASSO）、A-12（Hotline）、A-04（Persia）的技术路线的终结者**。这三篇都在"CPU 与 GPU 分工"上做文章，RecIS 一句话否掉："**这个架构在当前的硬件趋势下已经过时了**"——因为 GPU 带宽已经比 CPU 高两个数量级，HBM 容量也在涨。**从 PICASSO（2022，TF + 混合）到 RecIS（2025，PyTorch + 全上 GPU），是整条 track 最大的一次范式切换。**
- **它是 A-11（ScratchPipe）与 A-10（Bagpipe）的"下一个问题"**。那两篇都在"嵌入表放不下 GPU 所以缓存/预取"，RecIS 说"HBM 现在够用了"。当然这只对中大规模模型成立，TB 级模型仍需缓存——**这条分界线在哪，是本 track 最值得追问的量化问题**（**推断**）。
- **它继承了 A-09 的"D-Packing"思路但做得更彻底**：MSE 模型里 **600+ 特征变换算子融合成 3 个**（**事实**：3.2.1 节），而 PICASSO 的最好成绩是 18 → 4。**同样是"打包"，RecIS 的规模大了一个数量级。**
- **它的带宽 roofline / MBU 概念是引入 Track A 的唯一一个新评估范式**。其它 16 篇要么报加速比、要么报 QPS、要么报 NE，只有 RecIS 提出"应该用什么指标衡量稀疏系统"。

最值得学的其实是**那种"用硬件趋势否掉旧架构"的论证方式**：它不谈优化技巧，直接指出"GPU 带宽比 CPU 高两个数量级 + 单机 8 卡成主流"这两个事实，就足以判 A-04/A-09/A-12 那批"CPU 分担"方案死刑。**这种"用硬件曲线重新审视架构选择"的方法，比任何单个技术点都有生命力。**

## 局限

- **是预印本（arXiv 2509.20883），无同行评审**（**事实**：Zotero 类型 preprint，无 DOI、无会议）。34 位作者的工业报告风格明显，技术深度不均匀。
- **"up to 2×" 的口径模糊**。摘要与结论都说 2×，但不知道相对谁（TF？TorchRec？自己的旧版本？），也不知道在什么模型上（**待核实**）。
- **有些数字强得可疑**。"单节点最大 batch size 提升 200%、总训练时间降 70%"（vs TorchRec）——如果这是真的，TorchRec 早就该被全面替换了。可能的解释是特定模型（生成式排序 + 大稠密参数）下 TorchRec 的 IO 确实很弱（**推断**）。
- **MBU 的绝对值证明优化还很初级**。最高 58%（scatter）、47.5%（gather），bucketize/mod 只有 0.88%/1.68%。作者自己承认融合带来额外计算、实际带宽利用仍低（**事实**）。所以"MBU 是 SOTA 指标"这个提法成立，但"RecIS 的 MBU 已经很好了"不成立（**推断**）。
- **缺少对核心机制的技术说明**。"无冲突嵌入"（conflict-free embedding）是相对 TorchRec 的关键差异，但全文只在贡献列表与 LMA 的部分提到"TorchRec 不支持"，**没有讲它到底怎么实现**（**待核实**）。这是本文最大的技术空洞。
- **负载均衡算法未给**。第 2.2.2 节说"上千个特征列必须均匀分片以避免热点"，但算法细节在可读文本中没有（**待核实**）。
- **强烈绑定阿里内部基础设施**。ColumnIO 直连阿里云 DFS / MaxCompute / tunnel，生产数字全来自阿里内部场景。开源版本（RecIS 有开源部分）能否复现这些数字，未知（**推断**）。
- **与 TensorFlow 的对比不公平**。TF 用的是 1.x（LMA 那节说"tensor shape 约束在 TensorFlow 1.x"），而 RecIS 是 2025 年的 PyTorch 2.x 栈。跨代对比（**推断**）。
- **"破三堵墙"的框架很像 marketing**。IO / 内存 / 计算三堵墙的划分本身没问题，但把每个优化都塞进"破某堵墙"的框里，实际上掩盖了各优化的贡献分解——**没有消融实验**（**事实**：全文无 ablation 表）。

## 证据与待核实

- 证据等级：**fulltext**。正文取自 `https://arxiv.org/html/2509.20883`（完整 arXiv 版），章节完整（1 Introduction / 2 System Design / 3 Experiment / 4 Application / 5 Related Work / 6 Conclusion），含 Table 1、Table 2、Figure 1–5 的描述。
- **事实**：34 位作者与阿里、arXiv 编号、TF vs PyTorch 的生态断层、三类稀疏系统挑战、带宽 roofline 与 MBU 的定义与动机、三堵墙与对应策略、五个 IO 优化（列存/多线程异步/CSR/GPU 批处理）、"旧异构架构已过时"的论证（GPU 带宽高两个数量级）、IDMap + Blocks 两层存储与前后向流程、按需 offload 的弹性、混合精度/Fused kernel/ZeRO、六个框架组件与各自细节、SafeTensors、SparseAdam(W)、TF checkpoint 兼容、Table 1 全部 MBU、Table 2 全部端到端时间、600+→3 算子融合、第 4 节三个规模化应用的全部数字（200%/70%/5000万/1M 序列/100倍/+4.8%/50%/6GB/24GB/s/1B KV/1TB/10×）、对 TorchRec 的三条批评、LMA 因 TorchRec 不支持零冲突嵌入而无法部署。
- **待核实**：
  1. "conflict-free embedding" 的具体实现（本文最关键的技术空洞）；
  2. 稀疏参数的负载均衡算法；
  3. "up to 2×" 的基线定义；
  4. 带宽 roofline 中"带宽强度"的精确定义与 MBU 的计算公式；
  5. LMA / MSE 两个模型的完整配置（LMA 结构部分给了，MSE 的 660 特征给了，但 embedding 表规格没给）；
  6. 是否有正式发表版本；
  7. 开源代码的范围（RecIS 是否完整开源，还是只有部分）；
  8. 硬件配置里 H20 的具体规格与集群规模（32 / 64 卡）。
- Zotero 内无 PDF 附件（集合内 50 条均无，属预期）。

## 是否值得读原文

**必读**——这是 Track A 里最接近"当代工业推荐训练框架完整形态"的一篇，而且它是唯一一篇**给出了序列长度 1M、稠密参数 5000 万这类真实规模实证**的。重点读第 1.4.2 节（带宽 roofline / MBU，这个范式值得抄）与第 2.2 节（三堵墙的具体做法，尤其 IDMap + Blocks 的 offset 复用设计）。第 4 节（三个规模化应用）是本 track 里最接近"真实生产系统怎么长成"的材料，即使不做系统也该读。**第 3.1 节的 MBU 表要看原始数字**——它诚实地显示稀疏算子的利用率仍然很低，这比任何加速比都更能说明这个领域还在早期。
