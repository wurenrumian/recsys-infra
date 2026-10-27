---
title: NestPipe 论文集合导读
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, recsys-infra, embedding-table, distributed-training, generative-recsys, pipeline-parallelism]
related: [docs/papers/nestpipe/index.md]
zotero_collection: 9I82QPTL
papers: 50
---

# NestPipe 论文集合导读

NestPipe 是 Zotero 中的一组推荐系统基建论文（collection key `9I82QPTL`，50 篇）。这份导读回答两个问题：
**这 50 篇在共同解决什么问题**，以及**要按什么顺序读**。每篇的独立解读在 `papers/` 子目录下。

## 为什么这 50 篇会被放在一起

推荐模型的参数构成和一般深度学习模型不一样：**99% 的参数在嵌入表里，而这些参数是稀疏访问的**。
这带来一个连锁后果——模型变大时，最先撑不住的不是算力，而是内存层级、通信带宽和访存局部性。
这 50 篇可以看成同一条战线上的 50 次尝试：**每一篇都动了系统里的一层，并为此付出了一层代价**。

三种压力源对应三个分组：

| 压力源 | 具体表现 | 分组 |
|---|---|---|
| 算力与并行 | 单卡放不下、单机放不下、通信打满 | **A** 规模化训练系统与并行架构 |
| 内存与存储 | 嵌入表到 TB 级，HBM 装不下、DRAM 装不下 | **B** 嵌入表存储、缓存、压缩与分片 |
| 模型形态 | 从 DLRM 走向序列 Transformer、生成式、MoE，结构本身变重 | **C** 模型演进、生成式推荐与流水线调度 |

> 这个分组是人为的策展划分，不是论文自带的分区。一篇论文同时属于两三个组是常态，
> 分组只用来缩小阅读范围，不表示边界。

## 覆盖情况

| 项 | 数值 |
|---|---|
| 论文总数 | 50 |
| 取到正文 | 33 |
| 仅有摘要与官方页 | 17 |
| 完全未核实 | 0 |
| 建议精读 / 按章读 / 只需结论 | 25 / 18 / 7 |

按分组：A 组 17 篇（12 篇有正文），B 组 17 篇（7 篇有正文），C 组 16 篇（14 篇有正文）。

**B 组正文比例最低，原因是访问受限，不是检索不足**：这一组的发表场所集中在 ACM DL 与 USENIX，
ACM DL 对自动化访问返回 403，USENIX 的论文以 PDF 形式提供而当前取文通道不解析 PDF，
同时这批论文多数没有 arXiv 版本。这 10 篇的机制细节在文中都标为**待核实**，
没有把摘要内容外推成事实。

## 三个分组

每篇的标题链接指向独立解读页。`body` 列表示解析深度，`read` 列是阅读建议。

### A · 规模化训练系统与并行架构

| # | 论文 | body | read |
|---|---|---|---|
| A-01 | [Scaling Law of Large Sequential Recommendation Models](A-01-RJYMAGIN-scaling-law-sequential-recsys.md) | 正文 | 按章读 |
| A-02 | [Climber: Toward Efficient Scaling Laws for Large Recommendation Models](A-02-ABZSE7KJ-climber-efficient-scaling-laws.md) | 正文 | 精读 |
| A-03 | [Two-dimensional Sparse Parallelism for Large Scale Deep Learning Recommendation Model Training](A-03-DZZH6L95-two-dimensional-sparse-parallelism.md) | 正文 | 精读 |
| A-04 | [Persia: An Open, Hybrid System Scaling Deep Learning-based Recommenders up to 100 Trillion Parameters](A-04-892HASSZ-persia-hybrid-100t-params.md) | 正文 | 精读 |
| A-05 | [Hierarchical Training: Scaling Deep Recommendation Models on Large CPU Clusters](A-05-HP588YPM-hierarchical-training-cpu-clusters.md) | 摘要 | 按章读 |
| A-06 | [Software-Hardware Co-design for Fast and Scalable Training of Deep Learning Recommendation Models](A-06-J9ZPBVH5-sw-hw-codesign-neo-4d-parallelism.md) | 正文 | 精读 |
| A-07 | [Merlin HugeCTR: GPU-accelerated Recommender System Training and Inference](A-07-B2I9ZAIQ-merlin-hugetr-gpu-training-inference.md) | 正文 | 只需结论 |
| A-08 | [TorchRec: a PyTorch Domain Library for Recommendation Systems](A-08-4GP3GTPM-torchrec-pytorch-domain-library.md) | 摘要 | 只需结论 |
| A-09 | [PICASSO: Unleashing the Potential of GPU-centric Training for Wide-and-deep Recommender Systems](A-09-XGVN6JZA-picasso-gpu-centric-wdl-training.md) | 正文 | 精读 |
| A-10 | [Bagpipe: Accelerating Deep Recommendation Model Training](A-10-YYFDGRXK-bagpipe-accelerating-dlrm-training.md) | 正文 | 精读 |
| A-11 | [Training Personalized Recommendation Systems from (GPU) Scratch: Look Forward not Backwards](A-11-GYXLW7PP-scratchpipe-look-forward-not-backwards.md) | 正文 | 精读 |
| A-12 | [Heterogeneous Acceleration Pipeline for Recommendation System Training (Hotline)](A-12-9DTGQILE-hotline-heterogeneous-acceleration-pipeline.md) | 正文 | 按章读 |
| A-13 | [Accelerating Distributed DLRM Training with Optimized TT Decomposition and Micro-Batching (EcoRec)](A-13-TFTSRALD-ecorec-tt-decomposition-micro-batching.md) | 摘要 | 只需结论 |
| A-14 | [Accelerating Communication in Deep Learning Recommendation Model Training with Dual-Level Adaptive Lossy Compression](A-14-S996SB9M-dual-level-adaptive-lossy-compression.md) | 正文 | 精读 |
| A-15 | [FusedRec: Fused Embedding Communication for Distributed Recommendation Training on GPUs](A-15-SZH9CX5W-fusedrec-fused-embedding-communication.md) | 摘要 | 按章读 |
| A-16 | [RecIS: Sparse to Dense, A Unified Training Framework for Recommendation Models](A-16-ZZ2P8YPT-recis-unified-sparse-dense-framework.md) | 正文 | 精读 |
| A-17 | [Research on Model Parallelism and Data Parallelism Optimization Methods in Large Language Model-Based Recommendation Systems](A-17-3FPR4H9A-llm-recsys-parallelism-optimization.md) | 摘要 | 只需结论 |

### B · 嵌入表存储、缓存、压缩与分片

| # | 论文 | body | read |
|---|---|---|---|
| B-01 | [EVStore: Storage and Caching Capabilities for Scaling Embedding Tables in Deep Recommendation Systems](B-01-KP82AUTM-evstore-embedding-storage-cache.md) | 摘要 | 按章读 |
| B-02 | [Efficient and scalable huge embedding model training via distributed cache management](B-02-9XB4UUDV-het-distributed-embedding-cache.md) | 正文 | 精读 |
| B-03 | [CAFE+: Towards Compact, Adaptive, and Fast Embedding for Large-scale Online Recommendation Models](B-03-WBL86RZE-cafe-plus-compact-adaptive-embedding.md) | 摘要 | 按章读 |
| B-04 | [Embedding Optimization for Training Large-scale Deep Learning Recommendation Systems with EMBark](B-04-JXSQDUAF-embark-embedding-clusters-3d-sharding.md) | 摘要 | 按章读 |
| B-05 | [AutoShard: Automated Embedding Table Sharding for Recommender Systems](B-05-J7EYKA8M-autoshard-rl-embedding-sharding.md) | 正文 | 按章读 |
| B-06 | [Pre-train and Search: Efficient Embedding Table Sharding with Pre-trained Neural Cost Models](B-06-WXMZVWVJ-neuroshard-pretrain-and-search.md) | 正文 | 精读 |
| B-07 | [EL-Rec: Efficient Large-Scale Recommendation Model Training via Tensor-Train Embedding Table](B-07-UZF6ZILW-elrec-tensor-train-embedding.md) | 摘要 | 按章读 |
| B-08 | [Unified and Near-optimal Multi-GPU Cache for Embedding-based Deep Learning](B-08-6NY24IFU-ugache-unified-multi-gpu-cache.md) | 摘要 | 按章读 |
| B-09 | [Distributed Hierarchical GPU Parameter Server for Massive Scale Deep Learning Ads Systems](B-09-SLPPGD8C-hierarchical-gpu-parameter-server.md) | 正文 | 精读 |
| B-10 | [Mixed-Precision Embeddings for Large-Scale Recommendation Models](B-10-DH7ZFV2C-mpe-mixed-precision-embeddings.md) | 正文 | 精读 |
| B-11 | [DQRM: Deep Quantized Recommendation Models](B-11-Z56XJYUC-dqrm-deep-quantized-recommendation.md) | 正文 | 精读 |
| B-12 | [AdaEmbed: Adaptive Embedding for Large-Scale Recommendation Models](B-12-3K9KISR6-adaembed-adaptive-embedding-pruning.md) | 摘要 | 按章读 |
| B-13 | [OPER: Optimality-Guided Embedding Table Parallelization for Large-scale Recommendation Model](B-13-37Q3PZXS-oper-optimality-guided-parallelization.md) | 摘要 | 精读 |
| B-14 | [Accelerating Neural Recommendation Training with Embedding Scheduling](B-14-6HTV2P4W-herald-embedding-scheduling.md) | 摘要 | 精读 |
| B-15 | [HypeReca: Distributed Heterogeneous In-Memory Embedding Database for Training Recommender Models](B-15-VE9T6R4E-hypereca-in-memory-embedding-database.md) | 摘要 | 精读 |
| B-16 | [The Evolution of Embedding Table Optimization and Multi-Epoch Training in Pinterest Ads Conversion](B-16-L2VVL9ZT-pinterest-embedding-multi-epoch.md) | 正文 | 精读 |
| B-17 | [DCMA: Accelerating Parallel DMA Transfers with a Multi-Port Direct Cached Memory Access in a Massive-Parallel Vector Processor](B-17-WCV8CKED-dcma-multi-port-cached-dma.md) | 摘要 | 只需结论 |

### C · 模型演进、生成式推荐与流水线调度

| # | 论文 | body | read |
|---|---|---|---|
| C-01 | [Generative Recommendation Models - Progress and Directions](C-01-GMPG7G9I-generative-recsys-progress-directions.md) | 摘要 | 只需结论 |
| C-02 | [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](C-02-G73SFXWD-hstu-trillion-param-generative-recsys.md) | 正文 | 精读 |
| C-03 | [MTGR - Industrial-Scale Generative Recommendation Framework in Meituan](C-03-DDWBU7H3-mtgr-meituan-generative-recsys.md) | 正文 | 精读 |
| C-04 | [FuXi-α - Scaling Recommendation Model with Feature Interaction Enhanced Transformer](C-04-AXNSLSLL-fuxi-alpha-feature-interaction-transformer.md) | 正文 | 按章读 |
| C-05 | [Bending the Scaling Law Curve in Large-Scale Recommendation Systems](C-05-ASKWXRVH-ultra-hstu-bending-scaling-law.md) | 正文 | 精读 |
| C-06 | [Disaggregated Multi-Tower - Topology-aware Modeling Technique for Efficient Large-Scale Recommendation](C-06-NF8E5IDY-dmt-disaggregated-multi-tower.md) | 正文 | 按章读 |
| C-07 | [KuaiRand - An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos](C-07-FCJDT7DA-kuairand-unbiased-dataset.md) | 正文 | 按章读 |
| C-08 | [SlimPipe - Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training](C-08-9MCTBGR4-slimpipe-memory-thrifty-pp.md) | 正文 | 精读 |
| C-09 | [TD-Pipe - Temporally-Disaggregated Pipeline Parallelism Architecture for High-Throughput LLM Inference](C-09-2JX9YAFY-td-pipe-temporal-disaggregated-pp.md) | 正文 | 按章读 |
| C-10 | [OctoPipe - Reducing Pipeline Bubbles for Heterogeneous Models via Co-Optimizing Partitioning, Placement, and Scheduling](C-10-UFFU6KVA-octopipe-heterogeneous-bubbles.md) | 正文 | 按章读 |
| C-11 | [Mitigating Staleness in Asynchronous Pipeline Parallelism via Basis Rotation](C-11-W5IVJUAZ-basis-rotation-staleness.md) | 正文 | 精读 |
| C-12 | [Revisiting Parameter Server in LLM Post-Training](C-12-ZLR452K5-odc-revisiting-parameter-server.md) | 正文 | 精读 |
| C-13 | [Comet - Fine-grained Computation-communication Overlapping for Mixture-of-Experts](C-13-CAC4AVZQ-comet-moe-overlap.md) | 正文 | 按章读 |
| C-14 | [GBA - A Tuning-free Approach to Switch between Synchronous and Asynchronous Training for Recommendation Model](C-14-X7YZ329D-gba-tuning-free-sync-async.md) | 正文 | 按章读 |
| C-15 | [Conditional Memory via Scalable Lookup - A New Axis of Sparsity for Large Language Models](C-15-A8BIBZTJ-engram-conditional-memory.md) | 正文 | 精读 |
| C-16 | [Design of a Hybrid MPI-CUDA Benchmark Suite for CPU-GPU Clusters](C-16-ED6MKDZR-hybrid-mpi-cuda-benchmark-suite.md) | 摘要 | 只需结论 |

## 怎么读

50 篇是并列的清单，但技术上有先后。时间有限时按下面五条线读，其余当参考书查。

**线一 · 该不该扩、往哪扩（A-01 → C-05 → A-02）**

`A-01` 用纯 ID 的序列模型给出**未经修饰的幂律基线**，并证明小模型的 loss 能预测大模型。
`C-05` 是 HSTU 团队续作，主张曲线斜率还能压——注意它的"弯"有精确定义：
**scaling efficiency 是性能对算力的回归斜率**，所以 5×/21× 指的是效率倍数，不是"训练快 5 倍"。
`A-02` 反驳得更彻底：推荐上的 scaling 不理想不是算力不够，是架构与多源异构特征不兼容，所以得换架构。

`A-01` 与 `A-02` 必须对着读。`A-01` 刻意剥掉全部附属特征只留行为，`A-02` 恰恰把多场景多行为当成必须处理的输入。
只读一篇会得出相反的结论。

**线二 · 装不下怎么办：训练侧与推理侧是两件事（B-09 → B-02 → B-15 → B-01）**

前三是训练侧路线，`B-01` 是推理侧路线。这条线的关键分歧不是"分几层"，而是
**`B-09` 用生产数据论证"广告主模型不能损失 0.1% AUC"**——这条论证是 B 组分成
"压缩派"与"层级存储派"的原始分界。同时要注意 `B-09` 自身结论的边界：
它只测了一种哈希方法就断言哈希必然掉精度，而 `B-03` 的 CAFE+ 在一万倍压缩下 AUC 还涨了 3.94%。

**线三 · 怎么塞小：压缩的三条不同轴（B-07 → B-10 → B-11）**

`B-07` 压的是**表的数学表示**（一整行变成核矩阵的收缩结果，"行"这个概念消失）；
`B-10` 压的是**每行的位宽**（按频率分组给不同精度）；`B-11` 压的是**全部到 INT4**，并把量化当正则。
前两者是"逐行存储"的变体（行还在，只是变少或变短），`B-07` 换了表示层——这不是同一条轴上的更激进，
代价也不同：它每次查询要多一次张量收缩。
`B-10` 与 `B-11` 的分歧在副作用的定性：统一低精度带来正则收益，混合精度既没有这个副作用也没有这份收益。

**线四 · 通信打满：两个相反的假设（A-06 → A-14 → A-15）**

`A-14` 与 `A-15` 不是替代关系，是镜像。两者优化同一段 all-to-all，方向相反：
`A-14` 假设"消息太大是瓶颈"，要把消息压小；`A-15` 假设"消息太碎是瓶颈"，要把碎消息合成大消息。
这两个假设不可能同时成立，**最优解一定在中间某个消息尺寸**——这是这一组里最明确的开放问题。
`A-14` 自己也写了"先融合成合适大小的消息再压缩，两者可叠"。

**线五 · 模型换形状之后（C-02 → C-03 → C-11 → C-12）**

`C-02` 把推荐重写成序列 transduction，`C-03` 给出工业落地形态。
`C-11` 与 `C-12` 都在处理分布式训练的副作用，但不是同一件事：
`C-11` 发现异步流水并行的梯度过期时间**随流水深度线性增长**（正好抵消异步想拿的扩展性优势），
病理是 Hessian 基与坐标基错位，所以它在参数表示上转基；
`C-12` 的病理是**变长序列让"负载均衡"这个前提失效**，所以它在通信原语上换回 P2P 拉推。

## 使用这批解读时要知道的三件事

1. **没有跨篇定量对比**。每篇报的是自己的加速比，基线与数据集各不相同，**不能直接排序**。
   文中所有"X 比 Y 快"都限定在作者自报的基线下；跨篇比较只在同一系统的前后代之间成立
   （例如 `A-13` 相对 `B-07` 的 3.1×）。
2. **17 篇只有摘要**，其中被判"精读"却只有摘要的有 `B-13`、`B-14`、`B-15` 三篇。
   它们的机制细节标为待核实，要真正复述算法必须补到原文。
3. **全部结论来自论文自报**，没有一篇经过本仓库复现。

## 相关

- 单篇解读：同目录下 50 个文件（按 `A-/B-/C-` 编号）
- 索引与字段说明：[index.md](all-papers.md)
