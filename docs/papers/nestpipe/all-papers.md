---
title: NestPipe 论文索引
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, index, recsys-infra]
related: [docs/papers/nestpipe/index.md]
zotero_collection: 9I82QPTL
---

# NestPipe 论文索引

50 篇的扁平索引，按分组与编号排序。导读与阅读顺序见 [导读](index.md)。

## 字段说明

| 字段 | 含义 |
|---|---|
| 编号 | `A-/B-/C-` 为策展分组，数字为组内序号 |
| 年份 | 发表年份，取自 Zotero 元数据 |
| body | **正文** = 取到论文正文；**摘要** = 只有摘要与官方会议页 |
| read | **精读** = 值得逐节读；**按章读** = 读指定章节；**只需结论** = 知道做了什么即可 |
| last | 一句话结论（解决什么问题、付了什么代价） |

`zotero_key` 可直接用于在 Zotero 中定位条目（`libraryID=1`，collection `9I82QPTL`）。

## A · 规模化训练系统与并行架构

| 编号 | 年份 | 论文 | zotero_key | body | read | last |
|---|---|---|---|---|---|---|
| A-01 | 2024 | [Scaling Law of Large Sequential Recommendation Models](A-01-RJYMAGIN-scaling-law-sequential-recsys.md) | `RJYMAGIN` | 正文 | 按章读 | ID-only 序列推荐模型从 98.3K 扩到 0.8B 参数时，scaling law 依然成立，且小模型能预测大模型。 |
| A-02 | 2025 | [Climber: Toward Efficient Scaling Laws for Large Recommendation Models](A-02-ABZSE7KJ-climber-efficient-scaling-laws.md) | `ABZSE7KJ` | 正文 | 精读 | 把长序列拆成多尺度子序列 + 自适应温度，让 Transformer 的 scaling 曲线变陡同时不爆线上延迟。 |
| A-03 | 2025 | [Two-dimensional Sparse Parallelism for Large Scale Deep Learning Recommendation Model Training](A-03-DZZH6L95-two-dimensional-sparse-parallelism.md) | `DZZH6L95` | 正文 | 精读 | 嵌入表不再全切到所有 GPU，而是分组复制 + 组内切分，配上改过的 AdaGrad 保住效果。 |
| A-04 | 2022 | [Persia: An Open, Hybrid System Scaling Deep Learning-based Recommenders up to 100 Trillion Parameters](A-04-892HASSZ-persia-hybrid-100t-params.md) | `892HASSZ` | 正文 | 精读 | 嵌入层走 CPU 参数服务器异步更新、稠密层走 GPU AllReduce 同步更新，两套机制拼出百 T 参数。 |
| A-05 | 2021 | [Hierarchical Training: Scaling Deep Recommendation Models on Large CPU Clusters](A-05-HP588YPM-hierarchical-training-cpu-clusters.md) | `HP588YPM` | 摘要 | 按章读 | CPU 集群上把神经网络部分按算子切开、组内既做数据并行又做模型并行，提速 1.9× 不掉点。 |
| A-06 | 2022 | [Software-Hardware Co-design for Fast and Scalable Training of Deep Learning Recommendation Models](A-06-J9ZPBVH5-sw-hw-codesign-neo-4d-parallelism.md) | `J9ZPBVH5` | 正文 | 精读 | 4D 并行 + 算子级 kernel 融合 + 专用 ZionEX 网络，128 卡上训 12T 参数模型，端到端 40× 加速。 |
| A-07 | 2022 | [Merlin HugeCTR: GPU-accelerated Recommender System Training and Inference](A-07-B2I9ZAIQ-merlin-hugetr-gpu-training-inference.md) | `B2I9ZAIQ` | 正文 | 只需结论 | HugeCTR 用 GPU 嵌入缓存 + 三级存储参数服务器，把训练与推理都做成 GPU 优先的工业框架。 |
| A-08 | 2022 | [TorchRec: a PyTorch Domain Library for Recommendation Systems](A-08-4GP3GTPM-torchrec-pytorch-domain-library.md) | `4GP3GTPM` | 摘要 | 只需结论 | Meta 把生产推荐栈的稀疏原语开源成 PyTorch 库，用 sharder + planner 自动分片嵌入表。 |
| A-09 | 2022 | [PICASSO: Unleashing the Potential of GPU-centric Training for Wide-and-deep Recommender Systems](A-09-XGVN6JZA-picasso-gpu-centric-wdl-training.md) | `XGVN6JZA` | 正文 | 精读 | WDL 模型把 GPU 用废了，靠算子打包、交错执行和分层哈希缓存放回硬件利用率。 |
| A-10 | 2023 | [Bagpipe: Accelerating Deep Recommendation Model Training](A-10-YYFDGRXK-bagpipe-accelerating-dlrm-training.md) | `YYFDGRXK` | 正文 | 精读 | 离线训练能预知未来要查哪些嵌入，于是用 lookahead 预取 + 逻辑复制物理分区缓存，提速 5.6×。 |
| A-11 | 2022 | [Training Personalized Recommendation Systems from (GPU) Scratch: Look Forward not Backwards](A-11-GYXLW7PP-scratchpipe-look-forward-not-backwards.md) | `GYXLW7PP` | 正文 | 精读 | 训练集里已经写着未来要查哪些嵌入，用它做流水线预取，把 CPU-GPU 混合训练变成"等效 GPU-only"。 |
| A-12 | 2024 | [Heterogeneous Acceleration Pipeline for Recommendation System Training (Hotline)](A-12-9DTGQILE-hotline-heterogeneous-acceleration-pipeline.md) | `9DTGQILE` | 正文 | 按章读 | 造一块加速器把 mini-batch 切成热门/非热门两半，让 GPU 只算热门的、CPU 侧的数据搬运被完全藏住。 |
| A-13 | 2024 | [Accelerating Distributed DLRM Training with Optimized TT Decomposition and Micro-Batching (EcoRec)](A-13-TFTSRALD-ecorec-tt-decomposition-micro-batching.md) | `TFTSRALD` | 摘要 | 只需结论 | 用 TT 分解压嵌入表会带来额外计算，EcoRec 消除冗余计算并配微批，比 EL-Rec 快 3.1×。 |
| A-14 | 2024 | [Accelerating Communication in Deep Learning Recommendation Model Training with Dual-Level Adaptive Lossy Compression](A-14-S996SB9M-dual-level-adaptive-lossy-compression.md) | `S996SB9M` | 正文 | 精读 | all-to-all 传的嵌入向量有结构特征，用误差有界的有损压缩 + 双重自适应误差界，通信快 8.6×。 |
| A-15 | 2026 | [FusedRec: Fused Embedding Communication for Distributed Recommendation Training on GPUs](A-15-SZH9CX5W-fusedrec-fused-embedding-communication.md) | `SZH9CX5W` | 摘要 | 按章读 | 去重让 all-to-all 的消息变得又小又碎，把多个类别的嵌入融合成一组一次发，快了 37.8%。 |
| A-16 | 2025 | [RecIS: Sparse to Dense, A Unified Training Framework for Recommendation Models](A-16-ZZ2P8YPT-recis-unified-sparse-dense-framework.md) | `ZZ2P8YPT` | 正文 | 精读 | 把 TensorFlow 那套稀疏训练搬到 PyTorch 上，用带宽 roofline 指导优化，稀疏稠密统一训。 |
| A-17 | 2025 | [Research on Model Parallelism and Data Parallelism Optimization Methods in Large Language Model-Based Recommendation Systems](A-17-3FPR4H9A-llm-recsys-parallelism-optimization.md) | `3FPR4H9A` | 摘要 | 只需结论 | 在 LLM 推荐场景里把 TP、PP、DP 混起来用，配上自适应负载均衡与梯度压缩，吞吐 +30%。 |

## B · 嵌入表存储、缓存、压缩与分片

| 编号 | 年份 | 论文 | zotero_key | body | read | last |
|---|---|---|---|---|---|---|
| B-01 | 2023 | [EVStore: Storage and Caching Capabilities for Scaling Embedding Tables in Deep Recommendation Systems](B-01-KP82AUTM-evstore-embedding-storage-cache.md) | `KP82AUTM` | 摘要 | 按章读 | 给推荐推理加三层嵌入缓存，DRAM 不够时用近似换内存，最多省 94% 内存。 |
| B-02 | 2025 | [Efficient and scalable huge embedding model training via distributed cache management](B-02-9XB4UUDV-het-distributed-embedding-cache.md) | `9XB4UUDV` | 正文 | 精读 | worker 本地缓存嵌入，且允许"写也过期"，通信降 88%。 |
| B-03 | 2025 | [CAFE+: Towards Compact, Adaptive, and Fast Embedding for Large-scale Online Recommendation Models](B-03-WBL86RZE-cafe-plus-compact-adaptive-embedding.md) | `WBL86RZE` | 摘要 | 按章读 | 热特征独享、冷特征哈希共享，1 万倍压缩仍涨 3.94% AUC。 |
| B-04 | 2024 | [Embedding Optimization for Training Large-scale Deep Learning Recommendation Systems with EMBark](B-04-JXSQDUAF-embark-embedding-clusters-3d-sharding.md) | `JXSQDUAF` | 摘要 | 按章读 | 按通信方式分簇 + 3D 分片，训练吞吐平均 1.5×。 |
| B-05 | 2022 | [AutoShard: Automated Embedding Table Sharding for Recommender Systems](B-05-J7EYKA8M-autoshard-rl-embedding-sharding.md) | `J7EYKA8M` | 正文 | 按章读 | 用神经代价模型预测多表开销、用深度强化学习做分配，几百张表秒级出分片方案。 |
| B-06 | 2023 | [Pre-train and Search: Efficient Embedding Table Sharding with Pre-trained Neural Cost Models](B-06-WXMZVWVJ-neuroshard-pretrain-and-search.md) | `WXMZVWVJ` | 正文 | 精读 | 预训练代价模型当模拟器，再搜索出近优分片方案。 |
| B-07 | 2022 | [EL-Rec: Efficient Large-Scale Recommendation Model Training via Tensor-Train Embedding Table](B-07-UZF6ZILW-elrec-tensor-train-embedding.md) | `UZF6ZILW` | 摘要 | 按章读 | TT 分解压缩嵌入表，最大公开 DLRM 数据集单卡快 3×。 |
| B-08 | 2025 | [Unified and Near-optimal Multi-GPU Cache for Embedding-based Deep Learning](B-08-6NY24IFU-ugache-unified-multi-gpu-cache.md) | `6NY24IFU` | 摘要 | 按章读 | 多卡显存拼成缓存池，分源提取 + 热度策略，平均快 1.93×。 |
| B-09 | 2020 | [Distributed Hierarchical GPU Parameter Server for Massive Scale Deep Learning Ads Systems](B-09-SLPPGD8C-hierarchical-gpu-parameter-server.md) | `SLPPGD8C` | 正文 | 精读 | 显存→内存→SSD 三层存 10TB 参数，4 台机器快过 150 台。 |
| B-10 | 2024 | [Mixed-Precision Embeddings for Large-Scale Recommendation Models](B-10-DH7ZFV2C-mpe-mixed-precision-embeddings.md) | `DH7ZFV2C` | 正文 | 精读 | 按频率分组、用可微分布搜索每组位宽，Criteo 压 200× 不掉精度。 |
| B-11 | 2024 | [DQRM: Deep Quantized Recommendation Models](B-11-Z56XJYUC-dqrm-deep-quantized-recommendation.md) | `Z56XJYUC` | 正文 | 精读 | INT4 量化缩 8×，还顺带治好过拟合，精度不降反升。 |
| B-12 | 2023 | [AdaEmbed: Adaptive Embedding for Large-Scale Recommendation Models](B-12-3K9KISR6-adaembed-adaptive-embedding-pruning.md) | `3K9KISR6` | 摘要 | 按章读 | 训练中在线剪掉不重要的嵌入行，达到同等精度只要 35–60% 的表大小。 |
| B-13 | 2024 | [OPER: Optimality-Guided Embedding Table Parallelization for Large-scale Recommendation Model](B-13-37Q3PZXS-oper-optimality-guided-parallelization.md) | `37Q3PZXS` | 摘要 | 精读 | 分片必须按输入来定，否则必然失衡，快 2.3×/4.0×。 |
| B-14 | 2024 | [Accelerating Neural Recommendation Training with Embedding Scheduling](B-14-6HTV2P4W-herald-embedding-scheduling.md) | `6HTV2P4W` | 摘要 | 精读 | 预测"哪些嵌入会再被访问"，主动决定在哪训练、同步谁，通信降 48–89%。 |
| B-15 | 2025 | [HypeReca: Distributed Heterogeneous In-Memory Embedding Database for Training Recommender Models](B-15-VE9T6R4E-hypereca-in-memory-embedding-database.md) | `VE9T6R4E` | 摘要 | 精读 | 把嵌入表当分布式 KV 库，分块分配 + 热点复制，快 2.16–16.8×。 |
| B-16 | 2025 | [The Evolution of Embedding Table Optimization and Multi-Epoch Training in Pinterest Ads Conversion](B-16-L2VVL9ZT-pinterest-embedding-multi-epoch.md) | `L2VVL9ZT` | 正文 | 精读 | 给嵌入表配 50× 学习率、给低频行降学习率，多 epoch 过拟合基本能治。 |
| B-17 | 2025 | [DCMA: Accelerating Parallel DMA Transfers with a Multi-Port Direct Cached Memory Access in a Massive-Parallel Vector Processor](B-17-WCV8CKED-dcma-multi-port-cached-dma.md) | `WCV8CKED` | 摘要 | 只需结论 | DMA 加多端口缓存，CNN 少等内存，ResNet-50 最高 17×。 |

## C · 模型演进、生成式推荐与流水线调度

| 编号 | 年份 | 论文 | zotero_key | body | read | last |
|---|---|---|---|---|---|---|
| C-01 | 2025 | [Generative Recommendation Models - Progress and Directions](C-01-GMPG7G9I-generative-recsys-progress-directions.md) | `GMPG7G9I` | 摘要 | 只需结论 | 把生成式推荐（LLM 式、语义 ID、扩散式）的路线图讲清楚，是这条线的地图而非新方法。 |
| C-02 | 2024 | [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](C-02-G73SFXWD-hstu-trillion-param-generative-recsys.md) | `G73SFXWD` | 正文 | 精读 | 把推荐整个改写成 next-token 生成任务，配一个非 softmax 注意力编码器，换来算力可缩放。 |
| C-03 | 2025 | [MTGR - Industrial-Scale Generative Recommendation Framework in Meituan](C-03-DDWBU7H3-mtgr-meituan-generative-recsys.md) | `DDWBU7H3` | 正文 | 精读 | 生成式推荐不能丢交叉特征；MTGR 用 token 重排 + 分组归一化把它们留住。 |
| C-04 | 2025 | [FuXi-α - Scaling Recommendation Model with Feature Interaction Enhanced Transformer](C-04-AXNSLSLL-fuxi-alpha-feature-interaction-transformer.md) | `AXNSLSLL` | 正文 | 按章读 | 把时间、位置、语义拆成三条注意力通道，再补两层 FFN，让 Transformer 真的能扩层。 |
| C-05 | 2026 | [Bending the Scaling Law Curve in Large-Scale Recommendation Systems](C-05-ASKWXRVH-ultra-hstu-bending-scaling-law.md) | `ASKWXRVH` | 正文 | 精读 | 不改自注意力，改它的形状：半局部稀疏 + 输入减半 + FP8，把缩放曲线压平。 |
| C-06 | 2024 | [Disaggregated Multi-Tower - Topology-aware Modeling Technique for Efficient Large-Scale Recommendation](C-06-NF8E5IDY-dmt-disaggregated-multi-tower.md) | `NF8E5IDY` | 正文 | 按章读 | 把"全局嵌入 AlltoAll"换成"机内局部洗牌"，靠拆塔蹭数据中心的带宽层级。 |
| C-07 | 2022 | [KuaiRand - An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos](C-07-FCJDT7DA-kuairand-unbiased-dataset.md) | `FCJDT7DA` | 正文 | 按章读 | 快手把随机视频塞进真实推荐流两周，攒出带 12 种反馈的无偏序列数据集。 |
| C-08 | 2025 | [SlimPipe - Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training](C-08-9MCTBGR4-slimpipe-memory-thrifty-pp.md) | `9MCTBGR4` | 正文 | 精读 | 把每个 microbatch 切成 n 片序列，让 PP 的激活显存和气泡同时除以 n。 |
| C-09 | 2025 | [TD-Pipe - Temporally-Disaggregated Pipeline Parallelism Architecture for High-Throughput LLM Inference](C-09-2JX9YAFY-td-pipe-temporal-disaggregated-pp.md) | `2JX9YAFY` | 正文 | 按章读 | prefill 和 decode 分时段做，别混在一批里，气泡就没了。 |
| C-10 | 2025 | [OctoPipe - Reducing Pipeline Bubbles for Heterogeneous Models via Co-Optimizing Partitioning, Placement, and Scheduling](C-10-UFFU6KVA-octopipe-heterogeneous-bubbles.md) | `UFFU6KVA` | 正文 | 按章读 | 异构模型的气泡要治，得把切分、放置、调度三件事一起调，不能只调一样。 |
| C-11 | 2026 | [Mitigating Staleness in Asynchronous Pipeline Parallelism via Basis Rotation](C-11-W5IVJUAZ-basis-rotation-staleness.md) | `W5IVJUAZ` | 正文 | 精读 | 异步流水的 stale 梯度之所以害人，是因为 Hessian 基与坐标基错位；转个基就好。 |
| C-12 | 2026 | [Revisiting Parameter Server in LLM Post-Training](C-12-ZLR452K5-odc-revisiting-parameter-server.md) | `ZLR452K5` | 正文 | 精读 | FSDP 的 per-layer 同步在变长序列下变成拖累，换成 P2P 拉推就回到 PS 的宽容。 |
| C-13 | 2025 | [Comet - Fine-grained Computation-communication Overlapping for Mixture-of-Experts](C-13-CAC4AVZQ-comet-moe-overlap.md) | `CAC4AVZQ` | 正文 | 按章读 | MoE 的通信和计算粒度对不上，别在 kernel 外做流水，塞进一个 kernel 里按线程块分。 |
| C-14 | 2022 | [GBA - A Tuning-free Approach to Switch between Synchronous and Asynchronous Training for Recommendation Model](C-14-X7YZ329D-gba-tuning-free-sync-async.md) | `X7YZ329D` | 正文 | 按章读 | 把异步训练的全局 batch 对齐到同步，就能不调超参在两种模式间来回切。 |
| C-15 | 2026 | [Conditional Memory via Scalable Lookup - A New Axis of Sparsity for Large Language Models](C-15-A8BIBZTJ-engram-conditional-memory.md) | `A8BIBZTJ` | 正文 | 精读 | MoE 之外再开一条稀疏轴：用哈希查 N-gram 表当下一个"内存层"，查得比算得快。 |
| C-16 | 2014 | [Design of a Hybrid MPI-CUDA Benchmark Suite for CPU-GPU Clusters](C-16-ED6MKDZR-hybrid-mpi-cuda-benchmark-suite.md) | `ED6MKDZR` | 摘要 | 只需结论 | 给 CPU-GPU 混合集群造三个通信+计算密集的基准，用来暴露节点间通信瓶颈。 |

## 交叉引用

组间引用最密集的几处（要看某篇与其它篇的关系，直接查这几篇的「为什么值得读」一节）：

| 主题 | 涉及的篇目 |
|---|---|
| 嵌入表压缩的三条轴 | `B-07`（表示层）、`B-10`（位宽）、`B-11`（全量化）；`B-12` 与 `B-03` 在行列两个方向上的分解 |
| 分片方案 | `B-04`（分簇 + 3D）、`B-05`（DRL）、`B-06`（代价模型 + 搜索）、`B-13`（按输入分片） |
| 训练侧分层存储 | `B-09`（静态分片）、`B-02`（允许过期）、`B-15`（动态插入）、`A-07`（三级存储 PS） |
| 用未来信息预取 | `A-10`（lookahead + 逻辑复制）、`A-11`（训练集即未来）、`B-14`（预测访问） |
| 通信：压小 vs 合大 | `A-14`（压缩）、`A-15`（融合）；`A-03` 与 `A-06` 在并行维度上 |
| 生成式推荐主线 | `C-01`（地图）、`C-02`（HSTU）、`C-05`（ULTRA-HSTU）、`C-03`（工业落地）、`C-04`（特征交互） |
| 流水并行气泡与持久化 | `C-08`（切序列）、`C-09`（时分）、`C-10`（三者联调）、`C-11`（异步 stale） |
| 同步 vs 异步 | `C-11`、`C-12`、`C-14`；`B-02` 与 `B-14` 在嵌入侧 |
