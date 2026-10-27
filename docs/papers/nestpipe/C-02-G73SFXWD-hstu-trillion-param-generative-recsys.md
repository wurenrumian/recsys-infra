---
title: "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-C, HSTU, generative-recommendation, pointwise-attention, M-FALCON, scaling-law]
related: [docs/papers/nestpipe/index.md]
zotero_key: G73SFXWD
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Jiaqi Zhai、Lucy Liao、Xing Liu、Yueming Wang、Rui Li、Xuan Cao、Leon Gao、Zhaojie Gong、Fangda Gu、Michael He、Yinghai Lu、Yu Shi 等（Meta；`jiaqiz@meta.com` 等为通讯，**事实**，正文首页）。
- 年份：2024（arXiv v1 2024-02-27；后为 ICML 2024 论文，**事实**：正文关键词行标注 ICML，arXiv 提交历史）。
- 会议：ICML 2024。
- URL：`https://arxiv.org/abs/2402.17152`（DOI `10.48550/arXiv.2402.17152`）。
- 取到的正文：`https://arxiv.org/html/2402.17152v2`（HTML 全文，**事实**）。

## 一句话

把推荐整个改写成 next-token 生成任务，配一个非 softmax 注意力编码器，换来算力可缩放。

## 背景与问题

DLRM 统治推荐系统近十年（**事实**），但有个说不清的毛病：**算力加不上去**。摘要原话是 "most DLRMs in industry fail to scale with compute"，正文更直白——"Despite utilizing extensive human-engineered feature sets and training on vast amounts of data, most DLRMs in industry scale poorly with compute"（**事实**）。

作者把这条技术线要跨的三道坎列清楚了（**事实**，§1）：

| 坎 | 具体形态 | 数字 |
|---|---|---|
| 特征没有显式结构 | 异质特征混在一起：高基数 ID、交叉特征、计数器、比率 | 特征空间可达十亿级基数 |
| 词表是十亿级且持续变化 | LLM 是 10 万级静态词表，推荐是十亿级动态词表 | 100K vs 1B |
| 算力本身就是主瓶颈 | 用户序列长度可达 10^5 | 每天要处理的 token 比 GPT-3 训练总量还多几个数量级 |

第三点是全篇的立论支点：GPT-3 用 300B token 训 1–2 个月，而大型平台**每天**产生几十亿用户动作。**如果照搬 LLM 的逐样本训练法，算力账直接破产**（**推断**，由上述数字推出）。

## 核心做法

### 1. 先改数据组织：把 DLRM 的特征空间"序列化"

这是全篇最容易被跳读、但最关键的一步。GR（Generative Recommender）不是"换个模型"，是**先把输入换掉**（**事实**，§2.1）：

- **类别特征**：选出最长的时间序列（通常是用户互动过的 item）作为**主序列**；其余慢变序列（人口属性、关注的创作者）按"连续段只保留最早一条"压缩后**并进主序列**。因为慢变，序列长度几乎不增加。
- **数值特征**：计数器、比率这类高频特征**直接扔掉**。理由是它们的聚合维度（item 主题、地点等类别特征）已经被序列化了；只要序列够长、算力够、架构够表达 + target-aware，这些数值信息可以被注意力"算出来"。

这解释了为什么后面所有 HSTU 系工作的"输入设计"都值得单独做一节——**扔掉数值特征这一步，是拿模型容量换工程简化**。

### 2. 再把任务改成序列传导（sequential transduction）

给定按时间排的 token 序列 `x_0..x_{n-1}` 与时间戳 `t_0..t_{n-1}`，序列传导任务把它映射到 `y_0..y_{n-1}`，`y_i = ∅` 表示该位置无监督（**事实**，§2.2 定义）。

排序和召回被统一成两个同构任务（**事实**，Table 1）：

| 任务 | 输入 | 输出 | 关键点 |
|---|---|---|---|
| 召回 | `Φ_0, a_0, Φ_1, a_1, …` | `a_0, ∅, a_1, ∅, …` | 监督信号不是下一个 item，而是**用户对它的动作**（可能是负反馈） |
| 排序 | 同上，item 与 action **交错** | `Φ'_1, Φ'_2, …`（正向动作的 item） | 交错使得 target 与历史可以在**第一层**就交互 |

排序那格是精髓：标准自回归把 target 的"交互"放在最后（softmax 之后），而工业排序必须 **target-aware**——target 要尽早和历史特征交互。交错排列把这件事变成 `p(a_{i+1} | Φ_0,a_0,…,Φ_{i+1})`，于是**一次前向就能对全部 n_c 个历史互动做 target-aware 交叉注意力**（**事实**）。

### 3. 算法层的算力腾挪：从 impression-level 到 generative training

原做法每个曝光样本都要跑一遍编码器，代价 `Σ_i n_i(n_i²d + n_i d_ff d)`，整理后 `O(N³d + N²d²)`（**事实**，§2.2 复杂度推导）。

改成按用户采样、每个用户只跑一次编码器，把多个 target 摊到同一份编码成本上：采样率设 `s_u(n_i) = 1/n_i` 后总代价降到 `O(N²d + Nd²)`——**省了一个 O(N) 因子**（**事实**）。工程实现上就是"在用户请求/会话结束时发训练样本"。

数据流对比（**事实**，§2 描述）：

```
DLRM（impression-level）:
  for each (user, candidate):   -> 跑一次全网络 -> 一个梯度
  代价 ∝ 候选数 × 序列长³

GR（generative）:
  for each user:                -> 跑一次编码器 -> n 个位置的 n 个 loss
  代价 ∝ 序列长²
```

### 4. HSTU 层本身

每层三个子层：Pointwise Projection → Spatial Aggregation → Pointwise Transformation（**事实**，Eq.1–3）：

```
U,V,Q,K = Split( φ1( f1(X) ) )        # f1 只有一个线性层，QKVU 融合成一个 kernel
A(X)V(X) = φ2( Q(X)K(X)ᵀ + rab^{p,t} ) V(X)
Y(X)     = f2( Norm( A(X)V(X) ) ⊙ U(X) )
```

三个设计选择各自换了什么：

| 设计 | 换来什么 | 代价 |
|---|---|---|
| **pointwise 聚合注意力**（不是 softmax，没有全序列归一化因子） | 保留"有多少相关历史"这个强度特征；对非稳态词表更稳 | 需要 layer norm 兜底才不炸（**事实**："layer norm is needed after pointwise pooling to stabilize training"） |
| **`Norm(AV) ⊙ U` 门控** | 一个 element-wise 乘法同时干掉了 DLRM 的三件事：特征抽取（pairwise/target-aware pooling）、特征交叉（直接交互而非用 MLP 拟合点积）、MoE 式门控 | 表达能力上限依赖 U 的 SiLU（可看作 SwiGLU 变体） |
| **不存 relative bias 的独立参数表**，rab 融进注意力 | 位置 + 时间信息一起进 | — |

合成数据实验给了 pointwise vs softmax 的定量差距：在 Dirichlet Process 生成的流式非稳态词表上，两者差距可达 **44.7%**（**事实**，§3）。

### 5. 系统层三件事（这才是 NestPipe 视角的重点）

**(a) 稀疏化 kernel。** 用户历史长度是长尾分布，大量短序列。HSTU 的 kernel 把注意力算成**大小不一的 grouped GEMM**（fully raggified），内存访问量降到 `Θ(Σ_i n_i² d_qk² R⁻¹)`，R 是寄存器大小（**事实**）。单这一项带来 2–5× 吞吐（**事实**）。

**(b) Stochastic Length（SL）。** 主动再削一遍稀疏度：序列长 `n_{c,j} > N_c^{α/2}` 时，以概率 `1 − N_c^α/n_{c,j}²` 截到 `N_c^{α/2}`。注意力复杂度变成 `O(N^α d)`，α ∈ (1,2]（**事实**）。实测 α=1.6 时，4096 长的序列多数时候被截成 **776**（削掉 >80% token）；即便稀疏率升到 64%–84%，主任务 NE 的退化不超过 **0.002（0.2%）**（**事实**）。这是全篇性价比最高的一招——不改模型、不改系统，直接省算力。

**(c) 激活内存与嵌入内存。** 这两条是 DLRM→GR 迁移时真正的落地障碍：

- **激活内存**：推荐要**大 batch**（对比 LLM 常用小 batch、参数内存为主），于是激活内存成主瓶颈。HSTU 把注意力之外的线性层从 6 个减到 2 个，并把 φ1(f1(·))、layernorm、dropout、输出 MLP 全融进单个算子 → 每层 bf16 激活 `14d`（**事实**，§3.3 推导）。结果：同等算力下能堆**2 倍深度**的网络（**事实**）。
- **嵌入内存**：10B 词表 × 512 维 × Adam，fp32 存参数+优化器状态需 **60TB**（**事实**）。解法是 rowwise AdamW + 优化器状态放 DRAM，**每浮点 HBM 占用从 12 字节降到 2 字节**。

**(d) M-FALCON（推理）。** 排序要处理上万个候选。做法是把候选分 microbatch，篡改 attention mask 与 rab，使得**一个 microbatch 内 b_m 个候选的注意力计算完全相同**，于是交叉注意力成本从 `O(b_m n² d)` 降到 `O((n+b_m)² d) ≈ O(n²d)`（**事实**）。这正是"285× 更复杂的模型、同样推理预算"的来源。

## 关键数字

| 指标 | 数值 | 等级 |
|---|---|---|
| 公开数据集 NDCG 提升 | 最高 **65.8%** | 事实（摘要） |
| 8192 长序列上 vs FlashAttention2 Transformer | **5.3×–15.2×** 更快（训练最高 15.2×，推理最高 5.6×） | 事实 |
| 线上 A/B 指标提升 | **12.4%** | 事实 |
| 最大模型参数 | **1.5 万亿** | 事实（摘要+正文） |
| DLRM 质量饱和点 | 约 **200B** 参数 | 事实（正文 §4.3） |
| 同等推理预算下可服务的模型复杂度 | **285×** | 事实 |
| 推理吞吐 | 1.50×–2.99×（另一处写 1.5×–3×） | 事实 |
| 缩放律覆盖范围 | 算力跨 **3 个数量级**，上探到 GPT-3/LLaMa-2 量级 | 事实 |
| 小规模对照中 HSTU vs Transformer++ | 1.5×–2× 更快、HBM 少 50% | 事实 |
| SL 的 NE 代价 | < 0.002 退化，削掉 64%–84% token | 事实 |

## 为什么值得读

- **它是 Track C 的原点**。C-03 MTGR、C-04 FuXi-α、C-05 ULTRA-HSTU 全部以本篇为基线或改造对象（**事实**：三者正文均引用 Zhai et al. 2024 并直接对比）。不读这篇，后面三篇的"我们比 HSTU 好 X%"没有参照物。
- **与 C-03 是"取舍关系的正反题"**。本篇主动扔掉交叉特征和数值特征，MTGR 反过来说"扔掉交叉特征的损失，放大模型补不回来"（**事实**，MTGR 摘要）。两篇合读才能看清"序列化"这一步的边界在哪。
- **与 C-04 是"结构改进"**。FuXi-α 的批评点很具体：HSTU 把时间/位置信息**直接加到注意力权重上**（即 rab），会稀释影响；且 HSTU 没有 FFN 层，隐式特征交叉不足（**事实**，FuXi-α 原文）。
- **与 C-05 是"同一团队的后继"**。ULTRA-HSTU 明确说"self-attention 仍然优于 cross-attention"，并给出 5.3×/21.4× 的**缩放效率**（斜率）改进——继续本篇的 scaling-law 叙事，但把优化点从"架构"搬到"输入设计 + 稀疏注意力 + 系统"。
- **对 NestPipe 这条线的独特价值**：它示范了一件事——**模型层的改动可以直接改写系统层的成本曲线**。把输入序列长度减半（去交错）、把注意力变线性（SLA），本质上都是在动"每层激活多大、通信多少"这个系统账本。C-08 SlimPipe、C-10 OctoPipe 优化的那张表，输入参数就是本篇定下来的序列长度分布。

## 局限

1. **基准对比的公平性存疑**。DLRM baseline 是"hundreds of people over multiple years"的工业系统，而 GR 是"a typical configuration used in production"（**事实**，正文原话）。两边调优预算不对等，12.4% 的线上收益里有多少来自范式、多少来自工程投入，论文没有拆开。
2. **低算力区间 GR 反而更差**。作者自己承认："In the low compute regime, DLRMs might outperform GRs due to handcrafted features"（**事实**）。所以这不是"全面替代"，是"大规模时替代"。
3. **扔数值特征的代价被低估**。论文的论证是"序列够长就能算出来"，但这要求算力足够大；在算力受限的中等规模，这个假设很可能不成立（**推断**）。C-03 MTGR 用工业数据反证了这一点。
4. **单点故障风险**。整个系统的效率建立在 HSTU 的 ragged grouped-GEMM kernel 上；换硬件（非 Hopper、非 NVIDIA）就要重写。ULTRA-HSTU 后来确实为此补了 AMD MI300x 的 kernel（**事实**）。
5. **M-FALCON 的 285× 有条件**。它是"同样推理预算下能服务多复杂的模型"，不是"同样的模型快 285 倍"，引用时极易误读。
6. **10^5 级序列在正文实验里没有真正跑通**。最大实验配置是 8192 序列长、1024 维、24 层（**事实**）。

## 证据与待核实

| 结论 | 等级 | 依据 |
|---|---|---|
| 全部结构公式（Eq.1–3）、M-FALCON、SL 定义 | 事实 | arXiv HTML 全文 |
| 65.8% / 12.4% / 5.3–15.2× / 1.5T / 285× | 事实 | 摘要 + §4 正文 |
| 60TB 嵌入内存、12→2 字节 HBM、14d 激活 | 事实 | §3.3 正文 |
| ICML 2024 录用 | 事实 | 正文关键词行 |
| "低算力区间 DLRM 更好" | 事实 | §4.3 正文 |
| 12.4% 里范式贡献的占比 | **待核实** | 论文未做归因拆解 |
| 十亿级候选表下的真实 QPS | **待核实** | 正文只给相对倍数，未给绝对值 |

没拿到的东西：ICML 正式版（可能与 arXiv v2 有差异）；附录 C（Dirichlet Process 合成数据构造）、G（ragged attention kernel 细节）、H.1（推理吞吐全量）——HTML 版附录存在但本次只精读了主干。

## 是否值得读原文

**必读**。

理由：这是 Track C 唯一一篇同时改写"任务定义 + 模型结构 + 训练系统 + 推理算法"的论文，后面五篇全是它的衍生或反驳。至少要读 §2（任务重写与复杂度推导）、§3（HSTU 三子层 + SL + M-FALCON）、§4.3（缩放律与低算力反例）三节；附录可跳过。
