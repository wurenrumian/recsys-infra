---
title: FuXi-α - Scaling Recommendation Model with Feature Interaction Enhanced Transformer
type: research
status: stable
created: 2026-10-27
updated: 2026-10-27
tags: [nestpipe, nestpipe-C, FuXi, AMS, MFFN, feature-interaction, scaling-law, WWW]
related: [docs/papers/nestpipe/index.md]
zotero_key: AXNSLSLL
zotero_collection: 9I82QPTL
evidence: fulltext
---

## 出处

- 作者：Yufei Ye、Wei Guo、Jin Yao Chin、Hao Wang、Hong Zhu、Xi Lin、Yuyang Ye、Yong Liu、Ruiming Tang、Defu Lian、Enhong Chen（**事实**，Zotero + 正文）。
- 单位：华为消费者业务集团（Consumer Business Group, Huawei, Shenzhen）+ 中科大（USTC）等（**事实**，正文 affiliation 与代码仓库 `USTC-StarTeam/FuXi-alpha`）。
- 年份与会议：**WWW '25**（The Web Conference 2025，2025-04-28 至 05-02，悉尼），DOI `10.1145/3701716.3715448`（**事实**，正文标题页）。
- arXiv：`https://arxiv.org/abs/2502.03036`。
- 代码：`https://github.com/USTC-StarTeam/FuXi-alpha`（**事实**，摘要声明）。
- 取到的正文：`https://arxiv.org/html/2502.03036v1`（**事实**）。

## 一句话

把时间、位置、语义拆成三条注意力通道，再补两层 FFN，让 Transformer 真的能扩层。

## 背景与问题

FuXi-α 站在 HSTU 的肩膀上，但提出两个**具体**批评（**事实**，§1 + §3.3 对比）：

**批评 1：HSTU 浪费了时间/位置信息。** 现有做法有三种——把位置 embedding 加进输入（SASRec）、加进 Q/K 矩阵、或调注意力权重（HSTU 的 `rab^{p,t}`）。前两种"简单相加缺少表达力"；HSTU 那种"直接加到注意力权重上会**稀释**其影响"（原文："HSTU incorporates relative temporal and positional data by adding these features directly to attention weights, which can dilute their impact"）。

**批评 2：HSTU 没有 FFN，隐式特征交叉不足。** 原文："HSTU lacks an FFN layer, relying solely on self-attention and explicit feature interactions, limiting its ability to capture complex item relationships"。

这个批评要放到 DLRM 的语境里才成立。推荐的特征交叉分两类（**事实**，§2.2）：**显式交叉**（点积/双线性/卷积/注意力，直接用算子建模）与**隐式交叉**（DNN/FFN 一次性吃下所有特征）。DLRM 家族（DCN、xDeepFM、DeepFM）靠两腿走路；HSTU 把两者压进一个注意力块，省了参数省了激活，但也把隐式那一腿削掉了。FuXi-α 的判断是：**这一腿不能省，省了就没有扩层的道理**。

还有第三个隐含问题：**扩层不 work**。SASRec 扩到 8 层性能反而掉（**事实**，§4.2："SASRec fails to scale up to 8 layers across all three datasets, with a significant performance drop"）；HSTU/LLaMa 能扩，但扩得不够好。

## 核心做法

FuXi-α = b 层 FuXi Block，每层 = **AMS**（自适应多通道自注意力）+ **MFFN**（多阶段前馈网络）（**事实**，§3）。

### 1. AMS：三种通道，各管一类信息

对比标准多头注意力（所有 head 都在语义空间里算）与 AMS（**事实**，Fig.3）：

| 通道 | 注意力权重怎么算 | 参数 | 直觉 |
|---|---|---|---|
| **语义** | `a_h = (1/n)·φ(q·kᵀ)`，φ = SiLU | 需 Q/K 投影（`W_q, W_k`） | 常规内容匹配 |
| **时间** | `(a_t)_{i,j} = α(t_j − t_i)`，时间差**分桶**后每桶一个可学参数 | **不需要** Q/K | 只关心"隔多久" |
| **位置** | `(a_p)_{i,j} = β_{j−i}`，相对位置 bias | **不需要** Q/K | 只关心"隔几位" |

三个关键设计（**事实**）：① 时间/位置两通道**不算 Q/K 矩阵**（只依赖相对差），省算力；② 三通道**共享 V 矩阵**（`v_h = φ(x̃ W_h)`），不给时间/位置单独造 V；③ 输出拼接后 `RMSN(concat(...)) ⊗ φ(x W_u)`——这里的 `⊗ U` 门控**照搬 HSTU**，作用是引入显式二阶交叉（原文："We adopted the design of the matrix U in our architecture following HSTU to introduce explicit 2-order interactions"）。

完整一层：

```
x̃  = RMSN(x^{l-1})
q,k,v = φ(x̃ W_q), φ(x̃ W_k), φ(x̃ W_h)        # 只算一次，三通道共用 v
a_h = (1/n)·φ(q kᵀ)                            # 语义
a_t = α(t_j − t_i)      （分桶查表）           # 时间
a_p = β_{j−i}           （相对位置 bias）      # 位置
h   = RMSN( concat(a_h v, a_p v, a_t v) ) ⊗ φ(x W_u)
o   = h W_o + x^{l-1}                          # MFFN 第一阶段：融合 + 残差
y   = FFN( RMSN(o) ) + o                       # MFFN 第二阶段：隐式交叉
```

注意 `1/n` 这个归一化因子——**不是 softmax**，而是 HSTU 式的 pointwise 归一化（除以序列长度）。FuXi-α 明确引用 HSTU 的发现：SiLU 在序列推荐里优于 softmax（**事实**，§3.2）。

### 2. 为什么"拆通道"真的更有表达力（论文给了证明）

这是本篇少见的理论部分（**事实**，§3.3）。把 FuXi Block 简化（注意力权重当常数、去掉 MFFN 第二阶段、去掉激活和多数投影）后：

```
f^(l)(x_i; x_1..x_n) = x_i ∘ ( Σ_j a_{i,j}^(l) x_j ) + x_i
```

归纳证明得到：经过 b 层后，

```
x_{b,i} = x_{0,i} · F_{2^b − 1}
```

即第 b 层的输出包含 `x_{0,i}` 与**所有 item 的任意阶（最高 2^b − 1 阶）交互**（**事实**）。

然后是关键的对比推导：标准做法 `A = φ(q kᵀ + B)` 里，位置 bias `B` 被卷进**同一个非线性**里，跟前一项混在一起。而三通道拆开后：

```
o_i ≈ W_{o1}( (Σ φ(q_i k_jᵀ) v_j) ⊗ u_i^(1) )  +  W_{o2}( (Σ b_{i,j} v_j) ⊗ u_i^(2) )
        └──── 语义通道，独立门控 ────┘            └──── 位置通道，独立门控 ────┘
```

**位置项和语义项各自有独立的 `u` 门控、独立的输出投影**。这就是"AM"（Adaptive Multi-channel）的字面含义：不是把信息混在一起加，而是各走各的门再融合（**事实**）。论文用这个推导论证"AMS 对时间/位置信息的表达力严格强于直接加到权重上"。

### 3. MFFN：两阶段

- **阶段 1 融合**：`o = h W_o + x^{l-1}`，把三通道输出投回 d 维并加残差；
- **阶段 2 隐式交叉**：`y = SwiGLU(x)·W_3 + o`，其中 SwiGLU 是 `(φ(xW_1) ⊗ (xW_2))W_3`，照 LLaMa 的做法配 RMSNorm（**事实**）。

消融显示**阶段 2 是最关键的**："removing the second stage of the MFFN results in a significant performance drop"（**事实**）。这条消融直接支撑了"隐式特征交叉不能省"的立论。

### 4. 复杂度账

每层参数（**事实**，§3.4）：AMS 有 4 个投影共 `6d × d_H`，位置/时间 embedding `O(n + n_B)`（`n_B` 是桶数）；MFFN 有 4 个投影共 `3d_h × d + 3d_FFN × d`；item embedding 为 `|I| × d`。时间上，语义通道 `O(n²d)`、另两通道 `O(n²)`（不算 Q/K，便宜）、QKV 与 MFFN 各 `O(nd²)`、预测 `O(n|I|d)`，合计 `O(bn²d + n(bd² + |I|d))`（**事实**）。

**注意：这仍是 O(n²)**。FuXi-α 没解决长序列的二次复杂度——这一点要留到 C-05 ULTRA-HSTU 才被正面处理。

### 5. 实验设置

四份数据（**事实**，§4.1）：MovieLens-1M、MovieLens-20M（用 HSTU 官方预处理脚本，保证可比）、KuaiRand（快手，平均 200+ 交互/用户）、华为音乐私有工业集（月活数千万）。基线两族：传统（BPRMF、GRU4Rec、NARM）+ 自回归生成式（SASRec、LLaMa、HSTU）。深度对照用 2 层 vs 8 层（"XX-Large"）。损失是 sampled softmax，负样本数 32–256 扫描（**事实**）。

## 关键数字

| 指标 | 数值 | 等级 |
|---|---|---|
| 浅层（2 层）vs 最强基线 HSTU | NDCG@10 **+13.24%**、NDCG@50 +10.59%、HR@10 +10.81%、HR@50 +6.94%、MRR +13.72%（三数据集平均） | 事实 |
| 深层（8 层）vs HSTU-Large | NDCG@10 **+7.26%**、NDCG@50 +5.24%、HR@10 +6.14%、HR@50 +3.19%、MRR +6.90% | 事实 |
| 工业集 vs SASRec | LLaMa +64.82% NDCG@10、HSTU +71.75%，FuXi-α 再比两者高 11.54% / 8.19% | 事实 |
| 线上 A/B（华为音乐，30% 流量，7 天） | 人均播放歌曲数 **+4.76%**（另一处写 4.67%，见待核实） | 事实 |
| 线上 A/B 人均听歌时长 | **+5.10%** | 事实 |
| 最大扩层试到 | 32 层（受显存限制） | 事实 |
| 负样本的收益 | 从 32 加到 256 持续提升，**收益超过加层** | 事实 |

最后一条值得单独记：**"加负样本比加层划算"**。这在 LLM scaling law 的语料里几乎没人提（论文原文就点出 "The influence of negative sampling on large recommendation models has been overlooked in studies on LLM scaling laws"）。对做系统的人来说这是个便宜的旋钮。

## 为什么值得读
- **它是 C-02 HSTU 的"结构修补版"**，修补点非常聚焦：只动两处（时间/位置怎么进注意力、有没有 FFN），其余（pointwise 归一化、U 门控、采样 softmax）全照抄。这种"最小改动 + 消融到点"的论文最容易复现。
- **与 C-03 MTGR 是同一问题的两种解**。两者都认定"HSTU 的特征交互不够"：MTGR 靠**外部手工交叉特征**补（动数据组织），FuXi-α 靠**内部 FFN** 补（动层结构）。工程含义不同——MTGR 要求你已有 DLRM 特征资产，FuXi-α 不需要。**如果从零做生成式推荐，FuXi-α 的路径更可复制**（**推断**）。
- **与 C-05 ULTRA-HSTU 是"被超越的前作"**。ULTRA-HSTU 的立论是"self-attention 仍优于 cross-attention + 要线性复杂度"；FuXi-α 提供了"自注意力还能再改进"的证据（三通道拆解），但没解决 O(n²)，SLA 才是解法。两篇连读能看清"表达力"与"复杂度"是两条独立战线。
- **对 NestPipe 的价值**：AMS 的"时间/位置通道不算 Q/K"是**系统级**优化（省两套投影 GEMM），不是纯建模技巧。这也解释了为什么它的 TPS 实验有意义（论文做了吞吐对比，但只给了图，见待核实）。
- **它提供了一个反例**：`x_{b,i} = x_{0,i}·F_{2^b−1}` 的推导说明**堆门控和注意力，交互阶数是指数增长的**。这既是"扩层有效"的理论依据，也暗示了**为什么小数据集上加层反而掉**——论文自己观察到 MovieLens-1M 在 16 层时退化，归因于数据集太小限制了参数扩展（**事实**）。

## 局限

1. **复杂度没解决**。`O(bn²d)` 仍是二次的。序列长到 10k 就崩，而这正是工业推荐的常态（C-05 说 UIH 常超 10k）。FuXi-α 的实验序列长最多 800（KuaiRand 效率实验）和 200（默认）（**事实**）。**这是它作为"大规模"模型最大的名不副实之处**。
2. **"位置/时间通道不算 Q/K"省的是小头**。真正的大头是 `n²d` 的语义通道，没动。
3. **实验规模偏小**。公开数据集最大是 MovieLens-20M；工业集的模型规模未披露具体参数量（**待核实**，正文只说扩到 32 层）。
4. **线上数字前后不一致**：摘要与引言写 "+4.76% / +5.10%"，§4.4 的 A/B 段落写 "4.67%"（**事实**，两处原文并存，疑为笔误）。引用时用 4.76%。
5. **两个未讨论的点**：论文沿用 HSTU 的 `1/n` 归一化，但没讨论极长序列下这个因子会不会让梯度变小（**待核实**）；三通道的桶数 `n_B` 是超参、对时间跨度分布敏感，跨业务迁移要重调（**推断**）。

## 证据与待核实

| 结论 | 等级 | 依据 |
|---|---|---|
| 作者、WWW '25、DOI、代码仓库 | 事实 | 正文标题页 + 摘要 |
| AMS 三通道公式、MFFN 两阶段、`2^b−1` 阶交互证明 | 事实 | §3.1–3.3 |
| 13.24% / 7.26% / +4.76% / +5.10% | 事实 | §4.2、§4.4 |
| SASRec 扩 8 层退化 | 事实 | §4.2 |
| 负样本收益 > 加层收益 | 事实 | §4.3 |
| TPS 具体数值 | **待核实** | 正文有 Table 4（效率对比）但 HTML 抽取时表格数值错乱，未取到可靠读数 |
| 4.67% vs 4.76% 哪个是最终值 | **待核实** | 正文两处不一致 |
| 工业集模型参数量 | **待核实** | 未披露 |

没拿到的东西：Table 4（KuaiRand 上不同序列长的 TPS 对比，HTML 表格解析失败）；WWW '25 正式版 PDF（ACM DL 403）；附录（若与 arXiv 版不同）。

## 是否值得读原文

**挑章节读**。

理由：核心贡献集中在 §3（AMS + MFFN + 表达力证明），而且那部分自洽、可复现、消融到位，值得精读。但 §4 的实验规模（序列长 ≤800、公开集为主）撑不起"大规模"这个词，效率章节也不完整；如果你关心长序列或系统效率，直接跳到 C-05 ULTRA-HSTU 更划算。
