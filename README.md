# 推荐系统与训练基建

一个 VitePress 文档站，内容分两块：

- **教程**（`docs/guide/`）：13 章，从"推荐系统为什么和 LLM 不一样"讲到"设计清单与决策表"，一条逐步下探的学习路径。
- **论文库**（`docs/papers/nestpipe/`）：50 篇推荐系统基建论文的中文解读，按 A/B/C 三组分类。

## 本地运行

```bash
npm install
npm run dev      # http://localhost:5173
```

构建与预览：

```bash
npm run build    # 产物在 docs/.vitepress/dist/
npm run preview
```

## 校验

教程里每一个「引用某篇论文 + 具体数字」的断言，都能回到该论文页的正文里查证。校验脚本：

```bash
node tools/verify-claims.mjs
```

它会读 `tools/claims.mjs` 里的断言表，逐条到 `docs/papers/nestpipe/` 对应文件里搜。改了教程数字就同步改那张表——**这是这套文档可验证性的唯一来源**。

## 目录

```
docs/
├── index.md                     首页
├── glossary.md                  术语表
├── guide/                       教程 13 章
│   ├── index.md                 课程地图
│   ├── running-example.md       贯穿全教程的算例
│   └── 01..13-*.md              各章
├── papers/nestpipe/             论文库
│   ├── index.md                 导读与阅读路径
│   ├── all-papers.md            50 篇扁平索引
│   └── {A,B,C}-*.md             50 篇单篇解读
└── .vitepress/                  站点配置与主题
tools/
├── claims.mjs                   断言表
└── verify-claims.mjs            核验器
```

## 关于内容可靠性

- 所有加速比、压缩比、参数量来自**论文自报**，**没有一个经过本仓库复现**。
- 50 篇里有 17 篇只有摘要（无 PDF 可得），这些篇目的机制细节在文中标 ⚠️ 并说明哪部分无法确认。
- 教程里的算例数字是**示意性**的，用来展示数量级关系；与论文数字在文中用不同块区分。
- 不同论文的基线、数据集、硬件都不同，**跨论文的数字不可横向排序**。

## 内容来源

论文解读基于 Zotero 集合 **NestPipe**（collection key `9I82QPTL`，50 篇）。教程是按这批论文的适用范围内组织的，超出该范围（如小模型、纯在线学习、深层 Transformer）时结论需重新评估，各章「代价与边界」一节有说明。
