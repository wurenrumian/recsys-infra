import { defineConfig } from 'vitepress'

export default defineConfig({
  title: '推荐系统与训练基建',
  description: '从嵌入表到生成式推荐：一条可验证的学习路径',

  lang: 'zh-CN',
  srcExclude: ['**/README.md', 'README.md'],

  markdown: {
    lineNumbers: true,
    theme: { light: 'github-light', dark: 'github-dark' },
  },

  themeConfig: {
    logo: undefined,

    nav: [
      { text: '教程', link: '/guide/', activeMatch: '/guide/' },
      { text: '论文库', link: '/papers/nestpipe/', activeMatch: '/papers/' },
      { text: '术语表', link: '/glossary' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: '开始',
          items: [
            { text: '课程地图', link: '/guide/' },
            { text: '贯穿全教程的算例', link: '/guide/running-example' },
          ],
        },
        {
          text: '第一部分 · 看清结构',
          items: [
            { text: '1. 推荐系统为什么和 LLM 不一样', link: '/guide/01-why-different' },
            { text: '2. 嵌入表：九成参数所在的地方', link: '/guide/02-embedding-table' },
            { text: '3. 第一个瓶颈不是算力', link: '/guide/03-bandwidth-wall' },
          ],
        },
        {
          text: '第二部分 · 拆开规模',
          items: [
            { text: '4. 数据并行为什么在这里失效', link: '/guide/04-data-parallel-limits' },
            { text: '5. 切分谱系：表/行/列/副本', link: '/guide/05-sharding-taxonomy' },
            { text: '6. 通信解剖：all-to-all 与两种相反策略', link: '/guide/06-communication' },
          ],
        },
        {
          text: '第三部分 · 下沉层级',
          items: [
            { text: '7. 存储层级：把表放到更慢的地方', link: '/guide/07-storage-hierarchy' },
            { text: '8. 缓存与一致性：允许过期多少', link: '/guide/08-caching-consistency' },
            { text: '9. 压缩的三条轴', link: '/guide/09-compression' },
          ],
        },
        {
          text: '第四部分 · 换掉模型',
          items: [
            { text: '10. 从 DLRM 到生成式推荐', link: '/guide/10-model-evolution' },
            { text: '11. 流水并行与气泡', link: '/guide/11-pipeline-parallel' },
          ],
        },
        {
          text: '收束',
          items: [
            { text: '12. 怎么读一篇新的系统论文', link: '/guide/12-how-to-read' },
            { text: '13. 设计清单与决策表', link: '/guide/13-checklist' },
          ],
        },
      ],

      '/papers/': [
        {
          text: 'NestPipe 论文库',
          items: [
            { text: '导读与阅读路径', link: '/papers/nestpipe/' },
            { text: '50 篇索引', link: '/papers/nestpipe/all-papers' },
          ],
        },
        {
          text: 'A · 规模化训练系统与并行架构',
          collapsed: true,
          items: [
            { text: 'A-01 序列推荐 scaling law', link: '/papers/nestpipe/A-01-RJYMAGIN-scaling-law-sequential-recsys' },
            { text: 'A-02 Climber', link: '/papers/nestpipe/A-02-ABZSE7KJ-climber-efficient-scaling-laws' },
            { text: 'A-03 二维稀疏并行', link: '/papers/nestpipe/A-03-DZZH6L95-two-dimensional-sparse-parallelism' },
            { text: 'A-04 Persia', link: '/papers/nestpipe/A-04-892HASSZ-persia-hybrid-100t-params' },
            { text: 'A-05 层级训练', link: '/papers/nestpipe/A-05-HP588YPM-hierarchical-training-cpu-clusters' },
            { text: 'A-06 NEO 软硬协同', link: '/papers/nestpipe/A-06-J9ZPBVH5-sw-hw-codesign-neo-4d-parallelism' },
            { text: 'A-07 Merlin HugeCTR', link: '/papers/nestpipe/A-07-B2I9ZAIQ-merlin-hugetr-gpu-training-inference' },
            { text: 'A-08 TorchRec', link: '/papers/nestpipe/A-08-4GP3GTPM-torchrec-pytorch-domain-library' },
            { text: 'A-09 PICASSO', link: '/papers/nestpipe/A-09-XGVN6JZA-picasso-gpu-centric-wdl-training' },
            { text: 'A-10 Bagpipe', link: '/papers/nestpipe/A-10-YYFDGRXK-bagpipe-accelerating-dlrm-training' },
            { text: 'A-11 ScratchPipe', link: '/papers/nestpipe/A-11-GYXLW7PP-scratchpipe-look-forward-not-backwards' },
            { text: 'A-12 Hotline', link: '/papers/nestpipe/A-12-9DTGQILE-hotline-heterogeneous-acceleration-pipeline' },
            { text: 'A-13 EcoRec', link: '/papers/nestpipe/A-13-TFTSRALD-ecorec-tt-decomposition-micro-batching' },
            { text: 'A-14 两级有损压缩', link: '/papers/nestpipe/A-14-S996SB9M-dual-level-adaptive-lossy-compression' },
            { text: 'A-15 FusedRec', link: '/papers/nestpipe/A-15-SZH9CX5W-fusedrec-fused-embedding-communication' },
            { text: 'A-16 RecIS', link: '/papers/nestpipe/A-16-ZZ2P8YPT-recis-unified-sparse-dense-framework' },
            { text: 'A-17 LLM 推荐的并行优化', link: '/papers/nestpipe/A-17-3FPR4H9A-llm-recsys-parallelism-optimization' },
          ],
        },
        {
          text: 'B · 嵌入表存储与压缩',
          collapsed: true,
          items: [
            { text: 'B-01 EVStore', link: '/papers/nestpipe/B-01-KP82AUTM-evstore-embedding-storage-cache' },
            { text: 'B-02 HET', link: '/papers/nestpipe/B-02-9XB4UUDV-het-distributed-embedding-cache' },
            { text: 'B-03 CAFE+', link: '/papers/nestpipe/B-03-WBL86RZE-cafe-plus-compact-adaptive-embedding' },
            { text: 'B-04 EMBark', link: '/papers/nestpipe/B-04-JXSQDUAF-embark-embedding-clusters-3d-sharding' },
            { text: 'B-05 AutoShard', link: '/papers/nestpipe/B-05-J7EYKA8M-autoshard-rl-embedding-sharding' },
            { text: 'B-06 NeuroShard', link: '/papers/nestpipe/B-06-WXMZVWVJ-neuroshard-pretrain-and-search' },
            { text: 'B-07 EL-Rec', link: '/papers/nestpipe/B-07-UZF6ZILW-elrec-tensor-train-embedding' },
            { text: 'B-08 UGache', link: '/papers/nestpipe/B-08-6NY24IFU-ugache-unified-multi-gpu-cache' },
            { text: 'B-09 层级 GPU 参数服务器', link: '/papers/nestpipe/B-09-SLPPGD8C-hierarchical-gpu-parameter-server' },
            { text: 'B-10 混合精度嵌入', link: '/papers/nestpipe/B-10-DH7ZFV2C-mpe-mixed-precision-embeddings' },
            { text: 'B-11 DQRM', link: '/papers/nestpipe/B-11-Z56XJYUC-dqrm-deep-quantized-recommendation' },
            { text: 'B-12 AdaEmbed', link: '/papers/nestpipe/B-12-3K9KISR6-adaembed-adaptive-embedding-pruning' },
            { text: 'B-13 OPER', link: '/papers/nestpipe/B-13-37Q3PZXS-oper-optimality-guided-parallelization' },
            { text: 'B-14 Herald', link: '/papers/nestpipe/B-14-6HTV2P4W-herald-embedding-scheduling' },
            { text: 'B-15 HypeReca', link: '/papers/nestpipe/B-15-VE9T6R4E-hypereca-in-memory-embedding-database' },
            { text: 'B-16 Pinterest 多 epoch', link: '/papers/nestpipe/B-16-L2VVL9ZT-pinterest-embedding-multi-epoch' },
            { text: 'B-17 DCMA', link: '/papers/nestpipe/B-17-WCV8CKED-dcma-multi-port-cached-dma' },
          ],
        },
        {
          text: 'C · 模型演进与调度',
          collapsed: true,
          items: [
            { text: 'C-01 生成式推荐综述', link: '/papers/nestpipe/C-01-GMPG7G9I-generative-recsys-progress-directions' },
            { text: 'C-02 HSTU', link: '/papers/nestpipe/C-02-G73SFXWD-hstu-trillion-param-generative-recsys' },
            { text: 'C-03 MTGR', link: '/papers/nestpipe/C-03-DDWBU7H3-mtgr-meituan-generative-recsys' },
            { text: 'C-04 FuXi-α', link: '/papers/nestpipe/C-04-AXNSLSLL-fuxi-alpha-feature-interaction-transformer' },
            { text: 'C-05 ULTRA-HSTU', link: '/papers/nestpipe/C-05-ASKWXRVH-ultra-hstu-bending-scaling-law' },
            { text: 'C-06 Disaggregated Multi-Tower', link: '/papers/nestpipe/C-06-NF8E5IDY-dmt-disaggregated-multi-tower' },
            { text: 'C-07 KuaiRand', link: '/papers/nestpipe/C-07-FCJDT7DA-kuairand-unbiased-dataset' },
            { text: 'C-08 SlimPipe', link: '/papers/nestpipe/C-08-9MCTBGR4-slimpipe-memory-thrifty-pp' },
            { text: 'C-09 TD-Pipe', link: '/papers/nestpipe/C-09-2JX9YAFY-td-pipe-temporal-disaggregated-pp' },
            { text: 'C-10 OctoPipe', link: '/papers/nestpipe/C-10-UFFU6KVA-octopipe-heterogeneous-bubbles' },
            { text: 'C-11 Basis Rotation', link: '/papers/nestpipe/C-11-W5IVJUAZ-basis-rotation-staleness' },
            { text: 'C-12 ODC', link: '/papers/nestpipe/C-12-ZLR452K5-odc-revisiting-parameter-server' },
            { text: 'C-13 Comet', link: '/papers/nestpipe/C-13-CAC4AVZQ-comet-moe-overlap' },
            { text: 'C-14 GBA', link: '/papers/nestpipe/C-14-X7YZ329D-gba-tuning-free-sync-async' },
            { text: 'C-15 Engram', link: '/papers/nestpipe/C-15-A8BIBZTJ-engram-conditional-memory' },
            { text: 'C-16 MPI-CUDA 基准', link: '/papers/nestpipe/C-16-ED6MKDZR-hybrid-mpi-cuda-benchmark-suite' },
          ],
        },
      ],
    },

    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一章', next: '下一章' },
    darkModeSwitchLabel: '深色模式',
    sidebarMenuLabel: '目录',
    returnToTopLabel: '回到顶部',
    lastUpdated: { text: '最后更新于' },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '没有找到结果',
            resetButtonTitle: '清除',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },

    socialLinks: [],
    footer: {
      message: '内容基于 NestPipe 论文集合（Zotero collection <code>9I82QPTL</code>，50 篇）整理',
      copyright: '教程文本为学习用途整理；论文结论与数字归原作者所有',
    },
  },
})
