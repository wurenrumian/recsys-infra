/**
 * 教程中「引用论文 + 具体数字」的断言表。
 *
 * 规则：
 * - 每一条都必须是教程里真的写出来的断言，且数字必须能在对应论文页找到。
 * - needle 用论文页里的**原始写法**（含加粗也没关系，核验器会做变体匹配）。
 * - 加一条断言前，先确认论文页里真有这个数字；找不到就把教程里的说法改掉，
 *   或标为「待核实」并从本表移除。
 *
 * 这个表是本教程可验证性的唯一来源。改教程数字时同步改这里。
 */
export const CLAIMS = [
  // ── 第 1 章：两个系统 ──
  { chapter: '01-why-different', paper: 'A-10', needle: '1% 嵌入承担 >92% 访问' },
  { chapter: '01-why-different', paper: 'A-10', needle: '0.1% 承担近 90%' },
  { chapter: '01-why-different', paper: 'A-03', needle: '5.70 / 5.18' },
  { chapter: '01-why-different', paper: 'A-03', needle: '1.57 / 1.65' },
  { chapter: '01-why-different', paper: 'A-16', needle: '47.50%' },
  { chapter: '01-why-different', paper: 'A-16', needle: '0.88%' },
  { chapter: '01-why-different', paper: 'A-16', needle: '58.00%' },
  { chapter: '01-why-different', paper: 'B-02', needle: '79.64' },
  { chapter: '01-why-different', paper: 'B-02', needle: '74.61' },

  // ── 第 2 章：嵌入表 ──
  { chapter: '02-embedding-table', paper: 'A-16', needle: '55.10%' },
  { chapter: '02-embedding-table', paper: 'A-16', needle: '34.60%' },

  // ── 第 3 章：三堵墙 ──
  { chapter: '03-bandwidth-wall', paper: 'A-03', needle: '79.7%' },
  { chapter: '03-bandwidth-wall', paper: 'A-03', needle: '90%' },
  { chapter: '03-bandwidth-wall', paper: 'B-04', needle: '25%' },
  { chapter: '03-bandwidth-wall', paper: 'B-04', needle: '51%' },
  { chapter: '03-bandwidth-wall', paper: 'A-10', needle: '75%' },

  // ── 第 4 章：数据并行失效 ──
  { chapter: '04-data-parallel-limits', paper: 'A-03', needle: '5.70' },
  { chapter: '04-data-parallel-limits', paper: 'A-03', needle: '1.57' },
  { chapter: '04-data-parallel-limits', paper: 'A-03', needle: '0.1%' },
  { chapter: '04-data-parallel-limits', paper: 'A-03', needle: '0.02%' },
  { chapter: '04-data-parallel-limits', paper: 'A-03', needle: '1024' },

  // ── 第 5 章：切分谱系 ──
  { chapter: '05-sharding-taxonomy', paper: 'B-05', needle: '88.6%' },
  { chapter: '05-sharding-taxonomy', paper: 'B-06', needle: '+23.8%' },
  { chapter: '05-sharding-taxonomy', paper: 'B-06', needle: '+6.6%' },
  { chapter: '05-sharding-taxonomy', paper: 'B-13', needle: '2.3×' },
  { chapter: '05-sharding-taxonomy', paper: 'B-13', needle: '4.0×' },
  { chapter: '05-sharding-taxonomy', paper: 'B-04', needle: '1.5×' },

  // ── 第 6 章：通信 ──
  { chapter: '06-communication', paper: 'A-14', needle: '19.9×' },
  { chapter: '06-communication', paper: 'A-14', needle: '31.3%' },
  { chapter: '06-communication', paper: 'A-14', needle: '5.03%' },
  { chapter: '06-communication', paper: 'A-14', needle: '8.6×' },
  { chapter: '06-communication', paper: 'A-15', needle: '37.8%' },

  // ── 第 7 章：存储层级 ──
  { chapter: '07-storage-hierarchy', paper: 'B-09', needle: '46%' },
  { chapter: '07-storage-hierarchy', paper: 'B-09', needle: '4 台机器快过 150 台' },
  { chapter: '07-storage-hierarchy', paper: 'B-09', needle: '10TB' },
  { chapter: '07-storage-hierarchy', paper: 'B-09', needle: '0.1%' },
  { chapter: '07-storage-hierarchy', paper: 'B-01', needle: '94%' },
  { chapter: '07-storage-hierarchy', paper: 'B-08', needle: '1.93×' },
  { chapter: '07-storage-hierarchy', paper: 'B-15', needle: '2.16' },
  { chapter: '07-storage-hierarchy', paper: 'B-03', needle: '3.94%' },
  { chapter: '07-storage-hierarchy', paper: 'A-10', needle: '5.6×' },
  { chapter: '07-storage-hierarchy', paper: 'A-10', needle: '75%' },

  // ── 第 8 章：缓存与一致性 ──
  { chapter: '08-caching-consistency', paper: 'B-02', needle: '88%' },
  { chapter: '08-caching-consistency', paper: 'B-02', needle: '79.62' },
  { chapter: '08-caching-consistency', paper: 'B-02', needle: '78.84' },
  { chapter: '08-caching-consistency', paper: 'B-02', needle: '70.91' },
  { chapter: '08-caching-consistency', paper: 'A-10', needle: '92%' },
  { chapter: '08-caching-consistency', paper: 'A-10', needle: '91%' },
  { chapter: '08-caching-consistency', paper: 'A-10', needle: '82%' },
  { chapter: '08-caching-consistency', paper: 'A-11', needle: '6.6×' },
  { chapter: '08-caching-consistency', paper: 'A-11', needle: '5.1×' },
  { chapter: '08-caching-consistency', paper: 'A-11', needle: 'RAW-4' },

  // ── 第 9 章：压缩三轴 ──
  { chapter: '09-compression', paper: 'B-10', needle: '200' },
  { chapter: '09-compression', paper: 'B-11', needle: '8×' },
  { chapter: '09-compression', paper: 'B-11', needle: '61ms' },
  { chapter: '09-compression', paper: 'B-11', needle: '110ms' },
  { chapter: '09-compression', paper: 'B-12', needle: '35–60%' },
  { chapter: '09-compression', paper: 'B-07', needle: '没有"行"这个概念了' },
  { chapter: '09-compression', paper: 'B-07', needle: '3×' },
  { chapter: '09-compression', paper: 'B-03', needle: '3.94%' },
  { chapter: '09-compression', paper: 'A-13', needle: '3.1×' },
  { chapter: '09-compression', paper: 'B-12', needle: 'binary case of mixed-precision embeddings' },

  // ── 第 10 章：模型演进 ──
  { chapter: '10-model-evolution', paper: 'C-05', needle: 'scaling efficiency' },
  { chapter: '10-model-evolution', paper: 'C-05', needle: '5×' },
  { chapter: '10-model-evolution', paper: 'C-05', needle: '21×' },
  { chapter: '10-model-evolution', paper: 'C-05', needle: '4×' },
  { chapter: '10-model-evolution', paper: 'C-02', needle: 'M-FALCON' },
  { chapter: '10-model-evolution', paper: 'A-01', needle: '0.8B' },

  // ── 第 11 章：流水并行 ──
  { chapter: '11-pipeline-parallel', paper: 'C-11', needle: '81.7%' },
  { chapter: '11-pipeline-parallel', paper: 'C-11', needle: 'scales linearly with pipeline depth' },
  { chapter: '11-pipeline-parallel', paper: 'C-12', needle: 'balanced workload assumption no longer holds' },
  { chapter: '11-pipeline-parallel', paper: 'C-08', needle: 'SlimPipe' },
  { chapter: '11-pipeline-parallel', paper: 'C-09', needle: 'prefill' },

  // ── 第 12 章：怎么读论文 ──
  { chapter: '12-how-to-read', paper: 'A-14', needle: '8.6×' },
  { chapter: '12-how-to-read', paper: 'A-14', needle: '1.38×' },
  { chapter: '12-how-to-read', paper: 'A-14', needle: '31.3%' },
  { chapter: '12-how-to-read', paper: 'A-10', needle: '3.7×' },
  { chapter: '12-how-to-read', paper: 'A-10', needle: '5.6×' },
  { chapter: '12-how-to-read', paper: 'A-10', needle: '13 分钟' },
  { chapter: '12-how-to-read', paper: 'A-16', needle: '带宽 roofline' },
  { chapter: '12-how-to-read', paper: 'B-11', needle: '61ms' },
  { chapter: '12-how-to-read', paper: 'B-02', needle: '79.64' },

  // ── 第 13 章：清单与决策表 ──
  { chapter: '13-checklist', paper: 'A-06', needle: '96TB' },
  { chapter: '13-checklist', paper: 'A-06', needle: '24TB' },
  { chapter: '13-checklist', paper: 'B-11', needle: '110ms' },
  { chapter: '13-checklist', paper: 'A-10', needle: '82%' },
  { chapter: '13-checklist', paper: 'B-07', needle: '221' },
  { chapter: '13-checklist', paper: 'B-07', needle: '0.3%' },
  { chapter: '13-checklist', paper: 'A-03', needle: '1.57' },
]
