#!/usr/bin/env node
/**
 * 教程数字核验器
 *
 * 作用：把教程里「引用某篇论文 + 某个数字」的断言，逐条回到该论文页的正文里查证。
 * 目的：防止教程在二次创作时编造或记错数字。
 *
 * 用法：
 *   node tools/verify-claims.mjs          # 全部检查
 *   node tools/verify-claims.mjs --quiet  # 只输出失败项
 *
 * 断言来源：claims.mjs 里的表。加新断言就往那张表里加一行。
 */

import { readFileSync, existsSync, readdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { CLAIMS } from './claims.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const ROOT = join(__dirname, '..')
const PAPERS = join(ROOT, 'docs', 'papers', 'nestpipe')
const GUIDE = join(ROOT, 'docs', 'guide')

const quiet = process.argv.includes('--quiet')

/** 找到某篇论文对应的文件（按 A-01 / B-07 / C-12 前缀） */
function paperFile(id) {
  const f = readdirSync(PAPERS).find((n) => n.startsWith(id + '-'))
  return f ? join(PAPERS, f) : null
}

/** 把断言里可能出现的排版变体都试一遍（加粗、空格、全角） */
function variants(needle) {
  const out = new Set([needle])
  // 去掉加粗标记
  out.add(needle.replace(/\*\*/g, ''))
  // 去掉所有空白
  out.add(needle.replace(/\s+/g, ''))
  // 全角转半角
  out.add(needle.replace(/：/g, ':').replace(/（/g, '(').replace(/）/g, ')'))
  return [...out]
}

let pass = 0
const failures = []
const missingPapers = new Set()

for (const c of CLAIMS) {
  const file = paperFile(c.paper)
  if (!file || !existsSync(file)) {
    missingPapers.add(c.paper)
    failures.push({ ...c, reason: `找不到论文文件 ${c.paper}` })
    continue
  }

  const text = readFileSync(file, 'utf8')
  const flat = text.replace(/\s+/g, '')
  const hit = variants(c.needle).some(
    (v) => text.includes(v) || flat.includes(v.replace(/\s+/g, '')),
  )

  if (hit) {
    pass++
  } else {
    failures.push({ ...c, reason: `论文页里找不到 "${c.needle}"` })
  }
}

// 同时检查：教程章节里出现该论文编号时，文件确实存在
const guideFiles = readdirSync(GUIDE).filter((n) => n.endsWith('.md'))
const cited = new Set()
for (const f of guideFiles) {
  const t = readFileSync(join(GUIDE, f), 'utf8')
  for (const m of t.matchAll(/\b([ABC])-(\d{2})\b/g)) {
    cited.add(`${m[1]}-${m[2]}`)
  }
}

for (const id of cited) {
  if (!paperFile(id)) {
    failures.push({ chapter: '(多处)', paper: id, needle: '(引用)', reason: '教程引用了不存在的论文编号' })
  }
}

if (!quiet) {
  console.log(`断言总数：${CLAIMS.length}`)
  console.log(`通过：${pass}`)
  console.log(`失败：${failures.length}`)
  console.log(`教程引用的论文编号：${cited.size} 个`)
}

if (failures.length) {
  console.log('\n失败明细：')
  for (const f of failures) {
    const where = f.chapter ? `[${f.chapter}] ` : ''
    console.log(`  ✗ ${where}${f.paper} — ${f.reason}`)
  }
  process.exit(1)
}

if (!quiet) console.log('\n✓ 全部断言都能在论文页里找到')
