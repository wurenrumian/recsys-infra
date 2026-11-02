import { render } from 'pele'

/**
 * 在构建时用 Pele（https://github.com/obsidianmd/pele）把 ```mermaid 块
 * 渲染成内联 SVG。
 *
 * 为什么不用 mermaid 的浏览器插件：
 * - mermaid 运行时会给每个页面塞进约 1 MB 的客户端 JS，本站只有几张图；
 * - Pele 在构建时出静态 SVG，页面零运行时依赖，深色模式靠 CSS 变量换色。
 */
export function peleMermaid(md: any) {
  const fence = md.renderer.rules.fence

  md.renderer.rules.fence = (
    tokens: any[],
    idx: number,
    options: any,
    env: any,
    self: any,
  ) => {
    const token = tokens[idx]
    if (token.info.trim() !== 'mermaid') {
      return fence(tokens, idx, options, env, self)
    }

    try {
      const { svg } = render(token.content, { fontSize: 14, padding: 20 })
      return `<div class="mermaid-pele">${svg}</div>\n`
    } catch (err: any) {
      // 图渲染失败就中断构建：静默丢图会让人以为「教程里本来就没画」。
      throw new Error(
        `[pele] mermaid 渲染失败（${err?.message ?? err}）\n` +
          `图源：\n${token.content}`,
      )
    }
  }
}
