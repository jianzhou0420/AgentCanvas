# Results-Page Style（极简数据页）

Fixed 2026-07-22. Exemplar:
`/data/git_all/info/blackboard/docs-site/pages/teaps/v1-eccv-workshop/results.html`
(Blackboard V0.1 · TEAPS)。适用于任何 "results" 阅读页——展示实验数据的
doc-site / blackboard 页面。

**原则：信息全 · 零分析 · 字数最少。** 表格承载一切；散文只允许口径事实。
页面是真源（`results-data.md` 等）的阅读版，只整理、不再计算、不解读。

## 结构

1. `<h1>`：标题（版本 · 日期 · 用途）。
2. 开头一段 `class="muted"`，全部合为一段：数据来源与提取日期（含补全/修正日期）、
   真源指针、"本页只整理不再计算"、配置/cell 定义指针、版本钉、run window。
3. 正文 = 表格序列（分 Part / 小节，锚点齐全供 TOC 用）。每张表最多配两行散文：
   - **表前一行**（muted）：切法 / 单位 / 口径；必要时含表格没有但 load-bearing 的数字
     （如 thinking 涨幅、context 峰值）。
   - **表后一行**（muted）：†/‡ 脚注与修正记录，格式
     `07-22 修正：X a→b（原为中途快照）`。
4. 纯事实型小节（失败模式、难度谱、运行时）用短句 `<ul>` / muted 段，只留数字。
5. 末节「口径」`<ol>`：每条一行。

## 禁止

- 解读/结论段（pattern 归纳、方向性论断、"结论是…"）——属于论文或 analysis 文档。
- 复述表格列已直接可读的信息。
- 导览段（"本页分两部分…"之类）。

## 惯例

- 所有非表格散文加 `class="muted"`。
- 溯源标记：◆ = 逐集数据本机可核；仅有汇总值的行注明数据所在。
- 修正不抹痕迹：旧值→新值一行留在页内，带日期。
- 中文正文；术语、指标名、cell 名保留英文原文。
