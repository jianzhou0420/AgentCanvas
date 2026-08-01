# Model-Anatomy Style（大厂具身基座解剖页）

Fixed 2026-07-23. Exemplars:
- `/data/git_all/info/blackboard/docs-site/pages/teaps/knowledge/hy-embodied.html`（首例）
- `/data/git_all/info/blackboard/docs-site/pages/teaps/knowledge/qwen-robotnav.html`（含 agentic 层的完整形态，最新范式）

适用于任何大公司具身/导航基座模型的论文解剖页（落位 blackboard
`docs-site/pages/teaps/knowledge/`）。

**原则：机制卡片 · 忠实到句 · 未说明必标。** 页面回答四个问题：它是什么机制、
能力从哪来、agency 在哪一层、和我们（coding-agent / harness）路线什么关系。
一切陈述可回溯到论文原文或仓库原文；论文没写的，**标出来比编出来有价值**。

## 材料（动笔前）

- arXiv **HTML 全文细读**（非摘要、非二手稿）；版本号 + 日期记进 hero meta。
- GitHub README / HF 卡片查证**开源状态**，记查证日期；关键结论引 README 原话。
- 论文原图从 arXiv HTML 下载，压成 JPG（宽 ≤1600px、≲600KB）放
  `docs-site/assets/<slug>/`，文件名带图号（`fig18-agent-real-world.jpg`）。

## 骨架（固定段落，论文没有的层可省，顺序不变）

1. **Hero**：标题 `{Model} 解剖：组件 · 训练 · 数据 (· 系统) · 示例`；
   sub = 一句定位 + 家谱/分类学/姊妹解剖页互链；meta = 整理日期 · 依据
   （arXiv 版本 + GitHub）· 忠实性口径声明（"论文未写明处一律标〔未说明〕；
   示例节图片为论文原图，其余示意图为本页转绘"）。
2. **TL;DR callout**：第一条必须是给模型下的**机制判语**（「单轮多帧 VLM」/
   「可编排的单轮多帧回归 policy」），须能与姊妹页判语直接对话；后续 3–4 条：
   能力来源 / 训练形态 / agency 归属 / 战绩一句。
3. **§01 组件架构**：一张转绘 SVG，输入到输出一条线。论文架构图里的外部组件
   （上层 planner 等）**必须画进来**，虚线框 + "不在权重里" 注明；
   〔未说明〕就地标在图上。图后补控制参数表等紧贴架构的事实表。
4. **§02 底座、数据与训法**：底座表（骨干 / 视觉编码器 / 新增件 /
   **初始化与冻结——必查项**）→ 训练管线 SVG（几段画几段；只有一段也要画，
   供跨页对照）→ 数据表（块 / 规模 / 来源要点；**总量加和自洽核对写在页面上**）
   → 「与本仓最相关的一块」callout（VLN/导航配方，逐条对照姊妹页）。
5. **§03 Agentic / 调用面**（论文有系统层才设）：系统回路 SVG + 战绩速览表。
   战绩每行带口径备注（观测配置、先验、基准版本差异、agentic vs 裸 policy）；
   对模型不利的数字照登。
6. **§示例**：只放第一手材料——论文原图 / 论文原文示例 / 仓库调用例；每件标
   extag（`论文原图` / `仓库原文`）；区分「格式示意」与「真实运行 dump」；
   训练数据可视化 ≠ 推理输出，要点破；节尾写明**这些材料仍然没有展示什么**。
7. **§总结**：固定四问各一段：① 它是什么 ② 能力从哪来 ③ 本模型的特色维度
   （agency 位置 / 导航切片…）④ 与 coding-agent 路线的对照/镜像。
8. **开源状态 callout**：已开源 / 未开源 / 含义 三条，标查证日期，引原话。
9. **Sources**：编号列表，每条写明**从该来源取了哪些内容**；互链横评页
   （vln-r2rce-bigco 系）。

## 忠实性纪律

- 论文未写明 → 〔未说明〕；来源未指名 → 〔未点名〕；amber 高亮
  （散文 `.flag`、SVG `.fw`）。推测必须显式说"按惯例推测，不是论文陈述"。
- 三分材料：论文陈述 / 论文示意 / 本页转绘——不得混写。
- 数字全部回表核对；比例、总量做加和自洽并把核对句写出来。
- 战绩不裸报：每个数字带观测/先验/版本口径。

## 视觉语言

- 复制既有解剖页整个 `<style>` 块，改类前缀（`hy-doc` → `qn-doc` → …）；
  不新造样式体系。
- 色彩语义固定：蓝 = 视觉/观测，紫 = LLM/planner，绿 = 输出/证据，
  amber = 未说明 flag。
- 示意图为手绘 SVG（`.dwrap` 内横向滚动）；`b-dash` 虚线 = 外部组件或备注带。

## 落位与收编

- 页面 `docs-site/pages/teaps/knowledge/<slug>.html`；图 `docs-site/assets/<slug>/`。
- 写完跑 `python3 docs-site/_lib/_wrap_handwritten.py`（blackboard 根下）收编
  chrome + nav + search。
- 中文正文；术语、指标、模型名保留英文原文。
