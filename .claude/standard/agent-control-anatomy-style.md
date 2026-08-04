# Agent-Control-Anatomy Style（agent 控机器人论文解剖页）

Fixed 2026-08-03. Exemplar:
- `/data/git_all/info/blackboard/docs-site/pages/manipulation/anatomy/faea.html`（首例）

适用于"agent 写代码/发指令控制机器人"一族论文的解剖页（FAEA、RHO、ENPIRE、
ASPIRE、ALRM…），落位 blackboard **Manipulation tab** 的
`docs-site/pages/manipulation/anatomy/`，sidebar section
「Anatomy · Agent Control」（注意：不进 General tab 的 anatomy 目录——那里是
VLN 两族的地盘）。是 [model-anatomy-style](model-anatomy-style.md)
的姊妹 standard：VLN 基座解剖的核心轴是"能力从哪来 = 底座×数据×训法"，这族
论文训练往往缺位，能力来自 **harness = 模型×框架×工具面×接口**，故换轴；
编号骨架对齐，让两族页面判语能互相对话。

**原则：机制卡片 · 忠实到句 · 未说明必标**（与 model-anatomy 相同）。页面回答
四个问题：它是什么机制、agency 在哪层且改进的是什么、接口与口径的特色、
和我们（coding-agent / harness）路线什么关系。

## 材料（动笔前）

- arXiv **HTML 全文细读**（非摘要、非二手稿）；版本号 + 日期记进 hero meta。
- GitHub README / 项目页查证**开源状态**，记查证日期；关键结论引 README 原话。
- 论文原图从 arXiv HTML 下载，压成 JPG（宽 ≤1600px、≲600KB）放
  `docs-site/assets/<slug>/`，文件名带图号。

## 骨架（固定段落，论文没有的层可省，顺序不变）

1. **Hero**：标题 `{Work} 解剖：接口 · Harness · 回路 (· 战绩) · 示例`；
   sub = 一句定位 + 姊妹解剖页互链；meta = 整理日期 · 依据（arXiv 版本 +
   GitHub 查证日期）· 忠实性口径声明（同 model-anatomy）。
2. **TL;DR callout**：第一条必须是**机制判语**——本族词表：「test-time 程序
   合成」/「训练时 harness 搜索」/「真机闭环 policy 自改进」/「持续技能积累」…
   须能与姊妹页判语直接对话；后续 3 条：改进归属一句 / 接口一句 /
   战绩+成本一句。
3. **§01 Agent–Robot 接口**：转绘 SVG 一条线「观测面 → agent → 动作面」。
   观测（特权状态 / RGB / 文本化，谁做的转换）、动作（力矩 / EE 笛卡尔 /
   速度指令 / VLA 监督 / 写代码调 API）、控制频率与实时性——做成表。
   〔未说明〕就地标在图上。
4. **§02 Harness 构成**：基座模型表（**版本与冻结与否——必查项**）/ agent
   框架（现成 SDK 还是自研 loop，SDK 自带什么）/ 工具面表 / prompt 结构 /
   上下文管理策略。
5. **§03 反馈回路与改进归属**：回路 SVG——success 判定从哪来（仿真 oracle /
   真机验证 / 人）、失败信息怎么回流（报错 / 轨迹 / 视频）、重试与终止策略、
   跨任务积累。**明确"被改进的对象"**：脚本 / harness 仓库 / policy 权重 /
   skill library 哪一个，有无梯度。
6. **§04 战绩与成本**：每行带口径（特权与否、对照 VLA 的 demo 数、benchmark
   版本差异），加**成本列**（$/任务、尝试数、token、墙钟）；对模型不利的
   数字照登；有反作弊/轨迹审计的照登。
7. **§示例**：只放第一手材料——prompt 原文 / agent trace / 论文原图；每件标
   extag（`论文原图` / `论文原文` / `仓库原文`）；区分「格式示意」与「真实
   运行 dump」；节尾写明**这些材料仍然没有展示什么**。
8. **§总结**：固定四问各一段：① 它是什么机制 ② agency 在哪层、改进的是什么
   ③ 接口与口径的特色 ④ 与我们 coding-agent / harness 路线的对照或镜像。
9. **开源状态 callout**：已开源 / 未开源 / 含义 三条，标查证日期，引原话。
10. **Sources**：编号列表，每条写明**从该来源取了哪些内容**；互链
    manipulation tab 的材料页与姊妹解剖页。

## 忠实性纪律（与 model-anatomy 相同）

- 论文未写明 → 〔未说明〕；来源未指名 → 〔未点名〕；amber 高亮
  （散文 `.flag`、SVG `.fw`）。推测必须显式说"按惯例推测，不是论文陈述"。
- 三分材料：论文陈述 / 论文示意 / 本页转绘——不得混写。
- 数字全部回表核对；比例、总量做加和自洽并把核对句写出来。
- 战绩不裸报：每个数字带观测/先验/成本口径。

## 视觉语言（与 model-anatomy 相同）

- 复制既有解剖页整个 `<style>` 块，改类前缀（`hy-doc` → `fa-doc` → …）；
  不新造样式体系。
- 色彩语义固定：蓝 = 观测/环境，紫 = agent/LLM，绿 = 输出/证据，
  amber = 未说明 flag。
- 示意图为手绘 SVG（`.dwrap` 内横向滚动）；`b-dash` 虚线 = 外部组件或备注带。

## 落位与收编

- 页面 `docs-site/pages/manipulation/anatomy/<slug>.html`；图 `docs-site/assets/<slug>/`。
- nav：`pages/manipulation/_tab.json` 的「Anatomy · Agent Control」section
  （slug 加进 `order`）。
- 写完跑 `python3 docs-site/_lib/_wrap_handwritten.py`（blackboard 根下）收编
  chrome + nav + search。
- 中文正文；术语、指标、模型名保留英文原文。
