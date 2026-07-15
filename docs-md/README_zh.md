[English](../README.md) | **中文** | [Español](README_es.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

<div align="center">

# AgentCanvas

### Automating the Design of Embodied Agent Architectures

**Jian Zhou · Sihao Lin · Jin Li · Shuai Fu · Gengze Zhou · Qi Wu**

Australian Institute for Machine Learning, University of Adelaide

<p>
  <a href="https://arxiv.org/abs/2606.30111"><img src="https://img.shields.io/badge/arXiv-2606.30111-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://jianzhou0420.github.io/src/works/AgentCanvas/index.html"><img src="https://img.shields.io/badge/Project%20Page-1f6feb?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://jianzhou0420.github.io/src/works/AgentCanvas/paper.html"><img src="https://img.shields.io/badge/Paper%20Page-1f6feb?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Paper Page"></a>
  <a href="https://jianzhou0420.github.io/AgentCanvas/"><img src="https://img.shields.io/badge/Docs-2ea44f?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"></a>
  <a href="#6-引用"><img src="https://img.shields.io/badge/BibTeX-Cite-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white" alt="BibTeX"></a>
</p>

<img src="../assets/readme/editor-hero.gif" alt="AgentCanvas 编辑器：MapGPT executor 以节点-连线图的形式加载，随后一个真实的 R2R episode 端到端运行" width="760">

<sub><em>在编辑器中实时录制 —— MapGPT executor 加载完成，随后一个真实的 R2R episode 端到端运行。</em></sub>

</div>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Status: Research Preview](https://img.shields.io/badge/Status-Research_Preview-orange.svg)](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html)
[![GitHub stars](https://img.shields.io/github/stars/jianzhou0420/AgentCanvas?style=social)](https://github.com/jianzhou0420/AgentCanvas/stargazers)

**面向具身 AI 研究的可视化智能体设计平台。** 一张类型化的图，两种角色：既是运行具身智能体的*运行框架（harness）*，也是供编程智能体（coding agent）编辑与验证的*脚手架（scaffold）*。

AgentCanvas 让研究者通过绘制节点图来快速搭建具身智能体 —— 面向 VLN、EQA、VLA 及相邻任务 —— 这些图可以实时地在仿真器（Habitat-Sim、MatterSim、SAPIEN/ManiSkill2、MuJoCo/robosuite）上执行，原则上也可在真实世界的配置上执行。*一个 JSON = 一个智能体 = 一张图*：智能体的行为是一张数据流图，而非命令式代码；图就是唯一的真相来源，保存为单个 JSON 文件，并作为一个完整的智能体加载。

**为谁打造**：希望组合、比较并分享具身智能体架构，又不想每次都重写执行栈的研究者。该平台覆盖 VLN（视觉语言导航，Vision-and-Language Navigation）、EQA（具身问答，Embodied Question Answering）、VLA（视觉-语言-动作，Vision-Language-Action）策略基准，并通过 nodeset（节点集）模型适配其他具身 / 智能体场景。

> **状态**：研究预览，1.0 之前 —— 横跨四类可互换面板（palette）的 40+ 个 nodeset（**env** · **method** · **model** · **policy**）；公共 API 尚未冻结（[版本管理策略](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html)）。

> **贡献**：nodeset、图与核心 PR 皆受欢迎 —— 每一份贡献都会记入[致谢](#致谢)榜。参见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

---

## 最新动态！

- [2026/07] 🚀 **Graph SDK —— 用 Python 构建并运行智能体** —— 同样的画布图，如今成了一个可导入的库：`from agentcanvas import Graph`，添加/连接节点，在进程内运行与批量评估，或把一张图编译回一个独立的构建脚本。同一份 `GraphDefinition`，与画布 + JSON 完全可逆。参见 [Graph SDK 文档](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/capabilities/graph-sdk.html)。
- [2026/07] 🎥 **pySLAM 经典 SLAM 演示** —— pySLAM 在 TUM RGB-D 上唱主角：一个流式重放环境把一个基准序列逐帧喂进一个实时 SLAM 会话 —— 估计出的相机轨迹自顶向下地拟合到真值上，一张稀疏 3-D 地图实时地稠密化，没有仿真器也没有策略，纯 CPU。完整片段 + 讲解见 [pySLAM nodeset 文档](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-pyslam.html)。

  [![pySLAM 在 TUM RGB-D 上的流式 SLAM —— 实时相机轨迹对比真值，一张 3-D 地图实时稠密化，随后环绕展示完成的地图](../docs/assets/videos/pyslam-tum-slam-demo.gif)](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-pyslam.html)
- [2026/07] 🔥 **更广的基础模型支持** —— 现已有 29 个基础模型作为轻薄的 server-mode 外壳接入（transformers-native + 其他来源），同时供手工构建的图与 AAS optimizer 使用：近期的 VLM（Qwen3-VL、InternVL3、Gemma 3、SmolVLM2）、开放词表感知（SigLIP2、OWLv2、Grounding DINO），以及几何 / 深度骨干网络。参见[基础模型覆盖范围](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/index.html)与逐模型的 [Credits](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html)。
- [2026/07] 🔥 **在画布上编辑节点源码** —— 新的 Source 标签页会展示所选节点在其 nodeset 源码中的作用域切片（全局变量、被引用的函数、类本身），并以带语法检查的热重载把编辑拼接回去。PR：[#5](https://github.com/jianzhou0420/AgentCanvas/pull/5)。
- [2026/07] 🎉 **首个公开发布** —— AgentCanvas 作为研究预览（1.0 之前）开源。文档：[jianzhou0420.github.io/AgentCanvas](https://jianzhou0420.github.io/AgentCanvas/)。

---

## 目录

1. [为什么选择 AgentCanvas?](#1-为什么选择-agentcanvas) —— 一个面向具身智能体的可搜索基底，以及它要解决的痛点
2. [功能特性](#2-功能特性) —— *一个 JSON = 一个智能体*（§2.2）/ *一个 Python 类 = 一个节点*（§2.6）原则，外加画布编辑器、图执行引擎、隔离的运行时环境、嵌套图、状态容器、hook
3. [从仿真到真机的路径](#3-从仿真到真机的路径) —— 同一张智能体图，今天跑仿真，明天上真实机器人 —— 通过 env-as-nodeset + server mode + ROS
4. [快速开始](#4-快速开始) —— 前置条件、运行 Web 仪表盘、运行评估、运行架构搜索、本地服务文档
5. [贡献](#5-贡献) —— 最需要帮助的地方 · 致谢
6. [引用](#6-引用) —— 引用 AgentCanvas 论文
7. [许可证](#7-许可证) —— Apache 2.0

---

## 1. 为什么选择 AgentCanvas?

具身智能体 —— 横跨 VLN、EQA 与 VLA —— 越来越多地由基础模型与感知、建图、记忆、规划、动作等模块组合而成。与端到端策略（其结构被吸收进权重之中）不同，这种架构是*显式且可编辑的*。这就引出了 AgentCanvas 所围绕的问题 —— **智能体设计能否被搜索，而不是手工搭建?** —— 以及它在此过程中必须扫清的两摞痛点。

<details>
<summary><b>智能体架构是手工搭建的 —— 而它本可以被搜索</b></summary>

<br>

每个智能体都在每一个连接点上手工固定了一个选择 —— 传感器抽象、地图表示、记忆状态、提示词结构、规划器拓扑、模型放置、动作接口 —— 通常只针对单一基准。随着基础模型与具身工具不断增多，这个空间的增长速度超过了手工迭代所能覆盖的范围，于是自然的做法是去搜索它，而不是手工调它。

智能体架构搜索（AAS）在文本领域的智能体上已经做到了这一点，但迁移到具身场景并非免费午餐：有状态的仿真器、带噪声的多 episode 打分、漫长的感知/动作轨迹，以及没有现成的具身原语面板。AgentCanvas 是我们为补上这块缺失基底所做的尝试 —— 一个编程智能体能够读取、编辑、运行并验证的脚手架 —— 让搜索智能体设计对具身智能体也成为可能。

</details>

<details>
<summary><b>具身领域特有的痛点</b></summary>

<br>

- **现代具身技术栈很厚** —— 一个能用的具身智能体需要 LLM 推理 + 工具使用 + 仿真器耦合 + 空间工具，全部连在一起。每个项目都从零搭建这一切，成本高得令人却步，而且大部分精力都花在执行层上，而非要验证的想法本身。
- **工程噩梦** —— 具身智能体不是单个模型，而是一整套系统 —— 一个有状态的仿真器，加上一摞笨重的模型和工具。光是把它跑起来（更不用说在基准测试所需的规模上跑）本身就是一项艰难的工程：
  - **Python 环境地狱** —— 没有任何单一的 Python 环境能满足每一个部分；每个仿真器、VLM、检测器、策略都各自钉死了相互冲突的 CUDA / torch / Python，因此找到一个它们都能共享的运行时往往是不可能的 —— 你最终不得不维护好几个互不兼容的环境，仅仅是为了把智能体加载起来。
  - **批处理** —— 每个 worker 的仿真器都是一个独立的有状态进程，各自按自己的节奏推进；你可以对模型做批处理，却无法对仿真器做批处理，于是每一步都变成了一支异步的「收集观测 → 批量推理 → 分发动作」之舞。
  - **其他基础设施** —— 必须被记录且可重放的多模态轨迹、对*必然*会崩溃的数小时 GPU 运行做检查点/续跑，以及跨进程边界的调试。

  在一篇论文的整个研究周期里，研究者把太多成本花在了这类工程上，而不是聚焦于算法本身。
- **隐藏的真值依赖** —— 许多方法悄悄地依赖仿真器提供的真值（物体位姿、语义标签、可导航性），而非真实感知。有时这是控制实验的正当手段 —— 但无论是有意还是疏忽，它往往在论文里只字未提。

</details>

<details>
<summary><b>AI 研究的通用痛点（在这里被放大）</b></summary>

<br>

- **实现无法复现** —— 每篇论文都用不同的代码库从零搭建自己的智能体；公平地比较方法或复现结果都很痛苦 —— 而且其中很多还停留在 **`Code coming SOON`**（**S**omeday, **O**r **O**bviously **N**ever —— 「总有一天，或者显然永远不会」）。
- **论文 ≠ 代码** —— 论文展示干净的流程图，但实际代码以未记录的方式偏离。复现一篇论文意味着对其实现做逆向工程。
- **代码高度耦合** —— 领域逻辑（提示词、工具、策略）与基础设施纠缠在一起。替换一个组件就意味着重写整条流水线。

</details>

---

## 2. 功能特性

> **完整参考见文档** —— 下面大多数功能都有对应的实现页面（机制 · 关键文件 · 当前状态）：**[九大能力 →](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/capabilities/index.html)**

<details>
<summary><b>九大能力</b> —— 画布编辑器 · 图引擎 · 隔离运行时 · 嵌套图 · 状态容器 · Python 定义的节点 · hook · 批量评估 · 可观测性</summary>

<br>

### 2.1 可视化画布编辑器

一个 ComfyUI 风格的扁平工作区，所有节点类型在此共存 —— 环境、LLM、推理链、控制门、输出查看器。从侧边栏拖出节点，把它们连起来，按下 Play。

### 2.2 图执行引擎

**一个 JSON = 一个智能体。** 一个智能体的全部行为 —— 节点、连线、配置、状态容器、hook —— 就是单个 JSON 文件：加载它、运行它、分享它、diff 它。没有隐藏的流水线代码；你在画布上看到的就是实际执行的。

```jsonc
// Simplified — real graphs include state containers, hooks, and more nodes
{
  "name": "NavGPT-CE",
  "description": "VLN reasoning graph with planner, VLM, and navigation memory",
  "kind": "graph",
  "nodes": [
    { "id": "observe", "type": "env_habitat__observe_egocentric", "config": {} },
    { "id": "planner", "type": "llmCall",                         "config": { "temperature": 0.0 } },
    { "id": "step",    "type": "env_habitat__step_discrete",      "config": {} }
  ],
  "edges": [
    { "source": "observe", "sourceHandle": "rgb", "target": "planner", "targetHandle": "image" },
    { "source": "planner", "sourceHandle": "action", "target": "step", "targetHandle": "action" }
  ]
}
```

随后引擎运行那张图：节点在其输入到达时触发，而非按固定顺序。同一个引擎能处理 AgentCanvas v1 支持的所有图形态 —— 完整的、被 v1 有界静态拓扑范式所覆盖的智能体形态清单见 [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html)：

- **DAG 工作流** —— 无环流水线的单次前向传递
- **有环智能体循环** —— 通过一个**双枢轴**模型实现「观察-思考-行动-重复」：一个双侧的 **`IterIn`**（左侧是运行开始时的初始化输入，右侧是逐迭代的循环携带值）加上 **`IterOut`**，在让图在视觉上保持无环的同时启用运行时的环（ADR-dataflow-008，它把 ADR-dataflow-006 早先的三枢轴 `initialize`/IterIn/IterOut 折叠为两个）
- **多作用域迭代** —— 在一张扁平图中并存 N 对 `(IterIn, IterOut)`（ADR-dataflow-007 / ADR-executor-003）
- **ReAct 循环** —— 既可以隐藏在 `LLMCallNode` 子类内部，也可以显式表达为路由器 + N 个预先声明的工具分支
- **有界多智能体** —— 固定 N 或受 `K_max` 约束的扇出（例如 DiscussNav 式辩论、AutoGen 式固定角色）
- **Plan-and-Execute** —— 在一个有界工具池上，由路由器分派

引擎也可以在不触碰图节点的前提下扩展：shell hook 会在每个节点执行的前/后以及图的生命周期边界上触发 —— 记录输出、校验输入、阻断节点或修改数据 —— 并随保存的图一起携带。

### 2.3 隔离的运行时环境

研究工具常常需要相互冲突的 Python 环境（Habitat 需要 Python 3.8，SLAM 需要 ROS）。任何 `BaseNodeSet` 都能以 **server mode** 运行 —— 框架会根据该 nodeset 的端口定义自动生成一个 HTTP 服务器，运行在它自己的解释器中。无需任何额外代码：

```
# Same nodeset code, two deployment modes:
POST /api/components/nodesets/env_habitat/load              # in-process
POST /api/components/nodesets/env_habitat/load?mode=server  # separate process
```

### 2.4 嵌套图系统

把任意画布图保存为一个**图节点（graph node）**，并将它拖到另一张画布上作为可复用的积木。这使分层的智能体架构成为可能 —— 一个高层规划器内部包含若干子智能体图节点。快照语义：每个实例都是一次深拷贝。

### 2.5 状态容器系统

通过双连线架构在智能体循环的多次迭代间共享持久状态：

- **数据边（Data edges）** 在节点之间承载数据流（IMAGE、TEXT、ACTION、POSE、…）
- **访问授权（Access grants）** 让节点读/写 **StateContainers** —— 它们是画布上可见的元素，带有具名条目、可配置的 reducer（Accumulator、LastWrite、Counter），以及一条 **Lifetime** 轴（`forever` / `step` / `episode` / `run` / `custom`），会在恰当的信号边界上自动清空内存（ADR-dataflow-002、ADR-dataflow-004）

→ [状态容器设计文档](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/graph/state-containers.html)

### 2.6 Python 定义的节点

**一个 Python 类 = 一个节点。** 每个画布节点 —— 工具、环境、技能、策略 —— 都是单个 Python 类：声明端口、实现 `forward()`、把文件丢进 `workspace/`，平台便会自动发现它。无需改动框架，无需 TypeScript，无需注册样板代码。

```python
from app.components import BaseCanvasNode, PortDef

class MeasureDistanceNode(BaseCanvasNode):
    node_type    = "basic_agent__measure_distance"
    display_name = "Measure Distance"
    description  = "Euclidean distance between two 3D positions"
    category     = "tool"
    icon         = "Ruler"

    input_ports  = [
        PortDef("pos_a", "TEXT", "Position A as [x, y, z]"),
        PortDef("pos_b", "TEXT", "Position B as [x, y, z]"),
    ]
    output_ports = [
        PortDef("distance", "TEXT", "Euclidean distance (meters)"),
    ]

    async def forward(self, inputs, ctx):
        a, b = parse_vec3(inputs["pos_a"]), parse_vec3(inputs["pos_b"])
        dist = math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
        return {"distance": f"{dist:.2f}"}
```

随后该节点会出现在画布侧边栏，并能与任何端口类型匹配的其他节点连线。它的外观同样由 Python 驱动：`GenericBlockRenderer` 会根据 `NodeUIConfig` 自动渲染任意节点 —— 颜色、布局、内联配置控件（滑块、下拉框、文本框）以及显示控件 —— 因此无需任何自定义 React 组件。

### 2.7 批量评估与任务队列

同一张在画布上运行的图，可以作为一个评估任务提交，对它在数百个 episode 上打分。一个由后端拥有的 `JobScheduler` 会针对所有会话共享的 VRAM 预算把控准入（ADR-eval-003）；每个被准入的运行都是它自己的子进程，其生命周期绑定到后端（`PR_SET_PDEATHSIG`）—— 没有孤儿 GPU 进程，且每个已完成的 episode 都会持久化到磁盘。逐 episode 的日志落在一个自包含的布局里（ADR-eval-004），让队友无需重跑就能重放任意单个 episode。

### 2.8 执行日志与实时视图

每一步都通过 WebSocket 流式推送观测、推理、动作与指标，并按 `execution_id` 路由，使并发运行不会串流。来自任何来源的错误 —— 节点异常、server mode 子进程崩溃、HTTP 失败 —— 都流经统一的 `ErrorBus`，并以 Report 标签页条目 + toast 的形式呈现（ADR-observability-004）。（React 渲染错误由客户端的错误边界捕获。）

</details>

---

## 3. 从仿真到真机的路径

AgentCanvas 为可移植性而设计：同一张智能体图，今天可以在仿真器上执行，未来无需图层面的改动即可迁移到真实机器人。这一特性源自两项架构决策 —— 环境本身就是 nodeset（ADR-components-002），且任何 nodeset 都能通过 *server mode* 在隔离的运行时中执行（ADR-server-001）。

<details>
<summary><b>完整路径</b> —— 当下的仿真器 · 拥有相同接口的 ROS nodeset · 双向集成 · 真值可见性</summary>

<br>

### 当下：仿真器 Nodeset

已随附的环境 —— Habitat（VLN-CE）、MatterSim / MP3D、HM-EQA、OpenEQA、SIMPLER（real-to-sim VLA）以及 LIBERO（操作）—— 每一个都实现为一个暴露观测端口与动作端口的 `BaseNodeSet`。智能体图连接到这些端口，从不直接 import 仿真器，这使得图独立于任何具体的环境实现。

### 未来：拥有相同接口的 ROS Nodeset

真实机器人的部署通过把仿真器 nodeset 替换为一个暴露相同 `observation` / `act` 接口的 **ROS nodeset** 来实现。在其内部，该 nodeset 把现有的 ROS 组件 —— `cv_bridge`、`Nav2`、`MoveIt` 以及硬件驱动包 —— 组合成一个统一的门面。Server mode 在它自己的 ROS Python 环境中启动该 nodeset，并通过 HTTP 把它桥接到画布。智能体图本身保持不变。

这种分工之所以有利，是因为实质性的工程 —— 感知、控制、运动规划与硬件接口 —— 已经以成熟的 ROS 包的形式存在。因此 ROS 一侧的适配器是一项组合任务，而非从零开发，而 AgentCanvas 一侧的 env nodeset 则简化为一个轻薄的 HTTP 客户端。

### 双向集成

AgentCanvas 与 ROS 之间的边界是对称的；任意一侧都可以拥有控制循环：

- **ROS 作为 AgentCanvas 的子系统** *（原生模式；server mode 正是为这种情况而设计）* —— ROS nodeset 以 server mode 运行，AgentCanvas 驱动智能体循环，ROS 提供感知与执行。
- **AgentCanvas 作为 ROS 的子系统** *（同样支持；无需修改框架）* —— 当更宏观的项目以 ROS 为主导时，ROS 一侧的控制循环在每一步调用 AgentCanvas 的 `/run` 端点（把图当作一个策略），并发布返回的动作。这只需在 ROS 一侧加一个轻薄的 ROS 桥接节点。

### 真值依赖的可见性

同一套 nodeset 抽象直接回应了 §1 提出的两个痛点。一个查询仿真器真值的节点（例如 `env_habitat__get_object_pose`）与一个执行真实感知的节点（例如基于 SAM 的检测器）在画布上呈现为可见地相互区分的积木。因此，一个智能体究竟依赖真值还是依赖感知，是图拓扑的属性，而非隐藏的实现细节。把其中一个替换为另一个，是一次局部的边改动，而非一次代码重构。

### 状态

目前已随附的所有环境 nodeset 都是基于仿真器的。真实机器人的 **ROS nodeset 仍是一个[征集贡献](#5-贡献)的空位** —— 架构路径已经确立且是刻意为之，所需的 ROS 一侧组件也已在生态中就绪。

</details>

---

## 4. 快速开始

使用 AgentCanvas 有两种方式，都建立在同一个类型化图基底之上：

1. **手工构建并运行一张图** —— 在画布上组合节点，让一个智能体实时地对着仿真器运行，并在规模上评估它（本节其余部分）。
2. **智能体架构搜索（AAS）** —— 把一张种子图交给一个编程智能体，让它替你搜索架构（[跳转](#44-运行智能体架构搜索-aas)）。

### 4.1 前置条件

- Python 3.10+ 配合 Conda（默认的 `agentcanvas` 环境 —— ADR-platform-004）
- Node.js 18+
- *（可选，用于 Habitat-Sim）* 一个独立的 Python 3.8 环境 —— `habitat-sim 0.1.7` 只在这里运行；AgentCanvas 通过 server mode 与它通信，参见 [INSTALL.md](INSTALL.md)

### 4.2 运行 Web 仪表盘

```bash
# Activate environment
conda activate agentcanvas

# Start backend (FastAPI :8000) + frontend (Vite :5173)
cd agentcanvas && bash run_dev.sh
```

打开 [http://localhost:5173](http://localhost:5173) 访问画布编辑器。

### 4.3 运行评估

同一条评估流水线通过四种接口暴露 —— 按你手头的东西来选：

| # | 接口 | 受众 | 最适合 |
|---|-----------|----------|----------|
| 1 | **前端 Eval 页面** | 人类                | 点击驱动，在 UI 里实时观看进度 |
| 2 | **`/experiment:run` 斜杠命令** | 编程智能体（Claude Code） | profile 把控的 GPU 准入、自动分配端口、不会踩 `:8000` |
| 3 | **MCP server** | 编程智能体              | 对话式、临时评估 —— 没有斜杠命令的额外开销 |
| 4 | **HTTP API** | 脚本 / CI                | 直连 REST，无需 MCP |

#### 1. 前端 Eval 页面 —— 面向人类

在 **Eval** 页面打开一张已保存的图，选择一个 split + episode 范围，按 **Start**。进度通过 WebSocket 实时推送；结果以逐 episode 的 JSONL 落在 `outputs/eval_runs/{run_id}/episodes/ep{NNNN}/` 下（ADR-eval-004），并可在 Run Detail 面板中浏览。多 worker 的环境扇出与批量推理可从表单配置（ADR-eval-002）。

→ [批量评估教程](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/tutorials/batch-eval.html)

#### 2. `/experiment:run` —— 面向本仓库上的编程智能体

使用 Claude Code 时，`/experiment:run <profile> -- <cmd>` 会把任意评估调用包裹进后端 `JobScheduler` 的准入门（ADR-eval-003）：包装器在 `.claude/commands/experiment/profiles.yaml` 中声明的 profile 下认领 VRAM，在一个分配到的端口上启动后端（`BACKEND_URL=http://127.0.0.1:<port>` 会被导出给被包裹的命令），并在退出时释放该槽位。配套命令：`/experiment:status` 查看运行快照，`/experiment:teardown` 优雅取消。

→ [`.claude/commands/experiment/README.md`](../.claude/commands/experiment/README.md)

对于完整的架构搜索设计循环（在一张种子图上多次迭代「提议 → 评估 → 留下最优」），见下文[运行智能体架构搜索](#44-运行智能体架构搜索-aas)。

#### 3. MCP server —— 面向编程智能体

向任何支持 MCP 的客户端（Claude Code、Cursor、…）注册 `agentcanvas-backend`，并以对话方式调用类型化工具（`graph_list`、`eval_start`、`eval_status`、`eval_export`、`eval_stop`）。无需 iter-tree 记账 —— 只是对着一个借用或新建的后端做原始评估。

→ [`agentcanvas/mcp_server/README.md`](../agentcanvas/mcp_server/README.md)

#### 4. HTTP API —— 面向脚本与 CI

为脚本、CI 或非 MCP 环境提供直连 REST：

```bash
curl -X POST http://localhost:8000/api/eval/v2/start \
  -H 'content-type: application/json' \
  -d '{"graph_name": "navgpt_ce", "split": "val_unseen", "worker_count": 4}'
# poll  GET /api/eval/v2/status
# fetch GET /api/eval/v2/export/{run_id}
```

→ [从编程智能体驱动后端](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/tutorials/coding-agent-backend.html) —— 并排深入讲解所有编程式模式

### 4.4 运行智能体架构搜索 (AAS)

除了评估一张固定的图，AgentCanvas 还是 **智能体架构搜索** 的基底 —— 这是一个开发期循环：一个 LLM 编程智能体 *Optimizer* 反复向一张种子 *Executor* 提议图编辑，在仿真器中评估每个候选，并留下那些带来改进的（§1 —— [为何需要一个可搜索的基底](#1-为什么选择-agentcanvas)）。因为一个智能体就是一张类型化的图，每个候选都是一个经过类型检查的补丁，会在任何昂贵的 rollout 之前先运行，而逐节点的 episode 日志让 Optimizer 能把分数变化归因到具体模块。

<p align="center">
  <img src="../assets/readme/aas-search.gif" alt="编程智能体 optimizer 在一个具身 executor 的图上搜索 —— 提议编辑、运行它们、留下收益" width="800">
  <br><sub><em>编程智能体 optimizer 在一个具身 executor 的图上搜索 —— 提议一个编辑、运行它、留下收益。</em></sub>
</p>

搜索是**方法播种（method-seeded）**的：`iter_0` 是一个已发表的具身方法，循环在它周围搜索图层面的编辑。三种搜索变体作为 Claude Code 技能随附在 `.claude/commands/architect/` 下，共享同一套编程智能体 harness（proposer → implementer → evaluator），仅在 proposer 逻辑 + 持久记忆上有所不同：

| 变体技能 | 论文名称 | 搜索策略 |
|---|---|---|
| `myloop` | **KDLoop** | 四阶段 THINK → CRITIC → EXPERIMENT → DISTILL 循环，类型化记忆 + REFLECT 元阶段 |
| `adas-subagent` | **ADAS**（移植） | 在一个扁平的仅追加归档上做 Reflexion 式提议 |
| `aflow` | **AFlow**（移植） | 分数 softmax 的父节点选择 + 防重放记忆 |

```text
# In a Claude Code session on this repo — run KDLoop over the MapGPT executor
/architect:myloop:loop mapgpt_mp3d --goal "raise val_unseen SR"

# The ADAS / AFlow ports take the same  <graph> [<version>]  form
/architect:adas-subagent:loop smartway_ce
/architect:aflow:loop explore_eqa_hmeqa
```

目前已接入搜索的种子图：`mapgpt_mp3d`、`smartway_ce`（VLN）、`explore_eqa_hmeqa`（EQA）、`voxposer_libero_monolithic`（VLA）。每次迭代会把它的提议、补丁、评估分数与日志写到 `outputs/design_runs/{variant}/{graph}/vN/iter_M/` 下。

→ [AAS 流水线参考](https://jianzhou0420.github.io/AgentCanvas/pages/aas/index.html)

### 4.5 文档

```bash
# Serve the doc-site locally on :8092 (live-reload via SSE)
bash docs/run_dev.sh
```

---

## 5. 贡献

两类贡献，皆受欢迎 —— 参见 [CONTRIBUTING.md](../CONTRIBUTING.md)：

- **内容类 —— nodeset 与图。** 编写一个包装工具 / 仿真器 / 模型的 nodeset（例如实时 3D 高斯泼溅、一个基于体素的 SLAM 系统），或编码一个方法（例如 NavGPT、MapGPT），或组合一张把现有 nodeset 连成一个完整智能体的图。提一个 PR 合入 `workspace/`；评审从轻。
- **核心类 —— UI、后端、框架。** Bug 修复、新功能，乃至重构都受欢迎。唯一的请求：如果一个改动大到会耗费真金白银的时间，请先发起一个 [Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions)，以便在你动手之前对齐。

下方榜单会把每个 nodeset 和图记到其作者/维护者名下 —— 若有相关论文还会附上引用链接 —— 因此在这里贡献不会让你失去署名。**AgentCanvas 框架**以及首个发布的**方法、图与环境集成**由 **AC-Team** 完成。下方的**基础模型与策略**均为**第三方** —— AgentCanvas 只随附一层轻薄的 server-mode 包装，让每一个都能接入图（供人类用户与 AAS optimizer 同等使用）；每个模型的功劳都归其原作者所有 —— 这些基础模型被单独抽到**下方一张独立的表**中，按来源拆分（transformers-native 对比 `torch.hub` / `torchvision` / vendored 上游仓库），逐模型的完整署名在 Credits 页面上。该榜单在设计上只列名字：带有逐图验证细节的**权威清单**位于[文档站 Credits 页面](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html)以及 [VLN](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/vln-support-status.html) / [EQA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/eqa-support-status.html) / [VLA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/vla-support-status.html) 支持状态页面。

### 致谢

✅ 已验证 —— 复现其论文 / 参考实现 · 🚧 端到端可运行，验证进行中

<table>
  <thead align="center">
    <tr>
      <th>环境</th>
      <th>方法</th>
      <th>模型与策略</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/habitat.html">Habitat (VLN-CE)</a> ✅</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/matterport3d.html">MatterSim / MP3D</a> ✅</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/hmeqa.html">HM-EQA</a> ✅</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/openeqa.html">OpenEQA (EM-EQA)</a> ✅</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/simpler.html">SIMPLER</a> ✅</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/libero.html">LIBERO</a> ✅</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><b>VLN</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/navgpt.html">NavGPT</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/mapgpt.html">MapGPT</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/smartway.html">SmartWay</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/threestepnav.html">Three-Step Nav</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/aoplanner.html">AO-Planner</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/discussnav.html">DiscussNav</a> 🚧</li>
              <li>Open-Nav 🚧</li>
              <li>SpatialNav 🚧</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/common/tools/basic-agent.html">Basic Agent 工具包</a> ✅</li>
            </ul>
          </li>
          <li><b>EQA</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/openeqa.html">EM-EQA 基线</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/explore-eqa.html">Explore-EQA</a> ✅</li>
              <li>ToolEQA 🚧</li>
            </ul>
          </li>
          <li><b>VLA（零样本）</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/voxposer.html">VoxPoser-LIBERO</a> ✅</li>
            </ul>
          </li>
        </ul>
      </td>
      <td>
        <ul>
          <li><b>策略</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-cma.html">CMA</a> ✅</li>
              <li>Octo（SIMPLER 基线） ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-vla.html">VLA 框架（Pi0 / SmolVLA / DP / DROID-DP）</a> 🚧</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-adapters.html">R2R-CE 策略注册表（12 个变体）</a> 🚧</li>
            </ul>
          </li>
          <li><b>建图</b> <sub><i>（AgentCanvas 自研）</i></sub>
            <ul>
              <li>TSDF 建图 ✅</li>
              <li>语义场景图 ✅</li>
            </ul>
          </li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

**基础模型** —— 第三方模型被包装在**一层轻薄的 nodeset 外壳**之后（惰性加载 · single-flight GPU · base64-npy 数据流封装），使每一个都成为供**人类用户**与 **AAS optimizer** 使用的统一构件。*这些并非我们所作 —— 我们只随附外壳；功劳归原作者所有*（逐模型的完整署名 + 论文见 [Credits 页面](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html)）。按来源拆分：

<table>
  <thead align="center">
    <tr>
      <th>transformers-native <sub>（对 <code>AutoModel</code> / <code>pipeline</code> 的轻薄包装）</sub></th>
      <th>其他来源 <sub>（<code>torch.hub</code> / <code>torchvision</code> / vendored 上游仓库）</sub></th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-clip.html">CLIP</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-siglip2.html">SigLIP 2</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-aimv2.html">AIMv2</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-owlv2.html">OWLv2</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-sam.html">SAM</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-sam-video.html">SAM Video</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-segmentation.html">Segmentation（Mask2Former）</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-florence2.html">Florence-2</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-depth-anything.html">Depth Anything V2</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-depthpro.html">DepthPro</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-normal.html">Surface Normals（Sapiens）</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-pointmap.html">Pointmap（Sapiens 3D）</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-matching.html">SuperPoint + LightGlue</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-blip2.html">BLIP-2</a> + Faster R-CNN</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-instructblip.html">InstructBLIP</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-qwen2-5-vl.html">Qwen2.5-VL</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-qwen3-vl.html">Qwen3-VL</a> <sub>（图像 + 视频）</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-internvl3.html">InternVL3</a> <sub>（图像 + 视频）</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-gemma3.html">Gemma 3</a> <sub>（受限访问）</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-smolvlm2.html">SmolVLM2</a> <sub>（图像 + 视频）</sub></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-dinov2.html">DINOv2 / DINOv3</a> <sub>（torch.hub + transformers hf）</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-grounding-dino.html">Grounding DINO</a> <sub>（groundingdino-py + transformers hf_tiny）</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-opticalflow.html">Optical Flow (RAFT)</a> <sub>（torchvision）</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-vggt.html">VGGT</a> <sub>（上游仓库）</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-cotracker.html">CoTracker</a> <sub>（上游仓库）</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-detany3d.html">DetAny3D</a> <sub>（vendored）</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-ram.html">RAM / RAM++</a> <sub>（recognize-anything）</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-spatialbot.html">SpatialBot</a> <sub>（Bunny 远程代码）</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-prismatic.html">Prismatic VLM</a> <sub>（上游仓库）</sub></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

**征集贡献** —— 预留的空位，归到把它们实现落地的人名下（[如何贡献](../CONTRIBUTING.md)；ID 是 [Credits 页面](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html)上的路线图空位）：

<table>
  <thead align="center">
    <tr>
      <th>基准</th>
      <th>方法</th>
      <th>功能与基础设施</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul>
          <li>AI2-THOR —— ALFRED / TEACh <i>(E4)</i></li>
          <li>RxR-CE —— 多语言 VLN-CE <i>(E2)</i></li>
          <li>REVERIE —— 远程物体定位 <i>(E3)</i></li>
          <li>OpenEQA A-EQA —— 主动 EQA <i>(E10)</i></li>
        </ul>
      </td>
      <td>
        <ul>
          <li>HAMT —— 分层历史 transformer <i>(M5)</i></li>
          <li>DUET —— 双尺度图 transformer <i>(M6)</i></li>
          <li>InstructNav —— 动态 CoN + 价值图 <i>(M8)</i></li>
          <li>VLN-SIG —— 子指令定位 <i>(M4)</i></li>
        </ul>
      </td>
      <td>
        <ul>
          <li>记忆 nodeset —— 情景回忆 + 语义检索 <i>(F1)</i></li>
          <li>节点并行执行 —— Pregel 超步 <i>(F3)</i></li>
          <li>把图导出为独立 Python <i>(F4)</i></li>
          <li>Docker server mode —— Habitat / MP3D 容器 <i>(F7)</i></li>
          <li>ROS nodeset —— 真实机器人部署（<a href="#3-从仿真到真机的路径">§3</a>）</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


---

## 6. 引用

如果您在研究中使用了 AgentCanvas，请引用：

```bibtex
@misc{jian2026AgentCanvas,
  title         = {Automating the Design of Embodied Agent Architectures},
  author        = {Jian Zhou and Sihao Lin and Jin Li and Shuai Fu and Gengze Zhou and Qi Wu},
  year          = {2026},
  eprint        = {2606.30111},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.30111}
}
```

## 7. 许可证

Apache License 2.0 —— 见 [LICENSE](../LICENSE)。
