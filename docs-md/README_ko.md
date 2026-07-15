[English](../README.md) | [中文](README_zh.md) | [Español](README_es.md) | [日本語](README_ja.md) | **한국어**

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
  <a href="#6-인용"><img src="https://img.shields.io/badge/BibTeX-Cite-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white" alt="BibTeX"></a>
</p>

<img src="../assets/readme/editor-hero.gif" alt="AgentCanvas 에디터: MapGPT executor가 노드-와이어 그래프로 로드된 뒤, 실시간 R2R 에피소드가 엔드투엔드로 실행된다" width="760">

<sub><em>에디터에서 실시간으로 녹화 — MapGPT executor가 로드된 뒤, 실제 R2R 에피소드가 엔드투엔드로 실행됩니다.</em></sub>

</div>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Status: Research Preview](https://img.shields.io/badge/Status-Research_Preview-orange.svg)](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html)
[![GitHub stars](https://img.shields.io/github/stars/jianzhou0420/AgentCanvas?style=social)](https://github.com/jianzhou0420/AgentCanvas/stargazers)

**임바디드 AI 연구를 위한 비주얼 에이전트 설계 플랫폼.** 하나의 타입이 지정된 그래프, 두 가지 역할: 임바디드 에이전트를 실행하는 *하네스(harness)*, 그리고 코딩 에이전트가 편집하고 검증하는 *스캐폴드(scaffold)*.

AgentCanvas는 연구자가 노드 그래프를 그리는 것만으로 임바디드 에이전트 — VLN, EQA, VLA 및 인접 태스크용 — 를 프로토타이핑할 수 있게 해줍니다. 이 그래프는 시뮬레이터 (Habitat-Sim, MatterSim, SAPIEN/ManiSkill2, MuJoCo/robosuite) 에 대해, 또는 원리적으로는 실세계 환경에 대해 실시간으로 실행됩니다. *하나의 JSON = 하나의 에이전트 = 하나의 그래프*: 에이전트의 동작은 명령형 코드가 아니라 데이터플로 그래프이며, 그래프가 곧 진실의 원천(source of truth)으로서 단일 JSON 파일로 저장되고 완전한 에이전트로 로드됩니다.

**대상 사용자**: 매번 실행 스택을 새로 작성하지 않고 임바디드 에이전트 아키텍처를 구성하고, 비교하고, 공유하고자 하는 연구자. 이 플랫폼은 VLN (Vision-and-Language Navigation), EQA (Embodied Question Answering), VLA (Vision-Language-Action) 정책 벤치마크를 다루며, 노드셋 모델을 통해 다른 임바디드 / 에이전트 설정에도 적응합니다.

> **상태**: 연구 프리뷰, pre-1.0 — 4개의 교체 가능한 팔레트 (**env** · **method** · **model** · **policy**) 에 걸친 40개 이상의 노드셋; 공개 API는 아직 고정되지 않았습니다 ([Versioning Policy](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html)).

> **기여**: 노드셋, 그래프, 코어 PR 모두 환영합니다 — 모든 기여는 [크레딧](#크레딧) 보드에 기재됩니다. [CONTRIBUTING.md](../CONTRIBUTING.md)를 참조하세요.

---

## 새로운 소식!

- [2026/07] 🚀 **Graph SDK — Python에서 에이전트 구축 & 실행** — 동일한 캔버스 그래프를, 이제 임포트 가능한 라이브러리로: `from agentcanvas import Graph`로 노드를 추가/연결하고, 인프로세스로 실행 및 배치 평가하거나, 그래프를 독립 실행형 빌더 스크립트로 다시 컴파일하세요. 동일한 `GraphDefinition`이며, 캔버스 + JSON과 완전히 가역적입니다. [Graph SDK 문서](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/capabilities/graph-sdk.html)를 참조하세요.
- [2026/07] 🎥 **pySLAM 클래식 SLAM 데모** — TUM RGB-D 위에서 pySLAM이 주인공으로: 스트리밍 재생 env가 벤치마크 시퀀스를 프레임 단위로 실시간 SLAM 세션에 공급합니다 — 추정된 카메라 트래젝터리가 top-down 그라운드 트루스 위에 피팅되고 희소 3D 맵이 실시간으로 조밀해지며, 시뮬레이터나 정책 없이 CPU만으로 동작합니다. 전체 클립 + 상세 설명은 [pySLAM 노드셋 문서](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-pyslam.html)에서.

  [![TUM RGB-D 위의 pySLAM 스트리밍 SLAM — 실시간 카메라 트래젝터리 vs 그라운드 트루스, 실시간으로 조밀해지는 3D 맵, 그리고 완성된 맵의 궤도 회전](../docs/assets/videos/pyslam-tum-slam-demo.gif)](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-pyslam.html)
- [2026/07] 🔥 **더 넓어진 파운데이션 모델 지원** — 29개의 파운데이션 모델이 이제 얇은 서버 모드 셸 (transformers-native + 기타 소스) 로 배선되어, 손으로 구축한 그래프와 AAS 옵티마이저 양쪽에서 사용할 수 있습니다: 최신 VLM (Qwen3-VL, InternVL3, Gemma 3, SmolVLM2), 오픈 어휘 지각 (SigLIP2, OWLv2, Grounding DINO), 그리고 기하 / 깊이 백본. [파운데이션 모델 커버리지](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/index.html)와 모델별 [크레딧](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html)을 참조하세요.
- [2026/07] 🔥 **캔버스에서 노드 소스 편집** — 새로운 Source 탭은 선택된 노드가 속한 노드셋 소스의 스코프된 슬라이스 (전역 변수, 참조된 함수, 클래스 자체) 를 보여주고, 편집을 구문 검사된 핫 리로드로 다시 이어 붙입니다. PR: [#5](https://github.com/jianzhou0420/AgentCanvas/pull/5).
- [2026/07] 🎉 **첫 공개 릴리스** — AgentCanvas가 연구 프리뷰 (pre-1.0) 로 오픈소스화되었습니다. 문서: [jianzhou0420.github.io/AgentCanvas](https://jianzhou0420.github.io/AgentCanvas/).

---

## 목차

1. [왜 AgentCanvas 인가?](#1-왜-agentcanvas-인가) — 임바디드 에이전트를 위한 탐색 가능한 기반, 그리고 그것이 해결하는 고충점들
2. [기능](#2-기능) — *하나의 JSON = 하나의 에이전트* (§2.2) / *하나의 Python 클래스 = 하나의 노드* (§2.6) 원칙, 그리고 캔버스 에디터, 그래프 executor, 격리된 런타임 환경, 중첩 그래프, 상태 컨테이너, 훅
3. [Sim-to-Real 경로](#3-sim-to-real-경로) — 오늘은 시뮬레이터, 내일은 실제 로봇에서 동일한 에이전트 그래프를 — env-as-nodeset + 서버 모드 + ROS를 통해
4. [시작하기](#4-시작하기) — 사전 요구사항, 웹 대시보드 실행, 평가 실행, 아키텍처 탐색 실행, 문서 제공
5. [기여하기](#5-기여하기) — 도움이 가장 필요한 영역 · 크레딧
6. [인용](#6-인용) — AgentCanvas 논문 인용하기
7. [라이선스](#7-라이선스) — Apache 2.0

---

## 1. 왜 AgentCanvas 인가?

임바디드 에이전트 — VLN, EQA, VLA를 아우르는 — 는 점점 더 파운데이션 모델을 지각, 매핑, 메모리, 계획, 행동과 조합하여 구축됩니다. 구조가 가중치 안으로 흡수되는 엔드투엔드 정책과 달리, 이 아키텍처는 *명시적이고 편집 가능*합니다. 이는 AgentCanvas가 그 주위에서 구축된 질문 — **에이전트 설계는 손으로 만드는 대신 탐색될 수 있는가?** — 을 제기하며, 동시에 그 과정에서 넘어서야 할 두 무더기의 고충을 안깁니다.

<details>
<summary><b>에이전트 아키텍처는 손으로 만들어진다 — 그리고 탐색될 수 있다</b></summary>

<br>

각 에이전트는 모든 접합부에서의 선택 — 센서 추상화, 맵 표현, 메모리 상태, 프롬프트 구조, 플래너 토폴로지, 모델 배치, 행동 인터페이스 — 을 보통 단일 벤치마크를 위해 손으로 고정합니다. 파운데이션 모델과 임바디드 도구가 늘어남에 따라, 그 공간은 수동 반복이 감당할 수 있는 것보다 빠르게 커지므로, 자연스러운 수순은 그것을 손으로 튜닝하는 대신 탐색하는 것입니다.

Agent Architecture Search (AAS) 는 텍스트 도메인 에이전트에 대해 이미 이를 수행하지만, 임바디드 영역으로의 이전은 공짜가 아닙니다: 상태를 갖는 시뮬레이터, 노이즈가 많은 다중 에피소드 점수화, 긴 지각/행동 트레이스, 그리고 기성품으로 제공되는 임바디드 프리미티브 팔레트의 부재. AgentCanvas는 그 빠진 기반 — 코딩 에이전트가 읽고, 편집하고, 실행하고, 검증할 수 있는 스캐폴드 — 을 공급하려는 우리의 시도이며, 이를 통해 임바디드 에이전트에 대해서도 에이전트 설계 탐색이 가능해집니다.

</details>

<details>
<summary><b>임바디드 고유의 고충점</b></summary>

<br>

- **현대의 임바디드 스택은 두껍다** — 동작하는 임바디드 에이전트는 LLM 추론 + 도구 사용 + 시뮬레이터 결합 + 공간 도구가 모두 함께 배선되어야 합니다. 이를 프로젝트마다 처음부터 구축하는 것은 비용이 감당하기 어려울 만큼 크며, 노력의 대부분은 검증하려는 아이디어가 아니라 실행 계층으로 들어갑니다.
- **엔지니어링 악몽** — 임바디드 에이전트는 하나의 모델이 아니라 시스템 전체 — 상태를 갖는 시뮬레이터에 더해 무거운 모델과 도구의 스택 — 입니다. 벤치마킹이 요구하는 규모는 차치하고 그저 실행하는 것만으로도 그 자체가 어려운 엔지니어링 작업입니다:
  - **Python env 지옥** — 모든 부분을 만족시키는 단일 Python env는 없습니다; 각 시뮬레이터, VLM, 검출기, 정책은 서로 충돌하는 자체 CUDA / torch / Python을 고정하므로, 그들 모두가 공유하는 하나의 런타임을 찾는 것은 종종 불가능합니다 — 결국 에이전트를 로드하기 위해 여러 개의 호환되지 않는 환경을 유지하게 됩니다.
  - **배칭** — 각 워커의 시뮬레이터는 자체 속도로 진행되는 별도의 상태 보유 프로세스입니다; 모델은 배치 처리할 수 있지만 시뮬레이터는 그럴 수 없으므로, 모든 스텝이 비동기 관측-수집(gather-observations) → 배치 추론(batch-infer) → 행동-분배(scatter-actions) 의 춤이 됩니다.
  - **기타 인프라** — 로깅되고 재생 가능해야 하는 멀티모달 트래젝터리, *반드시* 크래시가 나는 수 시간짜리 GPU 실행의 체크포인트/재개, 그리고 프로세스 경계를 넘나드는 디버깅.

  단일 논문의 연구 사이클 동안, 연구자는 알고리즘 그 자체에 집중하는 대신 이 엔지니어링 비용을 지나치게 많이 치릅니다.
- **숨겨진 그라운드 트루스 의존성** — 많은 방법이 실제 지각이 아니라 시뮬레이터가 제공하는 그라운드 트루스 (물체 포즈, 시맨틱 레이블, 주행 가능성) 에 조용히 의존합니다. 때로는 그것이 실험을 통제하는 정당한 방법이기도 합니다 — 하지만 의도적이든 누락이든, 논문에서는 언급되지 않는 경우가 많습니다.

</details>

<details>
<summary><b>일반적인 AI 연구의 고충점 (여기서는 증폭됨)</b></summary>

<br>

- **재현 불가능한 구현** — 모든 논문이 서로 다른 코드베이스로 에이전트를 처음부터 구축합니다; 방법을 공정하게 비교하거나 결과를 재현하는 것은 고통스럽습니다 — 게다가 그들 중 다수는 **`Code coming SOON`** (**S**omeday, **O**r **O**bviously **N**ever — 즉 "언젠가, 아니면 당연히 영원히 안 나옴") 상태입니다.
- **논문 ≠ 코드** — 논문은 깔끔한 흐름도를 보여주지만, 실제 코드는 문서화되지 않은 방식으로 갈라집니다. 논문을 재현한다는 것은 그 구현을 리버스 엔지니어링한다는 뜻입니다.
- **강하게 결합된 코드** — 도메인 로직 (프롬프트, 도구, 정책) 이 인프라와 얽혀 있습니다. 한 컴포넌트를 교체하는 것은 파이프라인을 다시 작성하는 것을 의미합니다.

</details>

---

## 2. 기능

> **문서의 전체 레퍼런스** — 아래 대부분의 기능에는 구현 페이지 (메커니즘 · 핵심 파일 · 현재 상태) 가 있습니다: **[9가지 역량 →](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/capabilities/index.html)**

<details>
<summary><b>9가지 역량</b> — 캔버스 에디터 · 그래프 엔진 · 격리된 런타임 · 중첩 그래프 · 상태 컨테이너 · Python으로 정의되는 노드 · 훅 · 배치 평가 · 관측성</summary>

<br>

### 2.1 비주얼 캔버스 에디터

모든 노드 타입 — 환경, LLM, 추론 체인, 제어 게이트, 출력 뷰어 — 이 공존하는 ComfyUI 스타일의 평평한 워크스페이스. 사이드바에서 노드를 드래그하고, 서로 배선하고, Play를 누르세요.

### 2.2 그래프 실행 엔진

**하나의 JSON = 하나의 에이전트.** 에이전트의 동작 전체 — 노드, 배선, 설정, 상태 컨테이너, 훅 — 가 단일 JSON 파일입니다: 로드하고, 실행하고, 공유하고, diff하세요. 숨겨진 파이프라인 코드는 없습니다; 캔버스에서 보이는 것이 곧 실행되는 것입니다.

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

그러면 엔진이 그 그래프를 실행합니다: 노드는 고정된 순서가 아니라 입력이 도착할 때 발화합니다. 동일한 엔진이 AgentCanvas v1이 지원하는 모든 그래프 형태를 처리합니다 — v1의 bounded-static-topology 패러다임이 다루는 에이전트 형태의 전체 목록은 [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html)을 참조하세요:

- **DAG 워크플로** — 비순환 파이프라인을 위한 단일 포워드 패스
- **순환 에이전트 루프** — **2-피벗** 모델을 통한 observe-think-act-repeat: 양면을 가진 **`IterIn`** (왼쪽에 실행 시작 시점의 init 입력, 오른쪽에 반복마다의 loop-carry) 에 더해 **`IterOut`** 으로, 그래프를 시각적으로 비순환으로 유지하면서 런타임 사이클을 가능하게 합니다 (ADR-dataflow-008, 이는 ADR-dataflow-006의 이전 3-피벗 `initialize`/IterIn/IterOut을 둘로 접었습니다)
- **멀티 스코프 반복** — 하나의 평평한 그래프 안에 N개의 `(IterIn, IterOut)` 쌍이 공존 (ADR-dataflow-007 / ADR-executor-003)
- **ReAct 루프** — `LLMCallNode` 서브클래스 안에 숨기거나, router + N개의 사전 선언된 도구 분기로 명시적으로 표현
- **경계가 있는 멀티 에이전트** — 고정-N 또는 `K_max`로 경계가 있는 팬아웃 (예: DiscussNav 스타일 토론, AutoGen 스타일 고정 역할)
- **Plan-and-Execute** — 경계가 있는 도구 풀에 대해, router로 디스패치

엔진은 그래프 노드를 건드리지 않고도 확장 가능합니다: 셸 훅이 각 노드 실행 전/후 및 그래프 라이프사이클 경계에서 발화하여 — 출력을 로깅하고, 입력을 검증하고, 노드를 차단하거나, 데이터를 수정할 수 있으며 — 저장된 그래프와 함께 이동합니다.

### 2.3 격리된 런타임 환경

연구 도구는 종종 서로 충돌하는 Python 환경을 필요로 합니다 (Habitat은 Python 3.8이 필요하고, SLAM은 ROS가 필요합니다). 어떤 `BaseNodeSet`이든 **서버 모드**로 실행할 수 있습니다 — 프레임워크가 노드셋의 포트 정의로부터 HTTP 서버를 자동 생성하여, 자체 인터프리터에서 실행합니다. 추가 코드는 전혀 필요 없습니다:

```
# Same nodeset code, two deployment modes:
POST /api/components/nodesets/env_habitat/load              # in-process
POST /api/components/nodesets/env_habitat/load?mode=server  # separate process
```

### 2.4 중첩 그래프 시스템

임의의 캔버스 그래프를 **그래프 노드**로 저장하고 다른 캔버스 위로 드래그하여 재사용 가능한 블록으로 사용하세요. 이는 계층적 에이전트 아키텍처 — 서브 에이전트 그래프 노드를 포함하는 상위 레벨 플래너 — 를 가능하게 합니다. 스냅샷 시맨틱: 각 인스턴스는 딥 카피입니다.

### 2.5 상태 컨테이너 시스템

듀얼 와이어 아키텍처를 통해 에이전트 루프 반복 전반에 걸쳐 영속 상태를 공유합니다:

- **데이터 엣지**는 노드 간 데이터플로를 운반합니다 (IMAGE, TEXT, ACTION, POSE, …)
- **액세스 그랜트**는 노드가 **StateContainers** 를 읽고 쓸 수 있게 합니다 — 이름이 지정된 엔트리, 설정 가능한 리듀서 (Accumulator, LastWrite, Counter), 그리고 적절한 시그널 경계에서 메모리를 자동으로 비우는 **Lifetime** 축 (`forever` / `step` / `episode` / `run` / `custom`) 을 가진 캔버스 상의 가시적 요소입니다 (ADR-dataflow-002, ADR-dataflow-004)

→ [State Containers 설계 문서](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/graph/state-containers.html)

### 2.6 Python으로 정의되는 노드

**하나의 Python 클래스 = 하나의 노드.** 모든 캔버스 노드 — 도구, 환경, 스킬, 정책 — 는 단일 Python 클래스입니다: 포트를 선언하고, `forward()`를 구현하고, 파일을 `workspace/`에 두면 플랫폼이 자동으로 검색합니다. 프레임워크 변경도, TypeScript도, 등록 보일러플레이트도 필요 없습니다.

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

그러면 그 노드는 캔버스 사이드바에 나타나며 포트 타입이 일치하는 다른 어떤 노드와도 배선됩니다. 그 외관 또한 Python으로 구동됩니다: `GenericBlockRenderer`가 `NodeUIConfig`로부터 모든 노드를 자동으로 렌더링합니다 — 색상, 레이아웃, 인라인 설정 컨트롤 (슬라이더, 드롭다운, 텍스트 필드), 그리고 표시 위젯 — 따라서 커스텀 React 컴포넌트가 필요 없습니다.

### 2.7 배치 평가 & 작업 큐

캔버스에서 실행되는 것과 동일한 그래프를, 수백 개의 에피소드에 걸쳐 점수를 매기는 eval 작업으로 제출할 수 있습니다. 백엔드가 소유한 `JobScheduler`가 모든 세션에 걸쳐 공유되는 VRAM 예산에 대해 수용을 게이트합니다 (ADR-eval-003); 수용된 각 실행은 그 수명이 백엔드에 묶인 자체 서브프로세스이며 (`PR_SET_PDEATHSIG`) — 고아가 된 GPU 프로세스가 없고, 완료된 모든 에피소드는 디스크에 영속됩니다. 에피소드별 로그는 자기 완결적인 레이아웃에 저장되므로 (ADR-eval-004), 팀원이 재실행 없이 임의의 단일 에피소드를 재생할 수 있습니다.

### 2.8 실행 로그 & 실시간 뷰

모든 스텝이 관측, 추론, 행동, 메트릭을 WebSocket을 통해 스트리밍하며, `execution_id`로 라우팅되어 동시 실행이 스트림을 교차시키지 않습니다. 어떤 출처의 에러든 — 노드 예외, 서버 모드 서브프로세스 크래시, HTTP 실패 — 통합된 `ErrorBus`를 거쳐 Report 탭 항목 + 토스트로 표면화됩니다 (ADR-observability-004). (React 렌더 에러는 클라이언트 측 에러 바운더리에서 포착됩니다.)

</details>

---

## 3. Sim-to-Real 경로

AgentCanvas는 이식성을 염두에 두고 설계되었습니다: 단일 에이전트 그래프는 오늘은 시뮬레이터에 대해 실행되고, 미래에는 그래프 수준의 변경 없이 실제 로봇으로 이전될 수 있습니다. 이 속성은 두 가지 아키텍처 결정에서 비롯됩니다 — 환경 자체가 노드셋이라는 점 (ADR-components-002), 그리고 어떤 노드셋이든 *서버 모드*를 통해 격리된 런타임에서 실행될 수 있다는 점 (ADR-server-001).

<details>
<summary><b>전체 경로</b> — 오늘의 시뮬레이터 · 동일한 인터페이스를 갖는 ROS 노드셋 · 양방향 통합 · 그라운드 트루스 가시성</summary>

<br>

### 오늘: 시뮬레이터 노드셋

출시된 환경들 — Habitat (VLN-CE), MatterSim / MP3D, HM-EQA, OpenEQA, SIMPLER (real-to-sim VLA), LIBERO (조작) — 은 각각 관측 포트와 행동 포트를 노출하는 `BaseNodeSet`으로 구현됩니다. 에이전트 그래프는 이 포트들에 연결되며 시뮬레이터를 직접 임포트하지 않으므로, 그래프가 특정 환경 구현으로부터 독립적으로 유지됩니다.

### 내일: 동일한 인터페이스를 갖는 ROS 노드셋

실제 로봇 배포는 동일한 `observation` / `act` 인터페이스를 노출하는 **ROS 노드셋**으로 시뮬레이터 노드셋을 대체함으로써 달성됩니다. 내부적으로 이 노드셋은 기존 ROS 컴포넌트 — `cv_bridge`, `Nav2`, `MoveIt`, 그리고 하드웨어 드라이버 패키지 — 를 통합된 파사드로 조합합니다. 서버 모드는 노드셋을 자체 ROS Python 환경 안에서 시작하고 HTTP를 통해 캔버스에 브리지합니다. 에이전트 그래프 자체는 변경되지 않습니다.

이러한 분업은 유리합니다. 왜냐하면 실질적인 엔지니어링 — 지각, 제어, 모션 플래닝, 하드웨어 인터페이싱 — 이 이미 성숙한 ROS 패키지로 존재하기 때문입니다. 따라서 ROS 측 어댑터는 그린필드 개발이 아니라 조합 작업이며, AgentCanvas 측 env 노드셋은 얇은 HTTP 클라이언트로 귀결됩니다.

### 양방향 통합

AgentCanvas와 ROS 사이의 경계는 대칭적이며, 어느 쪽이든 제어 루프를 소유할 수 있습니다:

- **AgentCanvas의 서브시스템으로서의 ROS** *(네이티브 패턴; 서버 모드가 이 경우를 위해 설계됨)* — ROS 노드셋이 서버 모드로 실행되고, AgentCanvas가 에이전트 루프를 구동하며, ROS가 센싱과 작동을 제공합니다.
- **ROS의 서브시스템으로서의 AgentCanvas** *(역시 지원됨; 프레임워크 수정 불필요)* — 더 넓은 프로젝트가 ROS 주도일 때, ROS 측 제어 루프가 각 스텝에서 AgentCanvas의 `/run` 엔드포인트를 호출하고 (그래프를 정책으로 취급), 반환된 행동을 퍼블리시합니다. 이는 ROS 측에 얇은 ROS 브리지 노드 하나만을 요구합니다.

### 그라운드 트루스 의존성의 가시성

동일한 노드셋 추상화는 §1에서 제기된 두 가지 고충점을 직접적으로 다룹니다. 시뮬레이터 그라운드 트루스를 질의하는 노드 (예: `env_habitat__get_object_pose`) 와 실제 지각을 수행하는 노드 (예: SAM 기반 검출기) 는 캔버스 상에서 시각적으로 구별되는 블록으로 나타납니다. 따라서 에이전트가 그라운드 트루스에 의존하는지 지각에 의존하는지는 숨겨진 구현 세부사항이 아니라 그래프 토폴로지의 속성입니다. 한쪽을 다른 쪽으로 대체하는 것은 코드 리팩터링이 아니라 국소적인 엣지 변경입니다.

### 상태

현재 출시된 모든 환경 노드셋은 시뮬레이터 기반입니다. 실제 로봇용 **ROS 노드셋은 여전히 [기여 모집](#5-기여하기) 슬롯으로 남아 있습니다** — 아키텍처상의 경로는 확립되어 있고 의도적이며, 필요한 ROS 측 컴포넌트는 이미 에코시스템에 존재합니다.

</details>

---

## 4. 시작하기

AgentCanvas를 사용하는 방법은 두 가지가 있으며, 둘 다 동일한 타입이 지정된 그래프 기반 위에서 동작합니다:

1. **그래프를 손으로 구축하고 실행** — 캔버스에서 노드를 구성하고, 시뮬레이터에 대해 에이전트를 실시간으로 실행하고, 대규모로 평가 (이 섹션의 나머지 부분).
2. **Agent Architecture Search (AAS)** — 코딩 에이전트에게 시드 그래프를 건네주고 아키텍처를 대신 탐색하게 하세요 ([바로가기](#44-에이전트-아키텍처-탐색-aas-실행)).

### 4.1 사전 요구사항

- Conda를 갖춘 Python 3.10+ (기본 `agentcanvas` env — ADR-platform-004)
- Node.js 18+
- *(선택, Habitat-Sim용)* 별도의 Python 3.8 env — `habitat-sim 0.1.7`은 여기서만 동작합니다; AgentCanvas는 서버 모드를 통해 이와 통신합니다. [INSTALL.md](INSTALL.md)를 참조하세요

### 4.2 웹 대시보드 실행

```bash
# Activate environment
conda activate agentcanvas

# Start backend (FastAPI :8000) + frontend (Vite :5173)
cd agentcanvas && bash run_dev.sh
```

캔버스 에디터에 접근하려면 [http://localhost:5173](http://localhost:5173)을 여세요.

### 4.3 평가 실행

동일한 eval 파이프라인이 네 가지 인터페이스를 통해 노출됩니다 — 손에 쥐고 있는 것에 따라 고르세요:

| # | 인터페이스 | 대상 | 적합한 용도 |
|---|-----------|----------|----------|
| 1 | **프론트엔드 Eval 페이지** | 사람                | 클릭 기반, UI에서 실시간 진행 상황 관찰 |
| 2 | **`/experiment:run` 슬래시 커맨드** | 코딩 에이전트 (Claude Code) | 프로파일로 게이트되는 GPU 수용, 자동 할당 포트, `:8000` 충돌 없음 |
| 3 | **MCP 서버** | 코딩 에이전트              | 대화형, 즉석 평가 — 슬래시 커맨드 오버헤드 없음 |
| 4 | **HTTP API** | 스크립트 / CI                | 직접 REST 호출, MCP 불필요 |

#### 1. 프론트엔드 Eval 페이지 — 사람을 위한

저장된 그래프를 **Eval** 페이지에서 열고, split + 에피소드 범위를 선택한 뒤, **Start**를 누르세요. 진행 상황은 WebSocket을 통해 실시간으로 스트리밍됩니다; 결과는 에피소드별 JSONL로 `outputs/eval_runs/{run_id}/episodes/ep{NNNN}/` 아래에 저장되며 (ADR-eval-004) Run Detail 패널에서 열람할 수 있습니다. 멀티 워커 env 팬아웃과 배치 추론은 폼에서 설정할 수 있습니다 (ADR-eval-002).

→ [Batch Eval 튜토리얼](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/tutorials/batch-eval.html)

#### 2. `/experiment:run` — 이 저장소의 코딩 에이전트를 위한

Claude Code를 사용할 때, `/experiment:run <profile> -- <cmd>`는 임의의 eval 호출을 백엔드의 `JobScheduler` 수용 게이트 (ADR-eval-003) 로 감쌉니다: 래퍼는 `.claude/commands/experiment/profiles.yaml`에 선언된 프로파일 하에서 VRAM을 확보하고, 할당된 포트에서 백엔드를 띄우며 (`BACKEND_URL=http://127.0.0.1:<port>`가 래핑된 명령으로 익스포트됩니다), 종료 시 슬롯을 해제합니다. 동반 명령: 실행 스냅샷용 `/experiment:status`, 우아한 취소용 `/experiment:teardown`.

→ [`.claude/commands/experiment/README.md`](../.claude/commands/experiment/README.md)

시드 그래프에 대해 제안 → 평가 → 최선을 유지를 여러 번 반복하는 완전한 아키텍처 탐색 설계 루프에 대해서는 아래의 [에이전트 아키텍처 탐색 실행](#44-에이전트-아키텍처-탐색-aas-실행)을 참조하세요.

#### 3. MCP 서버 — 코딩 에이전트를 위한

임의의 MCP 인식 클라이언트 (Claude Code, Cursor, …) 에 `agentcanvas-backend`를 등록하고, 타입이 지정된 도구 (`graph_list`, `eval_start`, `eval_status`, `eval_export`, `eval_stop`) 를 대화형으로 호출하세요. iter 트리 기록 관리 없이 — 빌려오거나 띄운 백엔드에 대한 순수한 eval만 있습니다.

→ [`agentcanvas/mcp_server/README.md`](../agentcanvas/mcp_server/README.md)

#### 4. HTTP API — 스크립트와 CI를 위한

스크립트, CI, 또는 비-MCP 환경을 위한 직접 REST:

```bash
curl -X POST http://localhost:8000/api/eval/v2/start \
  -H 'content-type: application/json' \
  -d '{"graph_name": "navgpt_ce", "split": "val_unseen", "worker_count": 4}'
# poll  GET /api/eval/v2/status
# fetch GET /api/eval/v2/export/{run_id}
```

→ [코딩 에이전트로 백엔드 구동하기](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/tutorials/coding-agent-backend.html) — 모든 프로그래밍 방식 모드를 나란히 두고 심층 탐구

### 4.4 에이전트 아키텍처 탐색 (AAS) 실행

고정된 그래프를 평가하는 것을 넘어, AgentCanvas는 **Agent Architecture Search** 의 기반입니다 — LLM 코딩 에이전트 *Optimizer* 가 시드 *Executor* 에 대한 그래프 편집을 반복적으로 제안하고, 각 후보를 시뮬레이터에서 평가하며, 개선분을 유지하는 개발 시점 루프입니다 (§1 — [왜 탐색 가능한 기반인가](#1-왜-agentcanvas-인가)). 에이전트가 타입이 지정된 그래프이기 때문에, 각 후보는 비용이 큰 롤아웃 이전에 실행되는 타입 검사된 패치이며, 노드별 에피소드 로그는 Optimizer가 점수 변화를 특정 모듈에 귀속시킬 수 있게 합니다.

<p align="center">
  <img src="../assets/readme/aas-search.gif" alt="임바디드 executor의 그래프를 탐색하는 코딩 에이전트 옵티마이저 — 편집을 제안하고, 실행하고, 이득을 유지한다" width="800">
  <br><sub><em>임바디드 executor의 그래프를 탐색하는 코딩 에이전트 옵티마이저 — 편집을 제안하고, 실행하고, 이득을 유지합니다.</em></sub>
</p>

탐색은 **메서드 시드(method-seeded)** 방식입니다: `iter_0`은 발표된 임바디드 메서드이며, 루프는 그 주위에서 그래프 수준의 편집을 탐색합니다. 세 가지 탐색 변형이 `.claude/commands/architect/` 아래에 Claude Code 스킬로 제공되며, 하나의 코딩 에이전트 하네스 (proposer → implementer → evaluator) 를 공유하고 proposer 로직 + 영속 메모리에서만 차이가 납니다:

| 변형 스킬 | 논문 이름 | 탐색 정책 |
|---|---|---|
| `myloop` | **KDLoop** | 4단계 THINK → CRITIC → EXPERIMENT → DISTILL 사이클, 타입이 지정된 메모리 + REFLECT 메타 단계 |
| `adas-subagent` | **ADAS** (port) | 평탄한 append-only 아카이브에 대한 Reflexion 스타일 제안 |
| `aflow` | **AFlow** (port) | 점수 소프트맥스 부모 선택 + anti-replay 메모리 |

```text
# In a Claude Code session on this repo — run KDLoop over the MapGPT executor
/architect:myloop:loop mapgpt_mp3d --goal "raise val_unseen SR"

# The ADAS / AFlow ports take the same  <graph> [<version>]  form
/architect:adas-subagent:loop smartway_ce
/architect:aflow:loop explore_eqa_hmeqa
```

현재 탐색을 위해 배선된 시드 그래프: `mapgpt_mp3d`, `smartway_ce` (VLN), `explore_eqa_hmeqa` (EQA), `voxposer_libero_monolithic` (VLA). 각 반복은 그 제안, 패치, eval 점수, 로그를 `outputs/design_runs/{variant}/{graph}/vN/iter_M/` 아래에 기록합니다.

→ [AAS 파이프라인 레퍼런스](https://jianzhou0420.github.io/AgentCanvas/pages/aas/index.html)

### 4.5 문서

```bash
# Serve the doc-site locally on :8092 (live-reload via SSE)
bash docs/run_dev.sh
```

---

## 5. 기여하기

두 종류의 기여 모두 환영합니다 — [CONTRIBUTING.md](../CONTRIBUTING.md)를 참조하세요:

- **콘텐츠 — 노드셋 & 그래프.** 도구 / 시뮬레이터 / 모델 (예: 실시간 3D Gaussian Splatting, voxel 기반 SLAM 시스템) 을 래핑하거나 메서드 (예: NavGPT, MapGPT) 를 인코딩하는 노드셋을 작성하거나, 기존 노드셋을 완전한 에이전트로 배선하는 그래프를 구성하세요. `workspace/`에 PR을 여세요; 리뷰는 가볍습니다.
- **코어 — UI, 백엔드, 프레임워크.** 버그 수정, 새로운 기능, 심지어 리팩터링도 환영합니다. 한 가지 부탁: 변경이 실제로 시간을 들일 만큼 크다면, 만들기 전에 방향을 맞출 수 있도록 먼저 [Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions)을 열어주세요.

모든 노드셋과 그래프는 아래 보드에 그 작성자/유지보수자로 기재됩니다 — 관련 논문이 있다면 인용 링크와 함께 — 따라서 여기에 기여한다고 해서 저작권을 잃지 않습니다. **AgentCanvas 프레임워크**, 그리고 첫 릴리스의 **메서드, 그래프, 환경 통합**은 **AC-Team**이 만들었습니다. 아래의 **파운데이션 모델과 정책**은 **서드파티**입니다 — AgentCanvas는 각각이 그래프에 꽂히도록 (사람 사용자와 AAS 옵티마이저 모두를 위해) 얇은 서버 모드 래퍼만을 제공합니다; 모든 모델의 크레딧은 그 원저자에게 있습니다 — 파운데이션 모델은 소스별 (transformers-native 대 `torch.hub` / `torchvision` / vendored 업스트림 repo) 로 나뉘어 **아래의 별도 표**로 빠졌으며, 모델별 전체 출처 표기는 Credits 페이지에 있습니다. 이 보드는 의도적으로 이름만 담습니다: 그래프별 검증 세부사항을 담은 **정식 인벤토리**는 [문서 사이트 Credits 페이지](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html)와 [VLN](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/vln-support-status.html) / [EQA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/eqa-support-status.html) / [VLA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/vla-support-status.html) 지원 상태 페이지에 있습니다.

### 크레딧

✅ 검증됨 — 논문 / 레퍼런스 구현을 재현함 · 🚧 엔드투엔드로 동작, 검증 진행 중

<table>
  <thead align="center">
    <tr>
      <th>환경</th>
      <th>메서드</th>
      <th>모델 &amp; 정책</th>
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
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/common/tools/basic-agent.html">Basic Agent 툴킷</a> ✅</li>
            </ul>
          </li>
          <li><b>EQA</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/openeqa.html">EM-EQA 베이스라인</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/explore-eqa.html">Explore-EQA</a> ✅</li>
              <li>ToolEQA 🚧</li>
            </ul>
          </li>
          <li><b>VLA (제로샷)</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/voxposer.html">VoxPoser-LIBERO</a> ✅</li>
            </ul>
          </li>
        </ul>
      </td>
      <td>
        <ul>
          <li><b>정책(Policies)</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-cma.html">CMA</a> ✅</li>
              <li>Octo (SIMPLER 베이스라인) ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-vla.html">VLA 프레임워크 (Pi0 / SmolVLA / DP / DROID-DP)</a> 🚧</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-adapters.html">R2R-CE 정책 레지스트리 (12개 변형)</a> 🚧</li>
            </ul>
          </li>
          <li><b>매핑</b> <sub><i>(AgentCanvas 제작)</i></sub>
            <ul>
              <li>TSDF 매핑 ✅</li>
              <li>시맨틱 씬 그래프 ✅</li>
            </ul>
          </li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

**파운데이션 모델** — **하나의 얇은 노드셋 셸** (지연 로드 · single-flight GPU · base64-npy 데이터플로 엔벨로프) 뒤에 래핑된 서드파티 모델로, 각각이 **사람 사용자**와 **AAS 옵티마이저** 양쪽을 위한 균일한 빌딩 블록입니다. *우리는 이들을 만들지 않았습니다 — 우리는 셸만 제공하며, 크레딧은 원저자에게 있습니다* (모델별 전체 출처 표기 + 논문은 [Credits 페이지](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html)에). 소스별 분류:

<table>
  <thead align="center">
    <tr>
      <th>transformers-native <sub>(<code>AutoModel</code> / <code>pipeline</code>에 대한 얇은 래퍼)</sub></th>
      <th>기타 소스 <sub>(<code>torch.hub</code> / <code>torchvision</code> / vendored 업스트림 repo)</sub></th>
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
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-segmentation.html">세그멘테이션 (Mask2Former)</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-florence2.html">Florence-2</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-depth-anything.html">Depth Anything V2</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-depthpro.html">DepthPro</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-normal.html">표면 법선 (Sapiens)</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-pointmap.html">포인트맵 (Sapiens 3D)</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-matching.html">SuperPoint + LightGlue</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-blip2.html">BLIP-2</a> + Faster R-CNN</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-instructblip.html">InstructBLIP</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-qwen2-5-vl.html">Qwen2.5-VL</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-qwen3-vl.html">Qwen3-VL</a> <sub>(이미지 + 비디오)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-internvl3.html">InternVL3</a> <sub>(이미지 + 비디오)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-gemma3.html">Gemma 3</a> <sub>(게이트됨)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-smolvlm2.html">SmolVLM2</a> <sub>(이미지 + 비디오)</sub></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-dinov2.html">DINOv2 / DINOv3</a> <sub>(torch.hub + transformers hf)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-grounding-dino.html">Grounding DINO</a> <sub>(groundingdino-py + transformers hf_tiny)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-opticalflow.html">옵티컬 플로 (RAFT)</a> <sub>(torchvision)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-vggt.html">VGGT</a> <sub>(업스트림 repo)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-cotracker.html">CoTracker</a> <sub>(업스트림 repo)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-detany3d.html">DetAny3D</a> <sub>(vendored)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-ram.html">RAM / RAM++</a> <sub>(recognize-anything)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-spatialbot.html">SpatialBot</a> <sub>(Bunny remote code)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-prismatic.html">Prismatic VLM</a> <sub>(업스트림 repo)</sub></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

**기여 모집** — 예약된 슬롯으로, 이를 완수한 사람에게 크레딧이 돌아갑니다 ([기여 방법](../CONTRIBUTING.md); ID는 [Credits 페이지](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html)의 로드맵 슬롯입니다):

<table>
  <thead align="center">
    <tr>
      <th>벤치마크</th>
      <th>메서드</th>
      <th>기능 &amp; 인프라</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul>
          <li>AI2-THOR — ALFRED / TEACh <i>(E4)</i></li>
          <li>RxR-CE — 다국어 VLN-CE <i>(E2)</i></li>
          <li>REVERIE — 원격 물체 그라운딩 <i>(E3)</i></li>
          <li>OpenEQA A-EQA — 능동 EQA <i>(E10)</i></li>
        </ul>
      </td>
      <td>
        <ul>
          <li>HAMT — 계층적 히스토리 트랜스포머 <i>(M5)</i></li>
          <li>DUET — 듀얼 스케일 그래프 트랜스포머 <i>(M6)</i></li>
          <li>InstructNav — 동적 CoN + 밸류 맵 <i>(M8)</i></li>
          <li>VLN-SIG — 하위 지시 그라운딩 <i>(M4)</i></li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Memory 노드셋 — 에피소드 회상 + 시맨틱 검색 <i>(F1)</i></li>
          <li>병렬 노드 실행 — Pregel superstep <i>(F3)</i></li>
          <li>그래프를 독립 실행형 Python으로 익스포트 <i>(F4)</i></li>
          <li>Docker 서버 모드 — Habitat / MP3D 컨테이너 <i>(F7)</i></li>
          <li>ROS 노드셋 — 실제 로봇 배포 (<a href="#3-sim-to-real-경로">§3</a>)</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


---

## 6. 인용

연구에서 AgentCanvas를 사용하신다면 다음을 인용해 주세요:

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

## 7. 라이선스

Apache License 2.0 — [LICENSE](../LICENSE)를 참조하세요.
