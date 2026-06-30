[English](../README.md) | [中文](README_zh.md) | [Español](README_es.md) | [日本語](README_ja.md) | **한국어**

# AgentCanvas

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Status: Research Preview](https://img.shields.io/badge/Status-Research_Preview-orange.svg)](#7-프로젝트-현황)
[![GitHub stars](https://img.shields.io/github/stars/jianzhou0420/AgentCanvas?style=social)](https://github.com/jianzhou0420/AgentCanvas/stargazers)

**임바디드 AI 연구를 위한 비주얼 에이전트 설계 플랫폼.** 하나의 타입이 지정된 그래프, 두 가지 역할: 임바디드 에이전트를 실행하는 *하네스(harness)*, 그리고 코딩 에이전트가 편집하고 검증하는 *스캐폴드(scaffold)*.

<p align="center">
  <img src="../assets/readme/editor-hero.gif" alt="AgentCanvas 에디터: MapGPT executor가 노드-와이어 그래프로 로드된 뒤, 실시간 R2R 에피소드가 엔드투엔드로 실행된다" width="720">
  <br><sub><em>에디터에서 실시간으로 녹화 — MapGPT executor가 로드된 뒤, 실제 R2R 에피소드가 엔드투엔드로 실행됩니다.</em></sub>
</p>

AgentCanvas는 연구자가 노드 그래프를 그리는 것만으로 임바디드 에이전트 — VLN, EQA, VLA 및 인접 태스크용 — 를 프로토타이핑할 수 있게 해줍니다. 이 그래프는 시뮬레이터 (Habitat-Sim, MatterSim, SAPIEN/ManiSkill2, MuJoCo/robosuite) 에 대해, 또는 원리적으로는 실세계 환경에 대해 실시간으로 실행됩니다. *하나의 JSON = 하나의 에이전트 = 하나의 그래프*: 에이전트의 동작은 명령형 코드가 아니라 데이터플로 그래프이며, 그래프가 곧 진실의 원천(source of truth)으로서 단일 JSON 파일로 저장되고 완전한 에이전트로 로드됩니다.

**대상 사용자**: 매번 실행 스택을 새로 작성하지 않고 임바디드 에이전트 아키텍처를 구성하고, 비교하고, 공유하고자 하는 연구자. 이 플랫폼은 VLN (Vision-and-Language Navigation), EQA (Embodied Question Answering), VLA (Vision-Language-Action) 정책 벤치마크를 다루며, 노드셋 모델을 통해 다른 임바디드 / 에이전트 설정에도 적응합니다.

> **상태**: 연구 프리뷰, 활발히 개발 중 · 46개의 ADR · 4개의 교체 가능한 팔레트에 걸친 40개 이상의 노드셋 — **env** (시뮬레이터), **method** (추론 루프), **model** (파운데이션 모델), **policy** (신경망 컨트롤러) · 캔버스 에디터, 멀티 스코프 반복을 지원하는 그래프 executor, 상태 컨테이너, 자동 호스팅 server-mode 노드셋, 훅 시스템, 실행 단위 서브프로세스 JobScheduler + 워커 풀 + 배치 추론, 그리고 통합 에러 버스 — 모두 프로덕션에서 동작 중.

> **버전 관리**: pre-1.0 (v0.x). v1.0은 공개 API가 안정화되면 (오픈소스화 + SemVer 하에 고정) 출시됩니다 — 어떤 논문과도 독립적으로. [Versioning Policy](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html)를 참조하세요.

> **기여**: 두 종류 모두 환영합니다. **콘텐츠** — `workspace/`에 PR을 보내 노드셋 (도구 또는 메서드) 을 작성하거나 그래프를 구성하세요; [크레딧](#크레딧) 보드에 기여자로 기재되며, 논문이 있다면 인용 링크가 함께 표시됩니다. **코어** — 프레임워크 (UI, 백엔드, 기능, 리팩터링) 를 개선하세요; 규모가 큰 작업이라면 먼저 [Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions)을 여세요. [CONTRIBUTING.md](../CONTRIBUTING.md)를 참조하세요.

---

## 목차

1. [왜 AgentCanvas인가?](#1-왜-agentcanvas인가) — 임바디드 에이전트를 위한 탐색 가능한 기반, 그리고 그것이 해결하는 고충점들
2. [기능](#2-기능) — *하나의 JSON = 하나의 에이전트* (§2.2) / *하나의 Python 클래스 = 하나의 노드* (§2.6) 원칙, 그리고 캔버스 에디터, 그래프 executor, 격리된 런타임 환경, 중첩 그래프, 상태 컨테이너, 훅
3. [Sim-to-Real 경로](#3-sim-to-real-경로) — 오늘은 시뮬레이터, 내일은 실제 로봇에서 동일한 에이전트 그래프를 — env-as-nodeset + 서버 모드 + ROS를 통해
4. [시작하기](#4-시작하기) — 사전 요구사항, 웹 대시보드 실행, 평가 실행, 아키텍처 탐색 실행, 문서 제공
5. [아키텍처](#5-아키텍처) — 프론트엔드 · 백엔드 · workspace · 시뮬레이터
6. [프로젝트 구조](#6-프로젝트-구조) — 최상위 디렉터리 맵
7. [프로젝트 현황](#7-프로젝트-현황) — 버전: v0.1 실험 → v0.2 프리뷰 → v1.0 → v2.0
8. [기여하기](#8-기여하기) — 도움이 가장 필요한 영역 · 크레딧
9. [인용](#9-인용) — 연구에서 AgentCanvas를 인용하는 방법
10. [라이선스](#10-라이선스) — Apache 2.0

---

## 1. 왜 AgentCanvas인가?

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

### 2.1 비주얼 캔버스 에디터

모든 노드 타입 — 환경, LLM, 추론 체인, 제어 게이트, 출력 뷰어 — 이 공존하는 ComfyUI 스타일의 평평한 워크스페이스. 사이드바에서 노드를 드래그하고, 서로 배선하고, Play를 누르세요.

### 2.2 그래프 실행 엔진

**하나의 JSON = 하나의 에이전트.** 에이전트의 동작 전체 — 노드, 배선, 설정, 상태 컨테이너, 훅 — 가 단일 JSON 파일입니다: 로드하고, 실행하고, 공유하고, diff하세요. 숨겨진 파이프라인 코드는 없습니다; 캔버스에서 보이는 것이 곧 실행되는 것입니다.

```jsonc
// 간소화된 예시 — 실제 그래프에는 상태 컨테이너, 훅, 그리고 더 많은 노드가 포함됩니다
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

### 2.3 격리된 런타임 환경

연구 도구는 종종 서로 충돌하는 Python 환경을 필요로 합니다 (Habitat은 Python 3.8이 필요하고, SLAM은 ROS가 필요합니다). 어떤 `BaseNodeSet`이든 **서버 모드**로 실행할 수 있습니다 — 프레임워크가 노드셋의 포트 정의로부터 HTTP 서버를 자동 생성하여, 자체 인터프리터에서 실행합니다. 추가 코드는 전혀 필요 없습니다:

```
# 동일한 노드셋 코드, 두 가지 배포 모드:
POST /api/components/nodesets/env_habitat/load              # 인프로세스
POST /api/components/nodesets/env_habitat/load?mode=server  # 별도 프로세스
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

### 2.7 훅 시스템

셸 명령이 각 노드 실행 전/후 및 그래프 라이프사이클 경계에서 발화합니다. 훅은 출력을 로깅하고, 입력을 검증하고, 노드를 차단하거나, 데이터를 수정할 수 있습니다 — 모두 그래프 노드를 변경하지 않고서. 훅은 저장된 그래프와 함께 이동합니다.

### 2.8 배치 평가 & 작업 큐

캔버스에서 실행되는 것과 동일한 그래프를, 수백 개의 에피소드에 걸쳐 점수를 매기는 eval 작업으로 제출할 수 있습니다. 백엔드가 소유한 `JobScheduler`가 모든 세션에 걸쳐 공유되는 VRAM 예산에 대해 수용을 게이트합니다 (ADR-eval-003); 수용된 각 실행은 자체 서브프로세스이므로, 백엔드 재시작이 진행 중인 eval을 죽이지 않습니다. 에피소드별 로그는 자기 완결적인 레이아웃에 저장되므로 (ADR-eval-004), 팀원이 재실행 없이 임의의 단일 에피소드를 재생할 수 있습니다.

### 2.9 실시간 관측성

모든 스텝이 관측, 추론, 행동, 메트릭을 WebSocket을 통해 스트리밍하며, `execution_id`로 라우팅되어 동시 실행이 스트림을 교차시키지 않습니다. 어떤 출처의 에러든 — 노드 예외, 서버 모드 서브프로세스 크래시, HTTP 실패 — 통합된 `ErrorBus`를 거쳐 Report 탭 항목 + 토스트로 표면화됩니다 (ADR-observability-004). (React 렌더 에러는 클라이언트 측 에러 바운더리에서 포착됩니다.)

---

## 3. Sim-to-Real 경로

AgentCanvas는 이식성을 염두에 두고 설계되었습니다: 단일 에이전트 그래프는 오늘은 시뮬레이터에 대해 실행되고, 미래에는 그래프 수준의 변경 없이 실제 로봇으로 이전될 수 있습니다. 이 속성은 두 가지 아키텍처 결정에서 비롯됩니다 — 환경 자체가 노드셋이라는 점 (ADR-components-002), 그리고 어떤 노드셋이든 *서버 모드*를 통해 격리된 런타임에서 실행될 수 있다는 점 (ADR-server-001).

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

현재 출시된 모든 환경 노드셋은 시뮬레이터 기반입니다. 실제 로봇용 **ROS 노드셋은 여전히 [기여 모집](#8-기여하기) 슬롯으로 남아 있습니다** — 아키텍처상의 경로는 확립되어 있고 의도적이며, 필요한 ROS 측 컴포넌트는 이미 에코시스템에 존재합니다.

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
# 환경 활성화
conda activate agentcanvas

# 백엔드 (FastAPI :8000) + 프론트엔드 (Vite :5173) 시작
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
# 폴링  GET /api/eval/v2/status
# 가져오기 GET /api/eval/v2/export/{run_id}
```

→ [코딩 에이전트로 백엔드 구동하기](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/tutorials/coding-agent-backend.html) — 모든 프로그래밍 방식 모드를 나란히 두고 심층 탐구

### 4.4 에이전트 아키텍처 탐색 (AAS) 실행

고정된 그래프를 평가하는 것을 넘어, AgentCanvas는 **Agent Architecture Search** 의 기반입니다 — LLM 코딩 에이전트 *Optimizer* 가 시드 *Executor* 에 대한 그래프 편집을 반복적으로 제안하고, 각 후보를 시뮬레이터에서 평가하며, 개선분을 유지하는 개발 시점 루프입니다 (§1 — [왜 탐색 가능한 기반인가](#1-왜-agentcanvas인가)). 에이전트가 타입이 지정된 그래프이기 때문에, 각 후보는 비용이 큰 롤아웃 이전에 실행되는 타입 검사된 패치이며, 노드별 에피소드 로그는 Optimizer가 점수 변화를 특정 모듈에 귀속시킬 수 있게 합니다.

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
# 이 저장소의 Claude Code 세션에서 — MapGPT executor에 대해 KDLoop를 실행
/architect:myloop:loop mapgpt_mp3d --goal "raise val_unseen SR"

# ADAS / AFlow 포트는 동일한  <graph> [<version>]  형식을 받습니다
/architect:adas-subagent:loop smartway_ce
/architect:aflow:loop explore_eqa_hmeqa
```

현재 탐색을 위해 배선된 시드 그래프: `mapgpt_mp3d`, `smartway_ce` (VLN), `explore_eqa_hmeqa` (EQA), `voxposer_libero_monolithic` (VLA). 각 반복은 그 제안, 패치, eval 점수, 로그를 `outputs/design_runs/{variant}/{graph}/vN/iter_M/` 아래에 기록합니다.

→ [AAS 파이프라인 레퍼런스](https://jianzhou0420.github.io/AgentCanvas/pages/aas/index.html)

### 4.5 문서

```bash
# 문서 사이트를 로컬 :8092에서 제공 (SSE를 통한 라이브 리로드)
bash docs/run_dev.sh
```

---

## 5. 아키텍처

```
Frontend (React 18 + React Flow + Zustand)
    |
    |  REST + WebSocket
    v
Backend (FastAPI + Python 3.10+)
    |
    |-- WorkspaceComponentRegistry  -->  workspace/  (auto-discovery)
    |-- GraphExecutor   -->  graph execution (DAG + cyclic + multi-scope)
    |-- AutoServerApp      -->  server-mode nodesets (isolated envs)
    |-- HookRunner         -->  pre/post interceptors
    |-- JobScheduler       -->  subprocess-per-run eval admission (ADR-eval-003)
    |-- ErrorBus           -->  unified error reporting (ADR-observability-004)
    v
Simulators (Habitat-Sim, MatterSim/MP3D, HM3D, SAPIEN/ManiSkill2, MuJoCo/robosuite, ...)
```

**핵심 설계**: 프레임워크는 **도메인 지식을 전혀 갖지 않습니다** (ADR-platform-001). 모든 도메인 특화 코드 — VLN 정책, LLM 프롬프트, 내비게이션 도구, 환경 래퍼 — 는 `workspace/`에 존재합니다. 프레임워크는 런타임에 기반 클래스 상속을 통해 컴포넌트를 검색합니다. 도메인 코드를 직접 임포트하지 않습니다; 임포트 경계는 `agentcanvas/backend/app/test_import_boundary.py`에 의해 강제됩니다.

---

## 6. 프로젝트 구조

```
vlnworkspace/                  # repo root (legacy name; the platform is "AgentCanvas")
├── agentcanvas/               # Full-stack web application
│   ├── backend/app/         #   FastAPI backend (execution engine, APIs, services, errors)
│   ├── frontend/src/        #   React + TypeScript (canvas editor)
│   └── mcp_server/          #   MCP server for coding-agent integration
├── workspace/                 # User workspace — all domain components (auto-discovered)
│   ├── nodesets/            #   Nodesets by palette: env / method / model / policy (+ common, _upstream)
│   ├── graphs/              #   Saved agent graphs (kind="graph")
│   ├── graph_nodes/         #   Reusable composite nodes (kind="node")
│   ├── nodes/               #   Standalone BaseCanvasNode subclasses
│   ├── architect/           #   AAS search profiles + run scaffolding
│   └── hooks.json           #   Workspace-level hook definitions
├── data/                      # Datasets, model weights (gitignored)
├── outputs/                   # Eval + design-run outputs (eval_runs/, design_runs/, …)
├── docs/                      # Hand-authored HTML doc-site (run_dev.sh → :8092)
├── third_party/               # Git submodules (habitat-lab, VLN-CE, MatterSim, vla_workspace, …)
└── scripts/                   # Data setup + install scripts
```

---

## 7. 프로젝트 현황

AgentCanvas는 **pre-1.0이며 활발히 개발 중**입니다. 상태는 진행 중인 기능 체크리스트가 아니라 버전으로 추적됩니다 — 자세한 내용은 [Versioning Policy](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html)와 [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html)를 참조하세요.

- **v0.1 — AAS 실험.** 논문의 Agent Architecture Search 실행이 수행된 스냅샷 — 그 결과들을 위한 재현성 앵커이며, 공개 릴리스는 아닙니다.
- **v0.2 — 연구 프리뷰 (현재).** 첫 오픈소스 릴리스: 캔버스 에디터, 그래프 executor (DAG + 순환 + 멀티 스코프), 상태 컨테이너, 자동 호스팅 server-mode 노드셋, 배치 eval, 그리고 40개 이상의 노드셋 (env / method / model / policy) 이 모두 프로덕션에서 동작합니다. 공개 API는 아직 고정되지 않았으므로, 마이너 릴리스가 이를 깨뜨릴 수 있습니다. 출시된 인벤토리: [§2 기능](#2-기능) 그리고 [VLN](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/vln-support-status.html) / [EQA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/eqa-support-status.html) / [VLA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/vla-support-status.html) 지원 상태 페이지.
- **v1.0 — 진행 중.** 공개 API가 안정화되면 출시됩니다 — 오픈소스화되고 SemVer 하에 고정되며, 어떤 논문과도 독립적입니다.
- **v2.0 — 미래.** 토폴로지를 변형하는 실행: 무경계 서브에이전트 생성, 런타임 리스트에 대한 런타임 팬아웃, 런타임에 출현하는 새로운 도구 타입, 자기 수정 그래프. 논지와 미해결 질문에 대해서는 [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html) §2를 참조하세요.

---

## 8. 기여하기

두 종류의 기여 모두 환영합니다 — [CONTRIBUTING.md](../CONTRIBUTING.md)를 참조하세요:

- **콘텐츠 — 노드셋 & 그래프.** 도구 / 시뮬레이터 / 모델 (예: 실시간 3D Gaussian Splatting, voxel 기반 SLAM 시스템) 을 래핑하거나 메서드 (예: NavGPT, MapGPT) 를 인코딩하는 노드셋을 작성하거나, 기존 노드셋을 완전한 에이전트로 배선하는 그래프를 구성하세요. `workspace/`에 PR을 여세요; 리뷰는 가볍습니다.
- **코어 — UI, 백엔드, 프레임워크.** 버그 수정, 새로운 기능, 심지어 리팩터링도 환영합니다. 한 가지 부탁: 변경이 실제로 시간을 들일 만큼 크다면, 만들기 전에 먼저 [Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions)을 열어 방향을 맞출 수 있게 해주세요.

모든 노드셋과 그래프는 아래 크레딧 보드에 작성자/유지보수자로 기재됩니다 — 관련 논문이 있다면 인용 링크와 함께 — 따라서 여기에 기여한다고 해서 저작권을 잃지 않습니다.

### 크레딧

<table>
<tr><th>컴포넌트</th><th>작성자</th></tr>
<tr>
<td><b>AgentCanvas 프레임워크</b></td>
<td><a href="https://github.com/jianzhou0420">@jianzhou0420</a></td>
</tr>
<tr>
<td>

<details open>
<summary><b>첫 릴리스</b> — 번들 노드셋, 레퍼런스 그래프, 문서 사이트</summary>

<br>

<b>시뮬레이터 / 환경</b>

- Habitat (VLN-CE 연속 내비게이션)
- Matterport3D / MatterSim (이산 파노라마 내비게이션)
- HM-EQA (임바디드 QA 환경)
- OpenEQA (임바디드 QA 벤치마크, EM-EQA 모드)
- SIMPLER (SAPIEN / ManiSkill2 real-to-sim VLA 평가)
- LIBERO (MuJoCo / robosuite 조작, 5개 스위트)

<b>에이전트 메서드 / 추론</b>

<i>EQA</i>

- OpenEQA EM-EQA 베이스라인 — blind-LLM / single-frame / multi-frame (`openeqa_em_*.json`) ✅ 모두 검증됨; multi-frame LLM-Match 0.7025 vs 논문 0.466 (gpt-4o 추론기+판정기가 논문의 gpt-4 / gpt-4-vision-preview를 능가)
- Explore-EQA (HM-EQA에서 Prismatic 고정 프런티어 탐험) ✅ 검증됨 — SR 0.42로 0.44 베이스라인을 재현
- ToolEQA (HM-EQA 전용 — PortBench v1 기반) — 2026-06-08 모놀리스 우선으로 재작업; 엔드투엔드로 동작 (ReAct + 융합 TSDF go_next + 서버 모드 HTTP 상의 Qwen2.5-VL/DetAny3D), SR 튜닝 진행 중

<i>VLN</i>

- NavGPT (LLM thought–action 추론 프리미티브) ✅ gpt-4에서 동작 (비쌈); 다른 LLM은 미검증 (gpt-4o는 긴 ReAct 프롬프트에서 퇴행하는 것으로 알려짐)
- MapGPT (언어적 topo-map LLM 에이전트, ACL 2024) ✅ 검증됨 — MapGPT_72에서 SR 0.477 / 0.463
- SmartWay-mono (VLN-CE 웨이포인트 예측기) ✅ 논문 비교 가능 — SR 0.270 vs 논문 0.29
- SmartWay-CE ✅ silent-completion 레이스 수정됨; 20-워커 평가에서 엔드투엔드로 동작
- SpatialNav (공간 그래프 내비게이션) ❌ 미검증 — SR=0
- Open-Nav (오픈 어휘 내비게이션) ❌ 미검증 — SR=0
- DiscussNav (멀티 LLM 토론, 경계가 있는 팬아웃) ❓ 진행 중 — fitness가 아직 논문 비교 가능 수준까지 도달하지 못함
- Three-Step Nav (제로샷 웨이포인트 내비, Open-Nav를 서브클래싱) ❓ 엔드투엔드 검증됨 — SR 0.10 / oracle 0.30 @10ep; 논문 비교 가능 튜닝 대기 중
- AO-Planner (SAM + LLM + 3D 경로 플래너, AAAI 2025) ❓ 진행 중 — 노드셋 출시됨, 평가 대기 중
- Basic Agent (기초 VLN 툴킷 — 5개 카테고리에 걸친 11개 노드)

<i>VLA</i>

- VLA 특화 메서드 (Pi0 / SmolVLA / DP / DROID-DP / Octo / VoxPoser-LIBERO) 는 아래의 <b>정책(Policies)</b> 에 있습니다 — 이들은 추론형이 아니라 정책형 (env-observation → action) 이므로, 태스크 패밀리가 아니라 코드 구조로 그룹화됩니다

<b>지각 / 비전</b>

- SAM (Segment Anything)
- BLIP-2 + Faster R-CNN (캡셔닝 & 검출)
- RAM (recognize-anything model)
- SpatialBot (깊이 인식 VLM)
- Prismatic VLM (토큰 우도 점수화 + 자유 형식 생성)
- TSDF 매핑
- 시맨틱 씬 그래프

<b>정책(Policies)</b>

- CMA (Cross-Modal Attention VLN-CE 베이스라인) ✅ 검증됨 — `straightforward.json`이 verified/로 승격됨, SR 0.38 / SPL 0.348, 네이티브와 비트 단위로 동일
- Octo (VLA 제너럴리스트, 네이티브 SIMPLER 베이스라인) ✅ 베이스라인이 `octo_simpler.json`에서 동작
- 범용 VLA 프레임워크 (Pi0 / SmolVLA / DP / DROID-DP 어댑터) ✅ Pi0 검증됨 — `vla_policy_libero` libero_spatial task 0에서 5/5; SIMPLER 변형은 TBD
- VoxPoser-LIBERO (LMP + voxel-cost-map + OSC) ✅ 엔드투엔드 검증됨 (grasp + transport); SR 기록됨
- VLN-CE 정책 어댑터 (12개 변형 R2R-CE 레지스트리 — 2개는 업스트림 릴리스, 10개 ablation은 플레이스홀더 표시)

<b>문서 사이트</b> — 손으로 작성한 HTML (MkDocs 폐기 후 2026-05-18), 46개의 ADR, 용어집, 역량 페이지, 튜토리얼, 설계 문서 포함

</details>

</td>
<td><a href="https://github.com/jianzhou0420">@jianzhou0420</a></td>
</tr>
<tr>
<td><b>벤치마크:</b> AI2-THOR <i>(ALFRED / TEACh — E4)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>벤치마크:</b> RxR-CE <i>(다국어 VLN-CE — E2)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>벤치마크:</b> REVERIE <i>(원격 물체 그라운딩 — E3)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>벤치마크:</b> OpenEQA A-EQA <i>(능동 EQA 모드 — E10)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>메서드:</b> HAMT <i>(계층적 히스토리 트랜스포머 — M5)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>메서드:</b> DUET <i>(듀얼 스케일 그래프 트랜스포머 — M6)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>메서드:</b> MapGPT (metric-grid 변형) <i>(LLM + 깊이 기반 점유 — M2; 출시된 linguistic-topo 변형과는 구별됨)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>메서드:</b> InstructNav <i>(Dynamic CoN + Multi-Sourced Value Maps, CoRL 2024 — M8)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>메서드:</b> VLN-SIG <i>(하위 지시 그라운딩 — M4)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>기능:</b> Memory 노드셋 <i>(에피소드 회상 + 시맨틱 검색 — F1)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>기능:</b> 병렬 노드 실행 <i>(Pregel superstep 모델 — F3)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>기능:</b> 그래프를 독립 실행형 Python으로 익스포트 <i>(헤드리스 배치 평가 — F4)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>인프라:</b> Docker 서버 모드 <i>(Habitat / MP3D 컨테이너 — F7)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
<tr>
<td><b>인프라:</b> ROS 노드셋 <i>(서버 모드를 통한 실제 로봇 배포 — §3)</i></td>
<td><i><a href="../CONTRIBUTING.md">기여 모집</a></i></td>
</tr>
</table>


---

## 스타 히스토리

<a href="https://star-history.com/#jianzhou0420/AgentCanvas&Date">
  <img src="https://api.star-history.com/svg?repos=jianzhou0420/AgentCanvas&type=Date" alt="스타 히스토리 차트" width="600">
</a>

---

## 9. 인용

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

---

## 10. 라이선스

Apache License 2.0 — [LICENSE](../LICENSE)를 참조하세요.
