[English](../README.md) | [中文](README_zh.md) | [Español](README_es.md) | **日本語** | [한국어](README_ko.md)

<div align="center">

# AgentCanvas

### Automating the Design of Embodied Agent Architectures

**Jian Zhou · Sihao Lin · Jin Li · Shuai Fu · Gengze Zhou · Qi Wu**

Australian Institute for Machine Learning, University of Adelaide

<p>
  <a href="https://arxiv.org/abs/2606.30111"><img src="https://img.shields.io/badge/arXiv-2606.30111-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://jianzhou0420.github.io/src/works/AgentCanvas/index.html"><img src="https://img.shields.io/badge/Project%20Page-1f6feb?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://jianzhou0420.github.io/src/works/AgentCanvas/paper.html"><img src="https://img.shields.io/badge/Paper%20Page-1f6feb?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Paper Page"></a>
  <a href="#9-引用"><img src="https://img.shields.io/badge/BibTeX-Cite-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white" alt="BibTeX"></a>
</p>

<img src="../assets/readme/editor-hero.gif" alt="AgentCanvas エディタ: MapGPT executor がノードとワイヤのグラフとして読み込まれ、続いてライブの R2R エピソードがエンドツーエンドで実行される様子" width="760">

<sub><em>エディタ上でライブ収録 — MapGPT executor が読み込まれ、続いて実際の R2R エピソードがエンドツーエンドで実行されます。</em></sub>

</div>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Status: Research Preview](https://img.shields.io/badge/Status-Research_Preview-orange.svg)](#7-プロジェクトステータス)
[![GitHub stars](https://img.shields.io/github/stars/jianzhou0420/AgentCanvas?style=social)](https://github.com/jianzhou0420/AgentCanvas/stargazers)

**身体性 AI 研究のためのビジュアルなエージェント設計プラットフォーム。** 1 つの型付きグラフに 2 つの役割を持たせます。すなわち、身体性エージェントを実行する *ハーネス (harness)* と、コーディングエージェントが編集・検証する *スキャフォールド (scaffold)* です。

AgentCanvas を使うと、研究者は VLN・EQA・VLA やその周辺タスク向けの身体性エージェントを、ノードグラフを描くことでプロトタイピングできます。これらのグラフは、シミュレータ (Habitat-Sim、MatterSim、SAPIEN/ManiSkill2、MuJoCo/robosuite)、あるいは原理的には実世界のセットアップに対してリアルタイムに実行されます。*1 つの JSON = 1 つのエージェント = 1 つのグラフ*。エージェントの振る舞いは命令型コードではなくデータフローグラフであり、グラフこそが信頼できる唯一の情報源 (source of truth) として、単一の JSON ファイルに保存され、完全なエージェントとして読み込まれます。

**対象ユーザー**: 実行スタックを毎回書き直すことなく、身体性エージェントのアーキテクチャを組み立て、比較し、共有したい研究者。本プラットフォームは VLN (Vision-and-Language Navigation)、EQA (Embodied Question Answering)、VLA (Vision-Language-Action) のポリシーベンチマークをカバーし、nodeset モデルを通じてその他の身体性 / エージェント的な設定にも適応します。

> **ステータス**: リサーチプレビュー、活発に開発中 · 46 個の ADR · 入れ替え可能な 4 つのパレットにまたがる 40 以上の nodeset — **env** (シミュレータ)、**method** (推論ループ)、**model** (基盤モデル)、**policy** (ニューラルコントローラ) · キャンバスエディタ、マルチスコープ反復に対応したグラフ executor、ステートコンテナ、自動ホストされる server-mode nodeset、フックシステム、run ごとにサブプロセスを起動する JobScheduler + ワーカープール + バッチ推論、そして統一されたエラーバス — すべて本番稼働中。

> **バージョニング**: 1.0 以前 (v0.x)。v1.0 は公開 API が安定したとき (オープンソース化され、SemVer の下で凍結されたとき) にリリースされます — 論文とは独立しています。[Versioning Policy](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html) を参照してください。

> **コントリビューション**: 2 種類あり、どちらも歓迎します。**Content** — nodeset (ツールまたはメソッド) を書く、あるいはグラフを組み立て、`workspace/` への PR で貢献します。あなたは [クレジット](#クレジット) ボードにクレジットされ、論文があれば引用リンクも付きます。**Core** — フレームワーク (UI、バックエンド、機能、リファクタリング) を改善します。大きな変更については、まず [Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions) を開いてください。[CONTRIBUTING.md](../CONTRIBUTING.md) を参照してください。

---

## 目次

1. [なぜ AgentCanvas なのか?](#1-なぜ-agentcanvas-なのか) — 身体性エージェントのための探索可能な基盤と、それが解決する課題
2. [機能](#2-機能) — *1 つの JSON = 1 つのエージェント* (§2.2) / *1 つの Python クラス = 1 つのノード* (§2.6) の原則、加えてキャンバスエディタ、グラフ executor、分離されたランタイム環境、ネストグラフ、ステートコンテナ、フック
3. [Sim-to-Real への道筋](#3-sim-to-real-への道筋) — 同じエージェントグラフを今日はシミュレータで、明日は実ロボットで — env-as-nodeset + server mode + ROS を通じて
4. [はじめに](#4-はじめに) — 前提条件、Web ダッシュボードの起動、評価の実行、アーキテクチャ探索の実行、ドキュメントの配信
5. [アーキテクチャ](#5-アーキテクチャ) — フロントエンド · バックエンド · workspace · シミュレータ
6. [プロジェクト構成](#6-プロジェクト構成) — トップレベルのディレクトリ構成図
7. [プロジェクトステータス](#7-プロジェクトステータス) — バージョン: v0.1 実験 → v0.2 プレビュー → v1.0 → v2.0
8. [コントリビューション](#8-コントリビューション) — 最も助けが必要な領域 · クレジット
9. [引用](#9-引用) — 研究で AgentCanvas を引用する方法
10. [ライセンス](#10-ライセンス) — Apache 2.0

---

## 1. なぜ AgentCanvas なのか?

身体性エージェント — VLN・EQA・VLA にまたがる — は、基盤モデルを知覚・マッピング・記憶・プランニング・行動と組み合わせて構築されることが増えています。構造が重みの中に吸収されてしまうエンドツーエンドのポリシーとは異なり、このアーキテクチャは *明示的で編集可能* です。そこから、AgentCanvas が中心に据える問い — **エージェント設計は、手作業で構築するのではなく探索できるのか?** — が生まれます。そして、その道のりで乗り越えなければならない 2 つの課題の山も併せて立ちはだかります。

<details>
<summary><b>エージェントアーキテクチャは手作業で構築されている — そして探索できるはず</b></summary>

<br>

それぞれのエージェントは、あらゆる接合点 — センサーの抽象化、地図表現、記憶状態、プロンプト構造、プランナーのトポロジー、モデルの配置、行動インターフェース — での選択を手作業で、たいていは単一のベンチマーク向けに固定します。基盤モデルと身体性ツールが増えるにつれ、その空間は手作業の反復ではカバーしきれない速さで広がっていきます。そのため、自然な一手は、手でチューニングするのではなく探索することです。

Agent Architecture Search (AAS) はテキスト領域のエージェントについては既にこれを実現していますが、身体性への移行はタダではありません。状態を持つシミュレータ、ノイズの多いマルチエピソードのスコアリング、長い知覚 / 行動のトレース、そして既製の身体性プリミティブのパレットが存在しないこと。AgentCanvas は、その欠けている基盤 — コーディングエージェントが読み、編集し、実行し、検証できるスキャフォールド — を提供しようとする私たちの試みであり、身体性エージェントについてもエージェント設計の探索が可能になることを目指しています。

</details>

<details>
<summary><b>身体性に特有の課題</b></summary>

<br>

- **現代の身体性スタックは分厚い** — 動作する身体性エージェントには、LLM による推論 + ツール利用 + シミュレータとの結合 + 空間ツールが、すべて配線されて必要になります。これをプロジェクトごとにゼロから構築するのは法外なコストがかかり、その労力の大半は、検証したいアイデアそのものではなく実行レイヤーに費やされます。
- **エンジニアリングの悪夢** — 身体性エージェントは 1 つのモデルではなく、システム全体です — 状態を持つシミュレータに加え、重いモデルとツールのスタック。それを動かすだけでも、ましてやベンチマークが必要とする規模で動かすとなれば、それ自体が困難なエンジニアリングの仕事です。
  - **Python 環境地獄** — すべての部分を満たす単一の Python 環境は存在しません。各シミュレータ・VLM・検出器・ポリシーがそれぞれ衝突する CUDA / torch / Python を固定するため、それらすべてが共有できる 1 つのランタイムを見つけることはしばしば不可能です — 結局、エージェントを読み込むためだけに、互換性のない複数の環境を維持するはめになります。
  - **バッチ処理** — 各ワーカーのシミュレータは、それぞれ独自のペースで進む別個の状態付きプロセスです。モデルはバッチ化できてもシミュレータはできないため、あらゆるステップが「観測を非同期で集める → バッチ推論する → 行動を散らす」というダンスになります。
  - **その他のインフラ** — ログに残して再生可能にしなければならないマルチモーダルな軌跡、*必ず* クラッシュする数時間に及ぶ GPU run のチェックポイント / 再開、そしてプロセス境界をまたいだデバッグ。

  1 本の論文の研究サイクルを通じて、研究者はアルゴリズムそのものに集中する代わりに、このエンジニアリングコストをあまりにも多く支払うことになります。
- **隠れたグラウンドトゥルース依存** — 多くの手法は、実際の知覚ではなく、シミュレータが提供するグラウンドトゥルース (物体姿勢、セマンティックラベル、navigability) にひそかに依存しています。それが実験を制御するための正当な手段である場合もありますが — 意図的かどうかに関わらず — 論文では言及されないことがしばしばあります。

</details>

<details>
<summary><b>AI 研究によくある課題 (ここではさらに増幅される)</b></summary>

<br>

- **再現不可能な実装** — どの論文も、それぞれ異なるコードベースでエージェントをゼロから構築するため、手法を公平に比較したり結果を再現したりするのは骨が折れます — しかもその多くは **`Code coming SOON`** (**S**omeday, **O**r **O**bviously **N**ever — いつか、あるいは見ての通り絶対に来ない) です。
- **論文 ≠ コード** — 論文はきれいなフロー図を見せますが、実際のコードは文書化されていない形で乖離しています。論文を再現するということは、その実装をリバースエンジニアリングするということです。
- **密結合なコード** — ドメインロジック (プロンプト、ツール、ポリシー) がインフラと絡み合っています。1 つのコンポーネントを差し替えると、パイプラインを書き直すことになります。

</details>

---

## 2. 機能

> **完全なリファレンスはドキュメントに** — 以下のほとんどの機能には実装ページ (仕組み · 主要ファイル · 現在の状況) があります: **[The Nine Capabilities →](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/capabilities/index.html)**

### 2.1 ビジュアルキャンバスエディタ

すべてのノードタイプが共存する ComfyUI スタイルのフラットなワークスペース — 環境、LLM、推論チェーン、制御ゲート、出力ビューア。サイドバーからノードをドラッグし、配線し、Play を押します。

### 2.2 グラフ実行エンジン

**1 つの JSON = 1 つのエージェント。** エージェントの振る舞いのすべて — ノード、配線、設定、ステートコンテナ、フック — は単一の JSON ファイルです。読み込み、実行し、共有し、差分を取れます。隠れたパイプラインコードはありません。キャンバスで見えるものがそのまま実行されます。

```jsonc
// 簡略版 — 実際のグラフにはステートコンテナ、フック、さらに多くのノードが含まれます
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

エンジンはそのグラフを実行します。ノードは固定の順序ではなく、入力が到着したときに発火します。同じエンジンが、AgentCanvas v1 がサポートするすべてのグラフ形状を扱います — v1 の有界静的トポロジー (bounded-static-topology) パラダイムがカバーするエージェント形態の完全な一覧については、[`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html) を参照してください。

- **DAG ワークフロー** — 非巡回パイプラインのための単一フォワードパス
- **巡回エージェントループ** — **2 ピボット (two-pivot)** モデルによる observe-think-act-repeat: 両側を持つ **`IterIn`** (左側に run 開始時の init 入力、右側に反復ごとの loop-carry) と **`IterOut`** を用い、グラフを見た目上は非巡回に保ちながら実行時の巡回を可能にします (ADR-dataflow-008。これは ADR-dataflow-006 の従来の 3 ピボット `initialize`/IterIn/IterOut を 2 つに畳み込んだものです)
- **マルチスコープ反復** — 1 つのフラットグラフ内に共存する N 個の `(IterIn, IterOut)` ペア (ADR-dataflow-007 / ADR-executor-003)
- **ReAct ループ** — `LLMCallNode` のサブクラス内に隠すか、ルーター + 事前宣言された N 個のツール分岐として明示的に表現します
- **有界マルチエージェント** — 固定 N または `K_max` で上限を設けたファンアウト (例: DiscussNav 風のディベート、AutoGen 風の固定ロール)
- **Plan-and-Execute** — 有界なツールプール上で、ルーターによりディスパッチされる

### 2.3 分離されたランタイム環境

研究ツールはしばしば、互いに衝突する Python 環境を必要とします (Habitat は Python 3.8 を、SLAM は ROS を必要とします)。あらゆる `BaseNodeSet` は **server mode** で実行できます — フレームワークが nodeset のポート定義から HTTP サーバを自動生成し、それぞれ独自のインタプリタで動かします。追加コードはゼロです。

```
# 同じ nodeset コードで、2 つのデプロイモード:
POST /api/components/nodesets/env_habitat/load              # プロセス内
POST /api/components/nodesets/env_habitat/load?mode=server  # 別プロセス
```

### 2.4 ネストグラフシステム

任意のキャンバスグラフを **graph node** として保存し、別のキャンバスに再利用可能なブロックとしてドラッグできます。これにより階層的なエージェントアーキテクチャ — サブエージェントの graph node を内包する高レベルプランナー — が可能になります。スナップショットのセマンティクス: 各インスタンスはディープコピーです。

### 2.5 ステートコンテナシステム

デュアルワイヤ (dual-wire) アーキテクチャを通じた、エージェントループの反復をまたぐ共有永続状態:

- **データエッジ** はノード間のデータフロー (IMAGE、TEXT、ACTION、POSE、…) を運びます
- **アクセスグラント (access grants)** はノードに **StateContainers** の読み書きを許可します — これらは名前付きエントリ、設定可能なリデューサ (Accumulator、LastWrite、Counter)、そして適切なシグナル境界でメモリを自動クリアする **Lifetime** 軸 (`forever` / `step` / `episode` / `run` / `custom`) を持つ、キャンバス上に可視の要素です (ADR-dataflow-002、ADR-dataflow-004)

→ [State Containers 設計ドキュメント](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/graph/state-containers.html)

### 2.6 Python で定義されるノード

**1 つの Python クラス = 1 つのノード。** すべてのキャンバスノード — ツール、環境、スキル、ポリシー — は単一の Python クラスです。ポートを宣言し、`forward()` を実装し、ファイルを `workspace/` に置けば、プラットフォームが自動検出します。フレームワークの変更も、TypeScript も、登録のボイラープレートも不要です。

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

そのノードはキャンバスのサイドバーに現れ、ポートタイプが一致する他の任意のノードと配線できます。その見た目も Python 駆動です。`GenericBlockRenderer` が `NodeUIConfig` からあらゆるノードを自動的にレンダリングします — 色、レイアウト、インラインの設定コントロール (スライダー、ドロップダウン、テキストフィールド)、表示ウィジェット — そのため、カスタムの React コンポーネントは不要です。

### 2.7 フックシステム

シェルコマンドが、各ノード実行の前後やグラフのライフサイクル境界で発火します。フックは出力のログ記録、入力の検証、ノードのブロック、データの変更ができます — すべてグラフノードを変更することなく。フックは保存されたグラフとともに移動します。

### 2.8 バッチ評価とジョブキュー

キャンバス上で動作するのと同じグラフを、数百のエピソードにわたって採点する評価ジョブとして投入できます。バックエンドが所有する `JobScheduler` が、全セッションで共有される VRAM 予算に対して受け入れを制御します (ADR-eval-003)。受け入れられた各 run はそれぞれ独自のサブプロセスなので、バックエンドの再起動が実行中の評価を巻き込んで落とすことはありません。エピソードごとのログは自己完結したレイアウトに格納されるため (ADR-eval-004)、チームメイトは再実行なしに任意の単一エピソードを再生できます。

### 2.9 リアルタイム可観測性

すべてのステップが、観測・推論・行動・メトリクスを WebSocket 経由でストリーミングし、`execution_id` でルーティングされるため、同時実行される run のストリームが交差することはありません。あらゆるソースからのエラー — ノードの例外、server mode のサブプロセスクラッシュ、HTTP の失敗 — は統一された `ErrorBus` を通り、Report タブのエントリ + トーストとして表面化します (ADR-observability-004)。(React のレンダリングエラーはクライアント側のエラーバウンダリで捕捉されます。)

---

## 3. Sim-to-Real への道筋

AgentCanvas はポータビリティを念頭に設計されています。単一のエージェントグラフが、今日はシミュレータに対して実行され、将来はグラフレベルの変更なしに実ロボットへ移行できます。この性質は 2 つのアーキテクチャ上の決定から導かれます — 環境それ自体が nodeset であること (ADR-components-002)、そしてあらゆる nodeset が *server mode* を通じて分離されたランタイムで実行できること (ADR-server-001)。

### 現在: シミュレータ Nodeset

出荷されている環境 — Habitat (VLN-CE)、MatterSim / MP3D、HM-EQA、OpenEQA、SIMPLER (real-to-sim VLA)、LIBERO (マニピュレーション) — はそれぞれ、観測ポートと行動ポートを公開する `BaseNodeSet` として実装されています。エージェントグラフはこれらのポートに接続し、シミュレータを直接 import することは決してありません。これにより、グラフは特定の環境実装から独立に保たれます。

### 将来: 同じインターフェースを持つ ROS Nodeset

実ロボットへのデプロイは、シミュレータの nodeset を、同じ `observation` / `act` インターフェースを公開する **ROS nodeset** に置き換えることで実現されます。内部的には、この nodeset は既存の ROS コンポーネント — `cv_bridge`、`Nav2`、`MoveIt`、そしてハードウェアドライバのパッケージ — を統一されたファサードへと組み合わせます。server mode はこの nodeset を独自の ROS Python 環境内で起動し、HTTP 経由でキャンバスへブリッジします。エージェントグラフそれ自体は変更されません。

この分業が好都合なのは、実質的なエンジニアリング — 知覚、制御、動作プランニング、ハードウェアインターフェース — がすでに成熟した ROS パッケージとして存在しているからです。したがって ROS 側のアダプタは、ゼロからの開発ではなく組み合わせの作業となり、AgentCanvas 側の env nodeset は薄い HTTP クライアントへと縮小します。

### 双方向統合

AgentCanvas と ROS の境界は対称的であり、どちらの側も制御ループを所有できます。

- **AgentCanvas のサブシステムとしての ROS** *(ネイティブなパターン。server mode はこのケースのために設計されています)* — ROS nodeset が server mode で動作し、AgentCanvas がエージェントループを駆動し、ROS がセンシングとアクチュエーションを提供します。
- **ROS のサブシステムとしての AgentCanvas** *(これもサポート済み。フレームワークの変更は不要)* — より広いプロジェクトが ROS 主導である場合、ROS 側の制御ループが各ステップで AgentCanvas の `/run` エンドポイントを呼び出し (グラフをポリシーとして扱う)、返された行動をパブリッシュします。これには ROS 側に薄い ROS ブリッジノードが 1 つ必要なだけです。

### グラウンドトゥルース依存性の可視化

同じ nodeset の抽象化が、§1 で挙げた 2 つの課題に直接対処します。シミュレータのグラウンドトゥルースを問い合わせるノード (例: `env_habitat__get_object_pose`) と、実際の知覚を行うノード (例: SAM ベースの検出器) は、キャンバス上で見た目にも明確に異なるブロックとして現れます。したがって、エージェントがグラウンドトゥルースに依存するのか知覚に依存するのかは、隠れた実装の詳細ではなく、グラフのトポロジーの性質となります。一方を他方に差し替えることは、コードのリファクタリングではなく、局所的なエッジの変更です。

### ステータス

現在出荷されている環境 nodeset はすべてシミュレータベースです。実ロボット向けの **ROS nodeset は依然として [コントリビューション募集中](#8-コントリビューション) の枠** です — アーキテクチャ上の道筋は確立されており意図的なものであり、必要な ROS 側のコンポーネントはすでにエコシステムで利用可能です。

---

## 4. はじめに

AgentCanvas の使い方は 2 通りあり、どちらも同じ型付きグラフの基盤の上にあります。

1. **グラフを手で構築して実行する** — キャンバス上でノードを組み立て、シミュレータに対してエージェントをライブで実行し、大規模に評価します (本セクションの残りの部分)。
2. **Agent Architecture Search (AAS)** — シードグラフをコーディングエージェントに渡し、アーキテクチャを代わりに探索させます ([ジャンプ](#44-agent-architecture-search-aas-の実行))。

### 4.1 前提条件

- Conda を備えた Python 3.10+ (デフォルトの `agentcanvas` 環境 — ADR-platform-004)
- Node.js 18+
- *(オプション、Habitat-Sim 向け)* 別個の Python 3.8 環境 — `habitat-sim 0.1.7` はここでのみ動作します。AgentCanvas は server mode 経由でこれと通信します。[INSTALL.md](INSTALL.md) を参照してください

### 4.2 Web ダッシュボードの起動

```bash
# 環境をアクティベート
conda activate agentcanvas

# バックエンド (FastAPI :8000) + フロントエンド (Vite :5173) を起動
cd agentcanvas && bash run_dev.sh
```

キャンバスエディタにアクセスするには [http://localhost:5173](http://localhost:5173) を開きます。

### 4.3 評価の実行

同じ評価パイプラインが 4 つのインターフェースを通じて公開されています — 手元にあるものに応じて選んでください。

| # | インターフェース | 対象 | 最適な用途 |
|---|-----------|----------|----------|
| 1 | **フロントエンドの Eval ページ** | 人間                | クリック操作で、UI 上でライブの進捗を見守る |
| 2 | **`/experiment:run` スラッシュコマンド** | コーディングエージェント (Claude Code) | プロファイルでゲートされた GPU 受け入れ、自動割り当てポート、`:8000` を踏まない |
| 3 | **MCP サーバ** | コーディングエージェント              | 会話的でアドホックな評価 — スラッシュコマンドのオーバーヘッドなし |
| 4 | **HTTP API** | スクリプト / CI                | 直接 REST、MCP 不要 |

#### 1. フロントエンドの Eval ページ — 人間向け

保存したグラフを **Eval** ページで開き、split + エピソード範囲を選び、**Start** を押します。進捗は WebSocket 経由でライブにストリーミングされ、結果は `outputs/eval_runs/{run_id}/episodes/ep{NNNN}/` 配下にエピソードごとの JSONL として格納され (ADR-eval-004)、Run Detail パネルで閲覧できます。マルチワーカーの env ファンアウトとバッチ推論はフォームから設定できます (ADR-eval-002)。

→ [Batch Eval チュートリアル](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/tutorials/batch-eval.html)

#### 2. `/experiment:run` — このリポジトリのコーディングエージェント向け

Claude Code を使う場合、`/experiment:run <profile> -- <cmd>` は、あらゆる評価の呼び出しをバックエンドの `JobScheduler` の受け入れゲートでラップします (ADR-eval-003)。このラッパーは `.claude/commands/experiment/profiles.yaml` で宣言されたプロファイルの下で VRAM を確保し、割り当てられたポートでバックエンドを起動し (`BACKEND_URL=http://127.0.0.1:<port>` がラップされたコマンドにエクスポートされます)、終了時にスロットを解放します。関連コマンド: run のスナップショットには `/experiment:status`、グレースフルなキャンセルには `/experiment:teardown`。

→ [`.claude/commands/experiment/README.md`](../.claude/commands/experiment/README.md)

シードグラフに対する完全なアーキテクチャ探索の設計ループ (propose → evaluate → keep-the-best を何度も反復する) については、以下の [Run Agent Architecture Search](#44-agent-architecture-search-aas-の実行) を参照してください。

#### 3. MCP サーバ — コーディングエージェント向け

`agentcanvas-backend` を任意の MCP 対応クライアント (Claude Code、Cursor、…) に登録し、型付きツール (`graph_list`、`eval_start`、`eval_status`、`eval_export`、`eval_stop`) を会話的に呼び出します。iter-tree の管理は不要 — 借用または起動されたバックエンドに対する素の評価のみです。

→ [`agentcanvas/mcp_server/README.md`](../agentcanvas/mcp_server/README.md)

#### 4. HTTP API — スクリプトと CI 向け

スクリプト、CI、または非 MCP 環境のための直接 REST:

```bash
curl -X POST http://localhost:8000/api/eval/v2/start \
  -H 'content-type: application/json' \
  -d '{"graph_name": "navgpt_ce", "split": "val_unseen", "worker_count": 4}'
# ポーリング  GET /api/eval/v2/status
# 取得        GET /api/eval/v2/export/{run_id}
```

→ [Driving the Backend from a Coding Agent](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/tutorials/coding-agent-backend.html) — すべてのプログラム的モードを並べて深掘り

### 4.4 Agent Architecture Search (AAS) の実行

固定されたグラフを評価するだけでなく、AgentCanvas は **Agent Architecture Search** の基盤でもあります — これは開発時のループであり、LLM コーディングエージェントの *Optimizer* がシードの *Executor* に対するグラフ編集を繰り返し提案し、各候補をシミュレータで評価し、改善を保持します (§1 — [なぜ探索可能な基盤なのか](#1-なぜ-agentcanvas-なのか))。エージェントは型付きグラフなので、各候補は高価なロールアウトの前に実行される型チェック済みのパッチであり、ノードごとのエピソードログによって Optimizer はスコアの変化を特定のモジュールに帰属させられます。

<p align="center">
  <img src="../assets/readme/aas-search.gif" alt="身体性 executor のグラフ上を探索するコーディングエージェント optimizer — 編集を提案し、実行し、得られたゲインを保持する" width="800">
  <br><sub><em>身体性 executor のグラフ上を探索するコーディングエージェント optimizer — 編集を提案し、実行し、ゲインを保持する。</em></sub>
</p>

探索は **method-seeded (メソッドを種とする)** です。`iter_0` は公開された身体性メソッドであり、ループはその周辺のグラフレベルの編集を探索します。3 つの探索バリアントが `.claude/commands/architect/` 配下の Claude Code スキルとして出荷されており、1 つのコーディングエージェントハーネス (proposer → implementer → evaluator) を共有し、proposer のロジック + 永続メモリのみが異なります。

| バリアントスキル | 論文名 | 探索ポリシー |
|---|---|---|
| `myloop` | **KDLoop** | 4 フェーズの THINK → CRITIC → EXPERIMENT → DISTILL サイクル、型付きメモリ + REFLECT メタフェーズ |
| `adas-subagent` | **ADAS** (移植) | フラットな追記専用アーカイブ上での Reflexion スタイルの提案 |
| `aflow` | **AFlow** (移植) | スコア softmax による親選択 + リプレイ防止メモリ |

```text
# このリポジトリの Claude Code セッション内で — MapGPT executor 上で KDLoop を実行
/architect:myloop:loop mapgpt_mp3d --goal "raise val_unseen SR"

# ADAS / AFlow の移植版も同じ  <graph> [<version>]  形式を取る
/architect:adas-subagent:loop smartway_ce
/architect:aflow:loop explore_eqa_hmeqa
```

現在探索向けに配線されているシードグラフ: `mapgpt_mp3d`、`smartway_ce` (VLN)、`explore_eqa_hmeqa` (EQA)、`voxposer_libero_monolithic` (VLA)。各反復は、その提案、パッチ、評価スコア、ログを `outputs/design_runs/{variant}/{graph}/vN/iter_M/` 配下に書き込みます。

→ [AAS パイプラインリファレンス](https://jianzhou0420.github.io/AgentCanvas/pages/aas/index.html)

### 4.5 ドキュメント

```bash
# ドキュメントサイトをローカルの :8092 で配信 (SSE 経由のライブリロード)
bash docs/run_dev.sh
```

---

## 5. アーキテクチャ

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

**主要な設計**: フレームワークは **ドメイン知識をまったく持ちません** (ADR-platform-001)。すべてのドメイン固有コード — VLN ポリシー、LLM プロンプト、ナビゲーションツール、環境ラッパー — は `workspace/` に存在します。フレームワークは基底クラスの継承を通じて実行時にコンポーネントを検出します。ドメインコードを直接 import することは決してありません。import 境界は `agentcanvas/backend/app/test_import_boundary.py` によって強制されます。

---

## 6. プロジェクト構成

```
vlnworkspace/                  # リポジトリのルート (旧称。プラットフォーム名は "AgentCanvas")
├── agentcanvas/               # フルスタックの Web アプリケーション
│   ├── backend/app/         #   FastAPI バックエンド (実行エンジン、API、サービス、エラー)
│   ├── frontend/src/        #   React + TypeScript (キャンバスエディタ)
│   └── mcp_server/          #   コーディングエージェント統合のための MCP サーバ
├── workspace/                 # ユーザーワークスペース — すべてのドメインコンポーネント (自動検出)
│   ├── nodesets/            #   パレット別の nodeset: env / method / model / policy (+ common、_upstream)
│   ├── graphs/              #   保存されたエージェントグラフ (kind="graph")
│   ├── graph_nodes/         #   再利用可能な複合ノード (kind="node")
│   ├── nodes/               #   スタンドアロンの BaseCanvasNode サブクラス
│   ├── architect/           #   AAS 探索プロファイル + run のスキャフォールディング
│   └── hooks.json           #   ワークスペースレベルのフック定義
├── data/                      # データセット、モデルの重み (gitignore 対象)
├── outputs/                   # 評価 + design-run の出力 (eval_runs/、design_runs/、…)
├── docs/                      # 手書きの HTML ドキュメントサイト (run_dev.sh → :8092)
├── third_party/               # Git サブモジュール (habitat-lab、VLN-CE、MatterSim、vla_workspace、…)
└── scripts/                   # データセットアップ + インストールスクリプト
```

---

## 7. プロジェクトステータス

AgentCanvas は **1.0 以前で、活発に開発中** です。ステータスは、進行中の機能チェックリストではなくバージョンによって追跡されます — 詳細は [Versioning Policy](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html) と [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html) を参照してください。

- **v0.1 — AAS 実験。** 論文の Agent Architecture Search の run が実行されたスナップショット — それらの結果の再現性アンカーであり、公開リリースではありません。
- **v0.2 — リサーチプレビュー (現行)。** 最初のオープンソースリリース: キャンバスエディタ、グラフ executor (DAG + cyclic + multi-scope)、ステートコンテナ、自動ホストされる server-mode nodeset、バッチ評価、40 以上の nodeset (env / method / model / policy) がすべて本番稼働します。公開 API はまだ凍結されていないため、マイナーリリースで破壊的変更が入る可能性があります。出荷済みのインベントリ: [§2 機能](#2-機能) および [VLN](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/vln-support-status.html) / [EQA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/eqa-support-status.html) / [VLA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/design-docs/vla-support-status.html) サポートステータスページ。
- **v1.0 — 進行中。** 公開 API が安定したとき — オープンソース化され SemVer の下で凍結され、論文とは独立 — にリリースされます。
- **v2.0 — 将来。** トポロジーを変化させる実行: 上限のないサブエージェント生成、ランタイムリスト上でのランタイムファンアウト、実行時に新しいツールタイプが出現すること、自己改変するグラフ。テーゼと未解決の問いについては [`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html) §2 を参照してください。

---

## 8. コントリビューション

2 種類のコントリビューションがあり、どちらも歓迎します — [CONTRIBUTING.md](../CONTRIBUTING.md) を参照してください。

- **Content — nodeset とグラフ。** ツール / シミュレータ / モデルをラップする nodeset (例: リアルタイム 3D Gaussian Splatting、ボクセルベースの SLAM システム) を書く、メソッド (例: NavGPT、MapGPT) をコード化する、あるいは既存の nodeset を配線して完全なエージェントにするグラフを組み立てます。`workspace/` への PR を開いてください。レビューは軽めです。
- **Core — UI、バックエンド、フレームワーク。** バグ修正、新機能、さらにはリファクタリングも歓迎します。1 つだけお願いがあります。変更が実際の時間を要するほど大きい場合は、まず [Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions) を開いて、構築前に方向性を合わせましょう。

すべての nodeset とグラフは、以下のクレジットボードでその作者 / メンテナにクレジットされます — 関連論文があれば引用リンク付きで — ので、ここに貢献してもあなたのオーサーシップが損なわれることはありません。

### クレジット

<table>
<tr><th>コンポーネント</th><th>作成者</th></tr>
<tr>
<td><b>AgentCanvas フレームワーク</b></td>
<td><a href="https://github.com/jianzhou0420">@jianzhou0420</a></td>
</tr>
<tr>
<td>

<details open>
<summary><b>初回リリース</b> — 同梱の nodeset、リファレンスグラフ、ドキュメントサイト</summary>

<br>

<b>シミュレータ / 環境</b>

- Habitat (VLN-CE 連続ナビゲーション)
- Matterport3D / MatterSim (離散パノラマナビゲーション)
- HM-EQA (身体性 QA 環境)
- OpenEQA (身体性 QA ベンチマーク、EM-EQA モード)
- SIMPLER (SAPIEN / ManiSkill2 による real-to-sim VLA 評価)
- LIBERO (MuJoCo / robosuite マニピュレーション、5 スイート)

<b>エージェントメソッド / 推論</b>

<i>EQA</i>

- OpenEQA EM-EQA ベースライン — blind-LLM / single-frame / multi-frame (`openeqa_em_*.json`) ✅ すべて検証済み。multi-frame の LLM-Match 0.7025 対 論文の 0.466 (gpt-4o の reasoner+judge が論文の gpt-4 / gpt-4-vision-preview を上回る)
- Explore-EQA (HM-EQA 上での Prismatic 固定のフロンティア探索) ✅ 検証済み — SR 0.42 はベースラインの 0.44 を再現
- ToolEQA (HM-EQA のみ — PortBench v1 の基盤) — 2026-06-08 にモノリス優先で再構築。エンドツーエンドで動作 (ReAct + 融合 TSDF の go_next + server mode HTTP 経由の Qwen2.5-VL/DetAny3D)、SR チューニングは進行中

<i>VLN</i>

- NavGPT (LLM の思考–行動推論プリミティブ) ✅ gpt-4 で動作 (高コスト)。他の LLM は未検証 (gpt-4o は長い ReAct プロンプトで劣化することが知られている)
- MapGPT (言語的トポマップ LLM エージェント、ACL 2024) ✅ 検証済み — MapGPT_72 で SR 0.477 / 0.463
- SmartWay-mono (VLN-CE のウェイポイント予測器) ✅ 論文と同等 — SR 0.270 対 論文 0.29
- SmartWay-CE ✅ サイレント完了の競合を修正。20 ワーカー評価でエンドツーエンドに動作
- SpatialNav (空間グラフナビゲーション) ❌ 未検証 — SR=0
- Open-Nav (オープンボキャブラリナビゲーション) ❌ 未検証 — SR=0
- DiscussNav (マルチ LLM ディベート、有界ファンアウト) ❓ 進行中 — fitness はまだ論文と同等に至っていない
- Three-Step Nav (ゼロショットのウェイポイントナビ、Open-Nav のサブクラス) ❓ エンドツーエンドで検証済み — SR 0.10 / oracle 0.30 @10ep。論文と同等にするチューニングは保留中
- AO-Planner (SAM + LLM + 3D 経路プランナー、AAAI 2025) ❓ 進行中 — nodeset は出荷済み、評価は保留中
- Basic Agent (VLN の基礎ツールキット — 5 カテゴリにまたがる 11 ノード)

<i>VLA</i>

- VLA 固有のメソッド (Pi0 / SmolVLA / DP / DROID-DP / Octo / VoxPoser-LIBERO) は、下記の <b>ポリシー</b> 配下にあります — これらは推論型ではなくポリシー型 (env 観測 → 行動) であるため、タスクファミリーではなくコード構造でグループ化されています

<b>知覚 / ビジョン</b>

- SAM (Segment Anything)
- BLIP-2 + Faster R-CNN (キャプション生成 & 検出)
- RAM (recognize-anything model)
- SpatialBot (深度を考慮した VLM)
- Prismatic VLM (トークン尤度スコアリング + 自由形式の生成)
- TSDF マッピング
- セマンティックシーングラフ

<b>ポリシー</b>

- CMA (Cross-Modal Attention の VLN-CE ベースライン) ✅ 検証済み — `straightforward.json` を verified/ に昇格、SR 0.38 / SPL 0.348、ネイティブとビット単位で同一
- Octo (VLA ジェネラリスト、ネイティブの SIMPLER ベースライン) ✅ ベースラインは `octo_simpler.json` で動作
- 汎用 VLA フレームワーク (Pi0 / SmolVLA / DP / DROID-DP アダプタ) ✅ Pi0 検証済み — `vla_policy_libero` の libero_spatial task 0 で 5/5。SIMPLER バリアントは未定
- VoxPoser-LIBERO (LMP + ボクセルコストマップ + OSC) ✅ エンドツーエンドで検証済み (把持 + 搬送)、SR を記録
- VLN-CE ポリシーアダプタ (12 バリアントの R2R-CE レジストリ — 2 つは upstream リリース済み、10 のアブレーションはプレースホルダー表記)

<b>ドキュメントサイト</b> — 手書きの HTML (2026-05-18 の MkDocs 廃止後)。46 個の ADR、用語集、capability ページ、チュートリアル、設計ドキュメントを含む

</details>

</td>
<td><a href="https://github.com/jianzhou0420">@jianzhou0420</a></td>
</tr>
<tr>
<td><b>ベンチマーク:</b> AI2-THOR <i>(ALFRED / TEACh — E4)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>ベンチマーク:</b> RxR-CE <i>(多言語 VLN-CE — E2)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>ベンチマーク:</b> REVERIE <i>(リモートオブジェクトグラウンディング — E3)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>ベンチマーク:</b> OpenEQA A-EQA <i>(アクティブ EQA モード — E10)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>メソッド:</b> HAMT <i>(階層的ヒストリートランスフォーマー — M5)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>メソッド:</b> DUET <i>(デュアルスケールグラフトランスフォーマー — M6)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>メソッド:</b> MapGPT (メトリックグリッド版) <i>(LLM + 深度由来の occupancy — M2。出荷済みの言語的トポ版とは別物)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>メソッド:</b> InstructNav <i>(Dynamic CoN + Multi-Sourced Value Maps、CoRL 2024 — M8)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>メソッド:</b> VLN-SIG <i>(サブ命令グラウンディング — M4)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>機能:</b> Memory nodeset <i>(エピソード記憶の想起 + セマンティック検索 — F1)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>機能:</b> 並列ノード実行 <i>(Pregel スーパーステップモデル — F3)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>機能:</b> グラフをスタンドアロン Python としてエクスポート <i>(ヘッドレスバッチ評価 — F4)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>インフラ:</b> Docker server mode <i>(Habitat / MP3D コンテナ — F7)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
<tr>
<td><b>インフラ:</b> ROS nodeset <i>(server mode 経由の実ロボットデプロイ — §3)</i></td>
<td><i><a href="../CONTRIBUTING.md">コントリビューション募集</a></i></td>
</tr>
</table>


---

## Star History

<a href="https://star-history.com/#jianzhou0420/AgentCanvas&Date">
  <img src="https://api.star-history.com/svg?repos=jianzhou0420/AgentCanvas&type=Date" alt="Star History チャート" width="600">
</a>

---

## 9. 引用

研究で AgentCanvas を使用する場合は、以下を引用してください：

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

## 10. ライセンス

Apache License 2.0 — [LICENSE](../LICENSE) を参照してください。
