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
  <a href="https://jianzhou0420.github.io/AgentCanvas/"><img src="https://img.shields.io/badge/Docs-2ea44f?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation"></a>
  <a href="#6-引用"><img src="https://img.shields.io/badge/BibTeX-Cite-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white" alt="BibTeX"></a>
</p>

<img src="../assets/readme/editor-hero.gif" alt="AgentCanvas エディタ: MapGPT executor がノードとワイヤのグラフとして読み込まれ、続いてライブの R2R エピソードがエンドツーエンドで実行される様子" width="760">

<sub><em>エディタ上でライブ収録 — MapGPT executor が読み込まれ、続いて実際の R2R エピソードがエンドツーエンドで実行されます。</em></sub>

</div>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Status: Research Preview](https://img.shields.io/badge/Status-Research_Preview-orange.svg)](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html)
[![GitHub stars](https://img.shields.io/github/stars/jianzhou0420/AgentCanvas?style=social)](https://github.com/jianzhou0420/AgentCanvas/stargazers)

**身体性 AI 研究のためのビジュアルなエージェント設計プラットフォーム。** 1 つの型付きグラフに 2 つの役割を持たせます。すなわち、身体性エージェントを実行する *ハーネス (harness)* と、コーディングエージェントが編集・検証する *スキャフォールド (scaffold)* です。

AgentCanvas を使うと、研究者は VLN・EQA・VLA やその周辺タスク向けの身体性エージェントを、ノードグラフを描くことでプロトタイピングできます。これらのグラフは、シミュレータ (Habitat-Sim、MatterSim、SAPIEN/ManiSkill2、MuJoCo/robosuite)、あるいは原理的には実世界のセットアップに対してリアルタイムに実行されます。*1 つの JSON = 1 つのエージェント = 1 つのグラフ*。エージェントの振る舞いは命令型コードではなくデータフローグラフであり、グラフこそが信頼できる唯一の情報源 (source of truth) として、単一の JSON ファイルに保存され、完全なエージェントとして読み込まれます。

**対象ユーザー**: 実行スタックを毎回書き直すことなく、身体性エージェントのアーキテクチャを組み立て、比較し、共有したい研究者。本プラットフォームは VLN (Vision-and-Language Navigation)、EQA (Embodied Question Answering)、VLA (Vision-Language-Action) のポリシーベンチマークをカバーし、nodeset モデルを通じてその他の身体性 / エージェント的な設定にも適応します。

> **ステータス**: リサーチプレビュー、1.0 以前 — 入れ替え可能な 4 つのパレット (**env** · **method** · **model** · **policy**) にまたがる 40 以上の nodeset。公開 API はまだ凍結されていません ([Versioning Policy](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/repo/versioning.html))。

> **コントリビューション**: nodeset、グラフ、コアへの PR、すべて歓迎します — あらゆる貢献は [クレジット](#クレジット) ボードにクレジットされます。[CONTRIBUTING.md](../CONTRIBUTING.md) を参照してください。

---

## 新着情報

- [2026/07] 🚀 **Graph SDK — Python でエージェントを構築・実行** — 同じキャンバスグラフが、import 可能なライブラリになりました: `from agentcanvas import Graph` でノードの追加 / 接続を行い、プロセス内で実行やバッチ評価をしたり、グラフをスタンドアロンのビルダースクリプトへコンパイルし直したりできます。同じ `GraphDefinition` を用い、キャンバス + JSON と完全に相互変換可能です。[Graph SDK ドキュメント](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/capabilities/graph-sdk.html) を参照してください。
- [2026/07] 🎥 **pySLAM クラシック SLAM デモ** — TUM RGB-D 上で pySLAM が主役に: ストリーミング再生の env がベンチマークシーケンスをフレームごとにライブの SLAM セッションへ送り込みます — 推定されたカメラ軌跡が俯瞰視点でグラウンドトゥルースにフィットし、疎な 3-D マップがリアルタイムに密になっていきます。シミュレータもポリシーも使わず、CPU のみ。フルクリップ + ウォークスルーは [pySLAM nodeset ドキュメント](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-pyslam.html) にあります。

  [![TUM RGB-D 上での pySLAM ストリーミング SLAM — ライブのカメラ軌跡とグラウンドトゥルースの対比、リアルタイムに密になる 3-D マップ、そして完成したマップのオービット](../docs/assets/videos/pyslam-tum-slam-demo.gif)](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-pyslam.html)
- [2026/07] 🔥 **基盤モデルサポートの拡大** — 29 個の基盤モデルが薄い server-mode シェル (transformers-native + その他のソース) として配線され、手作りのグラフと AAS optimizer の双方から利用できるようになりました: 最近の VLM (Qwen3-VL、InternVL3、Gemma 3、SmolVLM2)、オープンボキャブラリ知覚 (SigLIP2、OWLv2、Grounding DINO)、そしてジオメトリ / 深度のバックボーン。[基盤モデルのカバレッジ](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/index.html) とモデルごとの [Credits](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html) を参照してください。
- [2026/07] 🔥 **キャンバスからノードのソースを編集** — 新しい Source タブは、選択したノードに対応する nodeset ソースのスコープ付きスライス (グローバル、参照される関数、クラスそのもの) を表示し、構文チェック付きのホットリロードで編集を差し戻します。PR: [#5](https://github.com/jianzhou0420/AgentCanvas/pull/5)。
- [2026/07] 🎉 **初の公開リリース** — AgentCanvas がリサーチプレビュー (1.0 以前) としてオープンソース化されました。ドキュメント: [jianzhou0420.github.io/AgentCanvas](https://jianzhou0420.github.io/AgentCanvas/)。

---

## 目次

1. [なぜ AgentCanvas なのか?](#1-なぜ-agentcanvas-なのか) — 身体性エージェントのための探索可能な基盤 (searchable substrate) と、それが解決する課題
2. [機能](#2-機能) — *1 つの JSON = 1 つのエージェント* (§2.2) / *1 つの Python クラス = 1 つのノード* (§2.6) の原則、加えてキャンバスエディタ、グラフ executor、分離されたランタイム環境、ネストグラフ、ステートコンテナ、フック
3. [Sim-to-Real への道筋](#3-sim-to-real-への道筋) — 同じエージェントグラフを今日はシミュレータで、明日は実ロボットで — env-as-nodeset + server mode + ROS を通じて
4. [はじめに](#4-はじめに) — 前提条件、Web ダッシュボードの起動、評価の実行、アーキテクチャ探索の実行、ドキュメントの配信
5. [コントリビューション](#5-コントリビューション) — 最も助けが必要な領域 · クレジット
6. [引用](#6-引用) — AgentCanvas 論文を引用する
7. [ライセンス](#7-ライセンス) — Apache 2.0

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

<details>
<summary><b>9 つの機能</b> — キャンバスエディタ · グラフエンジン · 分離されたランタイム · ネストグラフ · ステートコンテナ · Python で定義されるノード · フック · バッチ評価 · 可観測性</summary>

<br>

### 2.1 ビジュアルキャンバスエディタ

すべてのノードタイプが共存する ComfyUI スタイルのフラットなワークスペース — 環境、LLM、推論チェーン、制御ゲート、出力ビューア。サイドバーからノードをドラッグし、配線し、Play を押します。

### 2.2 グラフ実行エンジン

**1 つの JSON = 1 つのエージェント。** エージェントの振る舞いのすべて — ノード、配線、設定、ステートコンテナ、フック — は単一の JSON ファイルです。読み込み、実行し、共有し、差分を取れます。隠れたパイプラインコードはありません。キャンバスで見えるものがそのまま実行されます。

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

エンジンはそのグラフを実行します。ノードは固定の順序ではなく、入力が到着したときに発火します。同じエンジンが、AgentCanvas v1 がサポートするすべてのグラフ形状を扱います — v1 の有界静的トポロジー (bounded-static-topology) パラダイムがカバーするエージェント形態の完全な一覧については、[`major-versions.html`](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/core/major-versions.html) を参照してください。

- **DAG ワークフロー** — 非巡回パイプラインのための単一フォワードパス
- **巡回エージェントループ** — **2 ピボット (two-pivot)** モデルによる observe-think-act-repeat: 両側を持つ **`IterIn`** (左側に run 開始時の init 入力、右側に反復ごとの loop-carry) と **`IterOut`** を用い、グラフを見た目上は非巡回に保ちながら実行時の巡回を可能にします (ADR-dataflow-008。これは ADR-dataflow-006 の従来の 3 ピボット `initialize`/IterIn/IterOut を 2 つに畳み込んだものです)
- **マルチスコープ反復** — 1 つのフラットグラフ内に共存する N 個の `(IterIn, IterOut)` ペア (ADR-dataflow-007 / ADR-executor-003)
- **ReAct ループ** — `LLMCallNode` のサブクラス内に隠すか、ルーター + 事前宣言された N 個のツール分岐として明示的に表現します
- **有界マルチエージェント** — 固定 N または `K_max` で上限を設けたファンアウト (例: DiscussNav 風のディベート、AutoGen 風の固定ロール)
- **Plan-and-Execute** — 有界なツールプール上で、ルーターによりディスパッチされる

エンジンはグラフノードに触れることなく拡張することもできます。シェルフックが各ノード実行の前後やグラフのライフサイクル境界で発火し — 出力のログ記録、入力の検証、ノードのブロック、データの変更を行い — 保存されたグラフとともに移動します。

### 2.3 分離されたランタイム環境

研究ツールはしばしば、互いに衝突する Python 環境を必要とします (Habitat は Python 3.8 を、SLAM は ROS を必要とします)。あらゆる `BaseNodeSet` は **server mode** で実行できます — フレームワークが nodeset のポート定義から HTTP サーバを自動生成し、それぞれ独自のインタプリタで動かします。追加コードはゼロです。

```
# Same nodeset code, two deployment modes:
POST /api/components/nodesets/env_habitat/load              # in-process
POST /api/components/nodesets/env_habitat/load?mode=server  # separate process
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

### 2.7 バッチ評価とジョブキュー

キャンバス上で動作するのと同じグラフを、数百のエピソードにわたって採点する評価ジョブとして投入できます。バックエンドが所有する `JobScheduler` が、全セッションで共有される VRAM 予算に対して受け入れを制御します (ADR-eval-003)。受け入れられた各 run はそれぞれ独自のサブプロセスであり、そのライフタイムはバックエンドに紐づけられます (`PR_SET_PDEATHSIG`) — 孤児となる GPU プロセスは発生せず、完了したすべてのエピソードはディスクに永続化されます。エピソードごとのログは自己完結したレイアウトに格納されるため (ADR-eval-004)、チームメイトは再実行なしに任意の単一エピソードを再生できます。

### 2.8 実行ログとライブビュー

すべてのステップが、観測・推論・行動・メトリクスを WebSocket 経由でストリーミングし、`execution_id` でルーティングされるため、同時実行される run のストリームが交差することはありません。あらゆるソースからのエラー — ノードの例外、server mode のサブプロセスクラッシュ、HTTP の失敗 — は統一された `ErrorBus` を通り、Report タブのエントリ + トーストとして表面化します (ADR-observability-004)。(React のレンダリングエラーはクライアント側のエラーバウンダリで捕捉されます。)

</details>

---

## 3. Sim-to-Real への道筋

AgentCanvas はポータビリティを念頭に設計されています。単一のエージェントグラフが、今日はシミュレータに対して実行され、将来はグラフレベルの変更なしに実ロボットへ移行できます。この性質は 2 つのアーキテクチャ上の決定から導かれます — 環境それ自体が nodeset であること (ADR-components-002)、そしてあらゆる nodeset が *server mode* を通じて分離されたランタイムで実行できること (ADR-server-001)。

<details>
<summary><b>完全な道筋</b> — 現在のシミュレータ · 同じインターフェースを持つ ROS nodeset · 双方向統合 · グラウンドトゥルースの可視化</summary>

<br>

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

現在出荷されている環境 nodeset はすべてシミュレータベースです。実ロボット向けの **ROS nodeset は依然として [コントリビューション募集中](#5-コントリビューション) の枠** です — アーキテクチャ上の道筋は確立されており意図的なものであり、必要な ROS 側のコンポーネントはすでにエコシステムで利用可能です。

</details>

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
# Activate environment
conda activate agentcanvas

# Start backend (FastAPI :8000) + frontend (Vite :5173)
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
# poll  GET /api/eval/v2/status
# fetch GET /api/eval/v2/export/{run_id}
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
# In a Claude Code session on this repo — run KDLoop over the MapGPT executor
/architect:myloop:loop mapgpt_mp3d --goal "raise val_unseen SR"

# The ADAS / AFlow ports take the same  <graph> [<version>]  form
/architect:adas-subagent:loop smartway_ce
/architect:aflow:loop explore_eqa_hmeqa
```

現在探索向けに配線されているシードグラフ: `mapgpt_mp3d`、`smartway_ce` (VLN)、`explore_eqa_hmeqa` (EQA)、`voxposer_libero_monolithic` (VLA)。各反復は、その提案、パッチ、評価スコア、ログを `outputs/design_runs/{variant}/{graph}/vN/iter_M/` 配下に書き込みます。

→ [AAS パイプラインリファレンス](https://jianzhou0420.github.io/AgentCanvas/pages/aas/index.html)

### 4.5 ドキュメント

```bash
# Serve the doc-site locally on :8092 (live-reload via SSE)
bash docs/run_dev.sh
```

---

## 5. コントリビューション

2 種類のコントリビューションがあり、どちらも歓迎します — [CONTRIBUTING.md](../CONTRIBUTING.md) を参照してください。

- **Content — nodeset とグラフ。** ツール / シミュレータ / モデルをラップする nodeset (例: リアルタイム 3D Gaussian Splatting、ボクセルベースの SLAM システム) を書く、メソッド (例: NavGPT、MapGPT) をコード化する、あるいは既存の nodeset を配線して完全なエージェントにするグラフを組み立てます。`workspace/` への PR を開いてください。レビューは軽めです。
- **Core — UI、バックエンド、フレームワーク。** バグ修正、新機能、さらにはリファクタリングも歓迎します。1 つだけお願いがあります。変更が実際の時間を要するほど大きい場合は、まず [Discussion](https://github.com/jianzhou0420/AgentCanvas/discussions) を開いて、構築前に方向性を合わせましょう。

すべての nodeset とグラフは、以下のボードでその作者 / メンテナにクレジットされます — 関連論文があれば引用リンク付きで — ので、ここに貢献してもあなたのオーサーシップが損なわれることはありません。**AgentCanvas フレームワーク**、および初回リリースの **メソッド、グラフ、環境統合** は **AC-Team** によるものです。以下の **基盤モデルとポリシー** は **サードパーティ** です — AgentCanvas は各モデルがグラフに差し込めるようにする薄い server-mode ラッパーのみを出荷します (人間のユーザーと AAS optimizer の双方のために)。各モデルのクレジットはそれぞれのオリジナルの作者に帰属します — 基盤モデルは、ソース別 (transformers-native か `torch.hub` / `torchvision` / vendored upstream repo か) に分けて **以下の別テーブル** にまとめられており、モデルごとの完全な帰属は Credits ページに記載されています。このボードは設計上、名前のみです。グラフごとの検証詳細を含む **正規のインベントリ** は、[ドキュメントサイトの Credits ページ](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html) および [VLN](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/vln-support-status.html) / [EQA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/eqa-support-status.html) / [VLA](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/status/vla-support-status.html) のサポートステータスページにあります。

### クレジット

✅ 検証済み — 論文 / リファレンス実装を再現 · 🚧 エンドツーエンドで動作、検証は進行中

<table>
  <thead align="center">
    <tr>
      <th>環境</th>
      <th>メソッド</th>
      <th>モデルとポリシー</th>
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
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/common/tools/basic-agent.html">Basic Agent ツールキット</a> ✅</li>
            </ul>
          </li>
          <li><b>EQA</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/env/openeqa.html">EM-EQA ベースライン</a> ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/explore-eqa.html">Explore-EQA</a> ✅</li>
              <li>ToolEQA 🚧</li>
            </ul>
          </li>
          <li><b>VLA (ゼロショット)</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/voxposer.html">VoxPoser-LIBERO</a> ✅</li>
            </ul>
          </li>
        </ul>
      </td>
      <td>
        <ul>
          <li><b>ポリシー</b>
            <ul>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-cma.html">CMA</a> ✅</li>
              <li>Octo (SIMPLER ベースライン) ✅</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-vla.html">VLA フレームワーク (Pi0 / SmolVLA / DP / DROID-DP)</a> 🚧</li>
              <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/method/policy-adapters.html">R2R-CE ポリシーレジストリ (12 バリアント)</a> 🚧</li>
            </ul>
          </li>
          <li><b>マッピング</b> <sub><i>(AgentCanvas 製)</i></sub>
            <ul>
              <li>TSDF マッピング ✅</li>
              <li>セマンティックシーングラフ ✅</li>
            </ul>
          </li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

**基盤モデル (Foundation models)** — **1 つの薄い nodeset シェル** (遅延ロード · single-flight GPU · base64-npy のデータフローエンベロープ) の背後にラップされたサードパーティのモデルであり、各モデルは **人間のユーザー** と **AAS optimizer** の双方にとって統一的な構成要素となります。*これらを私たちが作ったわけではありません — 私たちはシェルのみを出荷しており、クレジットはオリジナルの作者に帰属します* (モデルごとの完全な帰属 + 論文は [Credits ページ](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html) にあります)。ソース別に分けています。

<table>
  <thead align="center">
    <tr>
      <th>transformers-native <sub>(<code>AutoModel</code> / <code>pipeline</code> の薄いラッパー)</sub></th>
      <th>その他のソース <sub>(<code>torch.hub</code> / <code>torchvision</code> / vendored upstream repo)</sub></th>
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
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-segmentation.html">セグメンテーション (Mask2Former)</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-florence2.html">Florence-2</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-depth-anything.html">Depth Anything V2</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-depthpro.html">DepthPro</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-normal.html">サーフェスノーマル (Sapiens)</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-pointmap.html">ポイントマップ (Sapiens 3D)</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-matching.html">SuperPoint + LightGlue</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-blip2.html">BLIP-2</a> + Faster R-CNN</li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-instructblip.html">InstructBLIP</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-qwen2-5-vl.html">Qwen2.5-VL</a></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-qwen3-vl.html">Qwen3-VL</a> <sub>(画像 + 動画)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-internvl3.html">InternVL3</a> <sub>(画像 + 動画)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-gemma3.html">Gemma 3</a> <sub>(ゲート付き)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-smolvlm2.html">SmolVLM2</a> <sub>(画像 + 動画)</sub></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-dinov2.html">DINOv2 / DINOv3</a> <sub>(torch.hub + transformers hf)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-grounding-dino.html">Grounding DINO</a> <sub>(groundingdino-py + transformers hf_tiny)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-opticalflow.html">オプティカルフロー (RAFT)</a> <sub>(torchvision)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-vggt.html">VGGT</a> <sub>(upstream リポジトリ)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-cotracker.html">CoTracker</a> <sub>(upstream リポジトリ)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-detany3d.html">DetAny3D</a> <sub>(ベンダー同梱)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/model-ram.html">RAM / RAM++</a> <sub>(recognize-anything)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-spatialbot.html">SpatialBot</a> <sub>(Bunny のリモートコード)</sub></li>
          <li><a href="https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/nodesets/model/vlm-prismatic.html">Prismatic VLM</a> <sub>(upstream リポジトリ)</sub></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

**コントリビューション募集** — 予約された枠であり、実装した人にクレジットされます ([貢献方法](../CONTRIBUTING.md); ID は [Credits ページ](https://jianzhou0420.github.io/AgentCanvas/pages/developer-guide/community/credits.html) のロードマップ枠です):

<table>
  <thead align="center">
    <tr>
      <th>ベンチマーク</th>
      <th>メソッド</th>
      <th>機能とインフラ</th>
    </tr>
  </thead>
  <tbody valign="top">
    <tr>
      <td>
        <ul>
          <li>AI2-THOR — ALFRED / TEACh <i>(E4)</i></li>
          <li>RxR-CE — 多言語 VLN-CE <i>(E2)</i></li>
          <li>REVERIE — リモートオブジェクトグラウンディング <i>(E3)</i></li>
          <li>OpenEQA A-EQA — アクティブ EQA <i>(E10)</i></li>
        </ul>
      </td>
      <td>
        <ul>
          <li>HAMT — 階層的ヒストリートランスフォーマー <i>(M5)</i></li>
          <li>DUET — デュアルスケールグラフトランスフォーマー <i>(M6)</i></li>
          <li>InstructNav — 動的 CoN + バリューマップ <i>(M8)</i></li>
          <li>VLN-SIG — サブ命令グラウンディング <i>(M4)</i></li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Memory nodeset — エピソード記憶の想起 + セマンティック検索 <i>(F1)</i></li>
          <li>並列ノード実行 — Pregel スーパーステップ <i>(F3)</i></li>
          <li>グラフをスタンドアロン Python としてエクスポート <i>(F4)</i></li>
          <li>Docker server mode — Habitat / MP3D コンテナ <i>(F7)</i></li>
          <li>ROS nodeset — 実ロボットデプロイ (<a href="#3-sim-to-real-への道筋">§3</a>)</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>


---

## 6. 引用

研究で AgentCanvas を使用する場合は、以下を引用してください:

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

## 7. ライセンス

Apache License 2.0 — [LICENSE](../LICENSE) を参照してください。
