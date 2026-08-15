<div align="center">

# Embodied Agents Take Control

### Minimal-Interface Zero-Shot Agents Rival Industrial-Scale Policies in Vision-and-Language Navigation

**Jian Zhou\* · Xunyi Zhao\* · Gengze Zhou · Zerui Li · Sihao Lin · Jiajun Liu · Qi Wu**

<p>
  <a href="https://arxiv.org/abs/2607.26148"><img src="https://img.shields.io/badge/arXiv-2607.26148-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://jianzhou0420.github.io/src/works/MIP/index.html"><img src="https://img.shields.io/badge/Project%20Page-1f6feb?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Cite-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white" alt="BibTeX"></a>
</p>

<img src="https://jianzhou0420.github.io/src/works/MIP/fig1_teaser.png" alt="MIP teaser: a general-purpose coding agent drives continuous VLN through a two-tool interface" width="760">

</div>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

> **Archive branch.** This branch is the frozen code state behind the paper, kept as-is
> for reproducibility. The same code may or may not be carried on `main` going forward —
> when in doubt, this branch is the paper's reference.

**The Minimal-Interface Probe (MIP).** We hand an unmodified, general-purpose coding agent — Claude Code, Codex CLI, or a ~100-line minimal agent loop — a robot in continuous Vision-and-Language Navigation, through the smallest interface that still permits the task: `observe()` returns one egocentric RGB frame, `step()` executes discrete motion actions. No map, no memory module, no waypoint predictor, no panorama, no task-specific training. Run zero-shot on R2R-CE val-unseen, this bare probe rivals industrial-scale trained policies. This repository is the complete experiment code behind the paper: the probe, the harness adapters, the frozen experiment board, and the simulator toolfaces it runs against.

---

## 1. The interface

The agent is briefed with the navigation instruction and given exactly two tools:

| Tool | Direction | Payload |
|---|---|---|
| `observe` | pull | one egocentric RGB frame (512×512) — nothing else |
| `step` | act | a list of discrete actions: `0` STOP · `1` forward 0.25 m · `2` left 30° · `3` right 30° |

Everything else the agent brings itself: how often to look, how far to commit between observations, when to declare STOP. The probe measures the model, not an architecture.

Every run is a **cell** — harness × model × condition with every knob frozen in code (`coding-agent/cells.py`, std-v2 freeze): R2R-CE val-unseen, the fixed 100-episode `rand100` sample, 512 px RGB, 200 agent turns, 500 env actions, 2 400 s per episode. Deviating from the freeze requires `--nonstd`, which renames the run so it can never sit on the board.

## 2. Repository layout

| Path | What it is |
|---|---|
| `coding-agent/` | the probe. `harnesses/` (Claude Code SDK · Codex CLI · the in-repo mini loop) · `driver.py` (shared episode loop, single writer of the event vocabulary) · `cells.py` (the paper's board as code) · `bridges/` (the stdio MCP tool surfaces the agents see) · `stdrun.py` (CLI: run / batch / board / compare) |
| `workspace/nodesets/env/` | simulator wrappers served as HTTP toolfaces: `env_habitat` (VLN-CE R2R-CE), `env_hmeqa` (HM-EQA), `env_vlnverse` (VLNVerse / Isaac Sim 5.1), and peers |
| `agentcanvas/backend/` | the server that hosts any nodeset as an HTTP toolface (`auto_host`) — the infrastructure the probe talks to |
| `scripts/install/` | per-simulator conda env installers (`install_ac_vlnce.sh`, `install_ac_hmeqa.sh`, `install_ac_vlnverse.sh`, …) |
| `outputs/archive/` | raw trajectories of the early calibration and baseline runs (JSONL, one file per episode) |

The backend belongs to [AgentCanvas](https://github.com/jianzhou0420/AgentCanvas), our visual agent-design platform; the probe uses only its server mode and needs neither the canvas UI nor any graph.

## 3. Setup

**Step-by-step in [INSTALL.md](INSTALL.md)** — the sequence there was exercised end-to-end from scratch on a clean RTX 3090 host. In short:

**Environments.** Two conda envs run the main board: `agentcanvas` (Python 3.10, the backend and the runner) and `ac-vlnce` (Python 3.8, habitat-sim 0.1.7 for VLN-CE) — one install script each under `scripts/install/`. The other lines each add one env: `ac-hmeqa` (HM-EQA), `ac-vlnverse` (Isaac Sim 5.1), `ac-wp` (the waypoint-predictor server for the wp/hybrid cells).

**Data.** `data/` is gitignored; fetch R2R-CE episodes and Matterport3D scenes with `scripts/data/fetch_data_vlnce.sh`. The paper's evaluation split ships with the repo: `bash scripts/data/materialize_r2r_rand100.sh` installs `rand100` (the SmartWay/OpenNav 100-episode val-unseen protocol) into the data tree; other derived splits are audited via the tracked manifests in `coding-agent/bridges/splits/`.

**Harness auth.** `sdk` = a Claude subscription (the adapter strips a stray `ANTHROPIC_API_KEY`); `codex` = a ChatGPT login (`codex login`); `mini` = `ANTHROPIC_API_KEY` / OpenAI key via litellm for API models, or a local ollama for the qwen cells.

## 4. Reproduce the paper

Start the env server (one per worker), then run cells by name:

```bash
# env server (ac-vlnce interpreter, ports 9200+):
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/env/env_habitat.py \
  --class EnvHabitatNodeSet --port 9200

# then, from the agentcanvas env:
python coding-agent/stdrun.py run std_sdk_fable-5_bare_default   # the primary cell
python coding-agent/stdrun.py run std_codex_gpt-5.6_bare_default
python coding-agent/stdrun.py batch Ad        # the whole SDK default-effort row
python coding-agent/stdrun.py board           # grid status from summaries on disk
```

Cell names are the ground truth: `std_<harness>_<model>_<surface>_<tier>` — the main
board cells are `std_{sdk,codex,mini}_{model}_bare_default`, the effort ablation
`std_*_bare_max`. (`stdrun` also accepts legacy `E`-shortcuts from `cells.py ·
EXPERIMENTS`, e.g. `run E3` = the fable-5 primary cell; the published paper itself
carries no experiment numbering.)

Batch names: `Ad/Bd/Gd/Xd` (default-effort rows), `A/B/G/X` (max-effort), `Q` (local qwen, $0), `W`/`WQ` (waypoint), `EQ` (HM-EQA).

**Beyond the main board** — the same loop, unchanged, on other benchmarks and surfaces:

| Line | Cells | Notes |
|---|---|---|
| Waypoint / Hybrid interface | `std_*_wp`, `std_*_hybrid` | adds the SmartWay waypoint predictor as a tool (server on `:9210`; see `coding-agent/README.md`) |
| HM-EQA | `hmeqa_sdk_*` | multiple-choice EQA; episode ends by `answer()`, split `mip100` with tracked manifest |
| VLNVerse | `vlnverse_sdk_*` | Isaac Sim 5.1 behind `env_vlnverse` |
| Real robot | `go2_sdk_*` | Unitree Go2 pilots via `bridges/go2_bridge.py`; operator-supplied instruction, human-judged |

The paper's tables are transcribed in §5 below; on the R2R-CE board the bare probe with frontier models lands within reach of industrial-scale trained policies — without any of their machinery.

## 5. Results (from the paper)

All numbers below are transcribed from the paper. Protocol: R2R-CE val-unseen,
the fixed `rand100` sample, frozen knobs of §1. \* = mean over three replications
(± s.d. where shown). Trained external rows report the full val-unseen split, so
they are context, not a same-set comparison.

### Minimal interface among recent systems on R2R-CE (paper Tab. 1)

| System (model / policy) | Control | Source | Visual | Nav. machinery | SR↑ | SPL↑ |
|---|---|---|---|---|---|---|
| *Human* | — | — | — | — | 94 | 80.80 |
| NaVid | policy | trained | M | explicit video memory | 37 | 35.00 |
| NaVILA | policy | trained | M | VLA + RL gait | 54 | 49.00 |
| StreamVLN | policy | trained | M | slow-fast cache | 57 | 51.90 |
| Hy-Embodied-VLM (A3B) | policy | trained | M | frame-history context | 58 | 54.20 |
| RynnBrain-Nav (8B) | policy | trained | M | multi-turn dialogue memory | 59 | 49.60 |
| NavFoM | policy | trained | P | TVI tokens, budget sampling | 62 | 55.30 |
| OmniNav | policy | trained | M | flow-matching head | 70 | 66.10 |
| Qwen-RobotNav (w/o its planner) | policy | trained | P | waypoint head, task-adaptive obs. encoding | 72 | 66.60 |
| SmartWay (GPT-5.5) | workflow | zero-shot | P+D | explicit memory, waypoint, backtracking | 44 | 35.04 |
| Vesta | workflow | trained | M | planner + ext. controller | 56 | 50.80 |
| InternVLA-N1 / DualVLN | workflow | trained | M | dual-system, diffusion | 64 | 58.50 |
| ABot-N1 | workflow | trained | 3-cam | dual-brain, pixel goal | 71 | 67.50 |
| AgenticNav (GPT-5.5) | agentic | zero-shot | P+D | map, explicit memory, action tools | 55 | 48.41 |
| **Minimal (fable-5, mini-swe-agent)** | **agentic** | **zero-shot** | **M** | **none** | **72** | **59.08** |
| **Minimal (fable-5, Claude Agent SDK)** | **agentic** | **zero-shot** | **M** | **none** | **\*68.3** | **58.02** |
| **Minimal (opus-5, Claude Agent SDK)** | **agentic** | **zero-shot** | **M** | **none** | **\*70.7** | **55.21** |
| **Minimal (fable-5, Claude Agent SDK, max effort)** | **agentic** | **zero-shot** | **M** | **none** | **78** | **65.27** |

Visual input: M = monocular, P = panorama, D = depth.

### The main minimal-interface board (paper Tab. 2)

| Harness | Model | SR↑ | SPL↑ | NE↓ | OSR↑ |
|---|---|---|---|---|---|
| mini-swe-agent | qwen3.5-4b | 5 | 4.58 | 8.93 | 11 |
| mini-swe-agent | qwen3.5-9b | 7 | 5.36 | 8.63 | 15 |
| mini-swe-agent | qwen3.5-plus | 34 | 26.74 | 6.32 | 48 |
| mini-swe-agent | qwen3.7-plus | 42 | 32.85 | 6.81 | 55 |
| mini-swe-agent | qwen3.6-plus | 45 | 33.27 | 6.25 | 57 |
| mini-swe-agent | gpt-5.5 | 52 | 44.24 | 7.29 | 57 |
| mini-swe-agent | gpt-5.6 | 60 | 42.04 | 4.99 | 68 |
| mini-swe-agent | sonnet-5 | 53 | 38.14 | 5.52 | 61 |
| mini-swe-agent | opus-4.8 | 63 | 52.77 | 4.21 | 65 |
| mini-swe-agent | fable-5 | 72 | 59.08 | 4.48 | 77 |
| mini-swe-agent | opus-5 | 69 | 50.24 | 5.15 | 78 |
| Claude SDK | sonnet-5 | \*51.3 ± 1.2 | 37.84 ± 0.89 | 5.80 ± 0.43 | 61.3 ± 1.2 |
| Claude SDK | opus-4.8 | \*55.7 ± 2.3 | 47.31 ± 3.81 | 5.24 ± 0.44 | 59.3 ± 0.6 |
| Claude SDK | fable-5 | \*68.3 ± 1.5 | 58.02 ± 1.50 | 5.13 ± 0.26 | 73.3 ± 1.5 |
| Claude SDK | opus-5 | \*70.7 ± 3.5 | 55.21 ± 2.73 | 4.79 ± 0.43 | 78.3 ± 5.7 |
| Codex CLI | gpt-5.5 | 45 | 35.74 | 5.66 | 51 |
| Codex CLI | gpt-5.6 | 56 | 41.57 | 6.15 | 64 |
| Claude SDK | fable-5 (max effort) | 78 | 65.27 | 3.84 | 83 |

<img src="https://jianzhou0420.github.io/src/works/MIP/capability_axes.png" alt="Capability located across the model, harness, and interface axes" width="760">

### Reasoning-effort ablation

| Model | Harness | Effort change | SR | Δ |
|---|---|---|---|---|
| sonnet-5 | Claude SDK | default → max | \*51.3 → 56 | +4.7 |
| opus-4.8 | Claude SDK | default → max | \*55.7 → 56 | +0.3 |
| fable-5 | Claude SDK | default → max | \*68.3 → 78 | +9.7 |
| opus-5 | Claude SDK | default → max | \*70.7 → 74 | +3.3 |
| gpt-5.5 | Codex CLI | default → xhigh | 45 → 50 | +5 |
| gpt-5.6 | Codex CLI | default → xhigh | 56 → 62 | +6 |
| gpt-5.5 | mini-swe-agent | default → xhigh | 52 → 50 | −2 |
| gpt-5.6 | mini-swe-agent | default → xhigh | 60 → 55 | −5 |

### Interface ablation — primitive vs. waypoint

| Benchmark | Model | SR primitive | SR waypoint | Δ | SPL primitive | SPL waypoint |
|---|---|---|---|---|---|---|
| R2R-CE | qwen3.5-4b | 5 | 43 | **+38** | 4.58 | 34.18 |
| R2R-CE | qwen3.5-9b | 7 | 44 | **+37** | 5.36 | 32.81 |
| R2R-CE | qwen3.5-plus | 34 | 53 | **+19** | 26.74 | 44.05 |
| R2R-CE | gpt-5.5 | 45 | 67 | **+22** | 35.74 | 58.85 |
| R2R-CE | gpt-5.6-sol | 56 | 73 | **+17** | 41.57 | 63.98 |
| R2R-CE | sonnet-5 | \*51.3 | 60 | +8.7 | 37.84 | 45.83 |
| R2R-CE | opus-4.8 | \*55.7 | 65 | +9.3 | 47.31 | 54.23 |
| R2R-CE | fable-5 | \*68.3 | 69 | +0.7 | 58.02 | 59.15 |
| R2R-CE | opus-5 | \*70.7 | 72 | +1.3 | 55.21 | 60.52 |
| VLNVerse | sonnet-5 | 78 | 72 | **−6** | 52.06 | 22.86 |
| VLNVerse | fable-5 | 84 | 80 | **−4** | 62.47 | 42.57 |

### Hybrid interface (fable-5, Claude SDK, R2R-CE)

| Interface | Effort | SR↑ | SPL↑ | NE↓ | Steps | Time (s) | Calls |
|---|---|---|---|---|---|---|---|
| primitives | default | \*68.3 ± 1.5 | 58.02 ± 1.50 | 5.13 ± 0.26 | 87 | 210 | 39 |
| waypoint | default | 69 | 59.15 | 4.30 | 38 | 91 | 15 |
| hybrid | default | \*76.7 ± 0.6 | 63.63 ± 1.67 | 3.49 ± 0.23 | 48 | 112 | 20 |
| primitives | max | 78 | 65.27 | 3.84 | 97 | 484 | 48 |

Steps / Time / Calls are per-episode medians.

### Long-horizon stress (fable-5, Claude SDK, default effort)

| Interface | R2R-CE SR | R2R-CE Time (s) | R2R-CE Ctx | RxR-CE SR | RxR-CE Time (s) | RxR-CE Ctx |
|---|---|---|---|---|---|---|
| primitives | 70 | 187 | 22.4k | 26 | 527 | 49.2k |
| waypoint | 69 | 91 | 14.6k | 39 | 336 | 44.2k |

Time and Ctx are per-episode medians (seconds; final-call input tokens).

### The same loop beyond R2R-CE (paper appendix)

| Benchmark | System | Control | Source | SR↑ | SPL↑ |
|---|---|---|---|---|---|
| VLNVerse | Qwen-RobotNav | policy | trained | 64 | — |
| VLNVerse | **Minimal (fable-5, Claude Agent SDK)** | **agentic** | **zero-shot** | **84** | **62.47** |
| HM-EQA | Explore-EQA | workflow | zero-shot | 51.5 | — |
| HM-EQA | FAST-EQA | workflow | zero-shot | 69.2 | — |
| HM-EQA | planner × Qwen-RobotNav | agentic | trained | 76.7 | — |
| HM-EQA | **Minimal (fable-5, Claude Agent SDK)** | **agentic** | **zero-shot** | **76.2** | **—** |

## 6. Citation

```bibtex
@article{zhou2026embodied,
  title   = {Embodied Agents Take Control: Minimal-Interface Zero-Shot Agents
             Rival Industrial-Scale Policies in Vision-and-Language Navigation},
  author  = {Zhou, Jian and Zhao, Xunyi and Zhou, Gengze and Li, Zerui and
             Lin, Sihao and Liu, Jiajun and Wu, Qi},
  journal = {arXiv preprint arXiv:2607.26148},
  year    = {2026}
}
```

## 7. License & acknowledgements

Apache 2.0 (see [LICENSE](LICENSE)).

The probe stands on: [VLN-CE](https://github.com/jacobkrantz/VLN-CE) and [habitat-sim](https://github.com/facebookresearch/habitat-sim) (the R2R-CE benchmark and simulator), Matterport3D scenes, [SmartWay](https://github.com/sxyxs/SmartWay-Code)'s waypoint predictor (the wp/hybrid cells), HM-EQA from [Explore-EQA](https://github.com/Stanford-ILIAD/explore-eqa) (Ren et al., RSS 2024), VLNVerse, the [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) loop our `mini` harness ports, and the Claude Code and Codex CLI agents under study.
