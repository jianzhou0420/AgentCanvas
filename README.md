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
| `coding-agent/` | the probe. `harnesses/` (Claude Code SDK · Codex CLI · the in-repo mini loop) · `driver.py` (shared episode loop, single writer of the event vocabulary) · `cells.py` (the paper's board as code, E-numbers included) · `bridges/` (the stdio MCP tool surfaces the agents see) · `stdrun.py` (CLI: run / batch / board / compare) |
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

Start the env server (one per worker), then run cells by name or by the paper's experiment number:

```bash
# env server (ac-vlnce interpreter, ports 9200+):
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/env/env_habitat.py \
  --class EnvHabitatNodeSet --port 9200

# then, from the agentcanvas env:
python coding-agent/stdrun.py run E3          # paper §4.1: SDK · fable-5, bare, default effort
python coding-agent/stdrun.py run std_codex_gpt-5.6_bare_default
python coding-agent/stdrun.py batch Ad        # the whole SDK default-effort row
python coding-agent/stdrun.py board           # grid status from summaries on disk
```

The E-number registry (`cells.py · EXPERIMENTS`) maps the paper's §4 numbering to cells:

| Paper section | Experiments | Cells |
|---|---|---|
| §4.1 main board (default effort) | E1–E9, E29–E30 | `std_{sdk,codex,mini}_{model}_bare_default` |
| §4.3 effort ablation (elevated) | E16–E20, E31 | `std_*_bare_max` |

Batch names: `Ad/Bd/Gd/Xd` (default-effort rows), `A/B/G/X` (max-effort), `Q` (local qwen, $0), `W`/`WQ` (waypoint), `EQ` (HM-EQA).

**Beyond the main board** — the same loop, unchanged, on other benchmarks and surfaces:

| Line | Cells | Notes |
|---|---|---|
| Waypoint / Hybrid interface | `std_*_wp`, `std_*_hybrid` | adds the SmartWay waypoint predictor as a tool (server on `:9210`; see `coding-agent/README.md`) |
| HM-EQA | `hmeqa_sdk_*` | multiple-choice EQA; episode ends by `answer()`, split `mip100` with tracked manifest |
| VLNVerse | `vlnverse_sdk_*` | Isaac Sim 5.1 behind `env_vlnverse` |
| Real robot | `go2_sdk_*` | Unitree Go2 pilots via `bridges/go2_bridge.py`; operator-supplied instruction, human-judged |

Full numbers are in the paper; on the R2R-CE board the bare probe with frontier models lands within reach of industrial-scale trained policies — without any of their machinery.

## 5. Citation

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

## 6. License & acknowledgements

Apache 2.0 (see [LICENSE](LICENSE)).

The probe stands on: [VLN-CE](https://github.com/jacobkrantz/VLN-CE) and [habitat-sim](https://github.com/facebookresearch/habitat-sim) (the R2R-CE benchmark and simulator), Matterport3D scenes, [SmartWay](https://github.com/sxyxs/SmartWay-Code)'s waypoint predictor (the wp/hybrid cells), HM-EQA from [Explore-EQA](https://github.com/Stanford-ILIAD/explore-eqa) (Ren et al., RSS 2024), VLNVerse, the [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) loop our `mini` harness ports, and the Claude Code and Codex CLI agents under study.
