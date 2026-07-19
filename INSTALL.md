# Installation

## Prerequisites

- **conda** (Miniforge/Miniconda) — Python, Node, and PyTorch all come from the env files, so you don't install them separately.
- **git**; plus an NVIDIA GPU + driver if you want to run neural policies.

## Quickstart — explore the UI

The fastest look at AgentCanvas: install the core env and open the canvas. No simulator,
data, or API key — you can browse the preset graphs, drag nodes, and inspect wiring. Running
a method end-to-end is the next quickstart.

```bash
git clone https://github.com/jianzhou0420/AgentCanvas.git
cd AgentCanvas
bash scripts/install/install_core.sh     # core env + frontend, then serves backend :8000 + frontend :5173
```

First run takes ~10 min (conda resolves PyTorch/CUDA, npm pulls React deps). Then open
**http://localhost:5173** and load a preset graph from the Explorer panel to poke around.

## Quickstart — run MapGPT on R2R

A complete minimal example: get one method running end-to-end. MapGPT is LLM-only
navigation on the Matterport3D Simulator — no local model weights, just the simulator,
the R2R data, and an API key.

```bash
# 1. Clone
git clone https://github.com/jianzhou0420/AgentCanvas.git
cd AgentCanvas

# 2. MapGPT's simulator env — MatterSim (GPU/EGL build; add --osmesa for CPU-only)
bash scripts/install/install_ac_mp3d.sh

# 3. Data: MP3D skyboxes (~18 GB, Matterport ToU) + preprocess + R2R episodes (~6 MB)
python3 scripts/data/fetch_scans_mp3d.py --accept-tos     # accepts the Matterport Terms of Use
bash    scripts/data/gen_skybox_rgb_mp3d.sh               # merge/downsize the skyboxes MatterSim reads
bash    scripts/data/fetch_episodes_vln.sh --r2r

# 4. LLM key — MapGPT's llmCall uses the gpt-5-mini profile (OpenAI)
export OPENAI_API_KEY=sk-...

# 5. Core canvas env + launch — installs agentcanvas, then serves backend :8000 + frontend :5173 (foreground)
bash scripts/install/install_core.sh
```

Env-creation order doesn't matter (each env is independent); only the launch in step 5 must come last, so the running canvas can drive the `ac-mp3d` env you built in step 2. Then open **http://localhost:5173**, load the **`mapgpt_mp3d`** graph from the Explorer panel, and hit **Play**.

## Doc-site (optional, no install)

```bash
bash docs/run_dev.sh        # http://localhost:8092, live reload
```

Read `core/blueprint.html`, `core/glossary.html`, and `core/architecture.html` first.

## Install scripts

Every env has one idempotent script under `scripts/install/` — run each as `bash scripts/install/<script>`; re-running updates the env in place. The core env alone runs the canvas + pure-LLM graphs; **add a server-mode env only when you need it**. Once an `ac-*` env exists, the matching nodesets auto-route to server mode under the `agentcanvas` backend.

**Core (canvas hub):**

| Script | Env | What it does |
|---|---|---|
| `install_core.sh` | `agentcanvas` | Core env + frontend, then launches backend `:8000` + frontend `:5173` (foreground) |
| `install_agentcanvas.sh` | `agentcanvas` | Same core env, install-only (no launch) — the building block `install_core.sh` calls |
| `install_core_test.sh` | `agentcanvas-test` | Core install into an isolated test env (for vetting install-script changes) |

**Server-mode (one isolated env each):**

| Script | Env | For |
|---|---|---|
| `install_ac_vlnce.sh` | `ac-vlnce` | VLN-CE / Habitat-Sim 0.1.7 (CMA, NavGPT-CE) |
| `install_ac_mp3d.sh` `[--osmesa]` | `ac-mp3d` | Matterport3D Simulator, built from source (R2R, RxR, REVERIE); `--osmesa` = CPU render |
| `install_ac_smartway.sh` | `ac-smartway` | SmartWay nodeset on the Habitat base |
| `install_ac_hmeqa.sh` | `ac-hmeqa` | HM-EQA evaluation (Habitat + HM-EQA scenes/questions) |
| `install_ac_vla_policy.sh` | `ac-vla-policy` | VLA policy stack (lerobot + libero + openpi-client) |
| `install_ac_libero.sh` | `ac-libero` | LIBERO manipulation suite |
| `install_ac_simpler.sh` | `ac-simpler` | SIMPLER eval (SimplerEnv + ManiSkill2) |
| `install_ac_octo.sh` | `ac-octo` | Octo JAX/Flax policy (fragile install; JAX is CPU-only here) |
| `install_ac_detany3d.sh` | `ac-detany3d` | 3D-detection model server (CUDA 11.8) |
| `install_ac_ram.sh` | `ac-ram` | RAM / RAM++ tagging + SpatialBot model server |

> Simulator scenes/episodes/checkpoints live separately under `scripts/data/` (e.g. `fetch_data_vlnce.sh`, `fetch_scans_mp3d.py`, `fetch_episodes_vln.sh`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Conda env 'agentcanvas' not found` | Run `install_core.sh` first |
| Frontend stuck on "Connecting…" | Backend isn't up on `:8000` — check the launcher terminal |
| Port in use (`:8000`/`:5173`/`:8092`) | Stop the holder or change the port |
| Server-mode eval returns 500 | The matching `ac-*` env isn't installed |

## Uninstall

```bash
conda env remove -n agentcanvas       # plus any ac-* envs you created
rm -f agentcanvas/launch.sh
```
