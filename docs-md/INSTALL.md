# Installation

This guide walks a new user through getting AgentCanvas running from a fresh clone.

## Prerequisites

Make sure you have these installed before running anything:

| Tool | Version | Why |
|---|---|---|
| **conda** (Miniforge/Miniconda) | any recent | Environment management |
| **Node.js** | ≥ 18 | Frontend build (Vite + React) |
| **Git** | any recent | Clone repo + submodules |
| **CUDA 12.1** *(optional)* | for GPU ML | Only if running neural policies like CMA |

## Step 1 — Clone

```bash
git clone <repo-url> AgentCanvas
cd AgentCanvas
git submodule update --init --recursive
```

The submodules include `third_party/Matterport3DSimulator` and `third_party/VLN-CE`.

## Step 2 — Install everything

```bash
bash scripts/install/install_all.sh
```

This orchestrator script does:

1. **`scripts/install/install_agentcanvas.sh`** → creates the `agentcanvas` conda env from `agentcanvas/environment.yaml`. Installs Python 3.10 + PyTorch 2.5 + CUDA 12.1 + transformers + SAM + FastAPI deps. Then runs `npm install` in `agentcanvas/frontend/`.
2. **Generates a launcher** (`agentcanvas/launch.sh`) with the conda env name baked in. Gitignored.
3. **Launches the canvas**: backend `:8000` + frontend `:5173` in the foreground. Press Ctrl+C to stop.

The doc-site is pure-stdlib HTML — no install needed. Run `bash docs/run_dev.sh` to serve it on `:8092` with live reload.

**First run takes ~10 minutes** — mostly conda resolving PyTorch + CUDA and npm pulling React deps.

### What gets created

```
vlnworkspace/
├── agentcanvas/
│   └── launch.sh              # generated — activates agentcanvas, runs uvicorn + vite
```

Conda env:
- `agentcanvas` — Python 3.10, the general-purpose env (ADR-platform-004)

## Step 3 — Verify it's running

Open in your browser:

- **http://localhost:5173** — canvas UI (drag nodes, build graphs, hit Play)
- **http://localhost:8000/docs** — backend Swagger API
- **http://localhost:8092** — documentation site (after `bash docs/run_dev.sh`)

## Step 4 — Read the essentials

After starting the doc-site (`bash docs/run_dev.sh`), read these three pages first:

1. **Blueprint** (`docs/core/blueprint.html`) — the *"one JSON = one agent = one graph"* philosophy behind AgentCanvas.
2. **Glossary** (`docs/core/glossary.html`) — ~50 terms: `BaseCanvasNode`, `IterIn/IterOut`, wire types, state containers.
3. **Architecture** (`docs/core/architecture.html`) — frontend ↔ backend ↔ `workspace/` component flow.

## Step 5 *(optional)* — Install the `vlnce` env for Habitat

Only needed if you want to run VLN-CE navigation (CMA policy, Habitat env). The `vlnce` env is a separate manual install because of the habitat-sim 0.1.7 binary constraint (ADR-server-001, ADR-platform-004).

```bash
bash scripts/install/install_ac_vlnce.sh              # creates vlnce env from scripts/install/envs/ac_vlnce.yaml
# Or follow third_party/VLN-CE/README.md manually
```

Download data:

```bash
bash scripts/data/fetch_data_vlnce.sh                 # MP3D scenes, VLN-CE episodes, CMA checkpoint
```

Once `vlnce` is ready, habitat/policy_cma nodesets auto-route to server mode under the `agentcanvas` backend via `?mode=server`.

## Step 6 *(optional)* — Install the `mp3d` env for Matterport3D Simulator

For the discrete panoramic Matterport3D Simulator (used by R2R, RxR, REVERIE, CVDN, R4R benchmarks):

```bash
bash scripts/install/install_ac_mp3d.sh               # builds MatterSim from source + creates mp3d env
bash scripts/install/install_ac_mp3d.sh --osmesa      # CPU-only build (no GPU)
bash scripts/install/install_ac_mp3d.sh --status      # check installation status
```

The mp3d env is created from `scripts/install/envs/ac_mp3d.yaml` (Python 3.9 + PyTorch 2.1 + CUDA 11.8).

## Step 7 — Start building

- **Load a preset graph**: Explorer panel → `Straightforward` (6-node CMA) or `NavGPT-CE` (12-node LLM reasoning).
- **Build from scratch**: drag nodes from the sidebar, wire them, hit Play.
- **Save as graph node**: freeze a composite graph for reuse across other graphs.
- **Drop your own components**: add Python files under `workspace/` — they're auto-discovered at backend startup. Subclass `BaseCanvasNode` or `BaseNodeSet`.

---

## Alternate workflows

### Docs only

```bash
bash docs/run_dev.sh                       # serves on :8092 with live reload
bash docs/run_dev.sh 9000                  # custom port
```

No install required — the dev server uses Python stdlib only.

### Canvas only (env already set up)

```bash
bash agentcanvas/launch.sh                 # uses agentcanvas env
# Or the lightweight alternative (assumes conda env already active):
bash agentcanvas/run_dev.sh
```

### Isolated test envs (for CI / testing install changes)

```bash
bash scripts/install/install_all_test.sh       # creates agentcanvas-test + agentcanvas-docs-test
```

This is useful when you want to verify changes to the install scripts or YAML files without touching your working envs.

---

## Troubleshooting

| Symptom | Cause & Fix |
|---|---|
| `No module named 'PIL'` in viewer nodes | ML stack not fully installed — re-run `bash scripts/install/install_all.sh` (Pillow comes from the conda YAML) |
| Backend starts but `/api/eval/v2` fails with 500 | `vlnce` env not installed — server-mode nodesets (Habitat, CMA) need it |
| Frontend hangs on "Connecting..." | Backend not running on `:8000` — check `agentcanvas/launch.sh` terminal output |
| `Conda env 'agentcanvas' not found` when running launch.sh | Run `bash scripts/install/install_all.sh` first, or check that env name in `launch.sh` matches an existing conda env |
| Port already in use | Another process holds `:8000`, `:5173`, or `:8092` — stop it or change the port in the launcher |
| `npm install` fails with permission errors | Check that `agentcanvas/frontend/node_modules/` is writable; delete it and re-run the installer |
| Pre-commit hook fails on commit | Run `pre-commit run --all-files` to see what's wrong — fix and re-commit |

---

## Uninstall

To fully remove AgentCanvas from your system:

```bash
conda env remove -n agentcanvas
conda env remove -n vlnce                  # if you installed it
conda env remove -n mp3d                   # if you installed it
rm -rf agentcanvas/frontend/node_modules
rm agentcanvas/launch.sh
# Delete the repo directory
```

---

## Project layout

```
vlnworkspace/
├── INSTALL.md                       # this file
├── CLAUDE.md                        # project instructions for Claude Code
│
├── scripts/
│   ├── install/                     # ALL install scripts live here
│   │   ├── install_all.sh           # full install orchestrator
│   │   ├── install_all_test.sh      # test variant with isolated env names
│   │   ├── install_agentcanvas.sh   # canvas conda env + frontend npm + generates launch.sh
│   │   ├── install_ac_vlnce.sh         # vlnce env (Habitat-Sim, manual)
│   │   ├── install_ac_mp3d.sh          # mp3d env (MatterSim build, manual)
│   │   └── envs/                    # env yamls without a "home" directory
│   │       ├── vlnce.yaml
│   │       └── mp3d.yaml
│   └── setup_data.sh                # data downloader (not install)
│
├── agentcanvas/                     # the AgentCanvas app
│   ├── backend/                     # FastAPI backend (Python 3.10+)
│   ├── frontend/                    # React + TypeScript + Vite
│   ├── environment.yaml             # canvas conda env definition
│   ├── launch.sh                    # generated launcher (gitignored)
│   └── run_dev.sh                   # lightweight alternative
│
├── workspace/                       # user workspace — drop components here
│   ├── nodesets/                    # BaseNodeSet subclasses (auto-discovered)
│   ├── graphs/                      # saved editable graph templates
│   ├── graph_nodes/                 # frozen composite graph nodes
│   ├── policies/                    # policy definitions
│   └── skills/                      # skill nodes
│
├── docs/                            # Hand-authored HTML doc-site (no build step)
│   ├── assets/                      #   shared style.css + nav.js
│   ├── index.html                   #   root landing page (/ route)
│   ├── pages/
│   │   └── developer-guide/         #   13 sections: capabilities, core, design-docs, ...
│   ├── _lib/                        #   build infra
│   │   ├── _layout.py / _nav.py     #     layout shell + tab/section catalog
│   │   ├── _wrap_handwritten.py     #     re-render chrome over all pages
│   │   └── _serve.py                #     pure-stdlib live-reload dev server
│   └── run_dev.sh                   #   user entry point — invokes _lib/_serve.py
│
├── third_party/                     # git submodules (Matterport3DSimulator, VLN-CE)
└── data/                            # datasets, checkpoints (gitignored)
```

For deeper project structure and architecture, read `docs/core/architecture.html`.
