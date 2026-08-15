# Installation

Everything the paper's main board needs: **two conda envs, the R2R-CE data, and one
login per agent harness**. The sequence below was exercised end-to-end from scratch
(envs deleted first) on a clean Ubuntu host with an RTX 3090 on 2026-08-15.

## Prerequisites

- **conda** (Miniforge/Miniconda) — Python, PyTorch, and Node all come from the env files.
- **NVIDIA GPU + driver with EGL support** — habitat-sim renders headless through EGL.
- **git**; disk budget ~35 GB: the two envs ≈ 15 GB, MP3D scenes ≈ 15 GB, episodes < 1 GB.
- Network able to reach GitHub and Google Drive (episode data is fetched from the
  official VLN-CE Drive links).

## 1. Environments

```bash
bash scripts/install/install_agentcanvas.sh   # runner env `agentcanvas` (~10 min)
bash scripts/install/install_ac_vlnce.sh      # simulator env `ac-vlnce` (~20 min)
```

Both scripts are idempotent (re-running updates in place). `install_ac_vlnce.sh`
auto-clones pinned copies of VLN-CE and habitat-lab into `third_party/` and ends
with an import probe of the full server wire stack — if you see `[ok]`, the env works.

## 2. Data (R2R-CE)

```bash
bash scripts/data/fetch_data_vlnce.sh --r2r    # episodes (250 MB, Google Drive)
bash scripts/data/fetch_data_vlnce.sh --mp3d   # Matterport3D scenes (~15 GB, interactive ToU)
bash scripts/data/materialize_r2r_rand100.sh   # the paper's evaluation split (versioned in-repo)
ln -sfn ../habitat/scene_datasets/mp3d data/scene_datasets/mp3d 2>/dev/null \
  || (mkdir -p data/scene_datasets && ln -s ../habitat/scene_datasets/mp3d data/scene_datasets/mp3d)
bash scripts/data/fetch_data_vlnce.sh --status # verify
```

The resulting layout:

```
data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/rand100/   # the paper split
data/habitat/scene_datasets/mp3d/<scan>/                     # MP3D scenes
data/scene_datasets/mp3d -> ../habitat/scene_datasets/mp3d   # compatibility symlink
```

`rand100` is the SmartWay/OpenNav evaluation protocol — a 100-episode val-unseen
subset with real spawn headings and regenerated ground-truth paths. It cannot be
derived from the official files by filtering, so the split ships whole with this
repository (~710 KB); provenance in `coding-agent/bridges/splits/r2r_rand100/README.md`.

## 3. Harness auth

| Harness | Needs |
|---|---|
| `sdk` (Claude Code) | a Claude subscription login (`claude` once, interactively); the adapter strips a stray `ANTHROPIC_API_KEY` |
| `codex` (Codex CLI) | a ChatGPT login (`codex login`) |
| `mini` | `ANTHROPIC_API_KEY` / OpenAI key via litellm for API models; a local [ollama](https://ollama.com) for the qwen cells |

## 4. Verify

Start the simulator server, check its manifest, then run one real episode:

```bash
# terminal 1 — env server (ac-vlnce interpreter):
cd agentcanvas/backend && PYTHONPATH=$PWD:$PWD/../.. \
  ~/miniforge3/envs/ac-vlnce/bin/python -m app.server.auto_host \
  --file ../../workspace/nodesets/env/env_habitat.py \
  --class EnvHabitatNodeSet --port 9200

# terminal 2:
curl -s http://127.0.0.1:9200/manifest | head -c 200   # server up (~10 s after start)

# one real episode through the probe (agentcanvas env, needs harness auth):
python coding-agent/stdrun.py run std_sdk_fable-5_bare_default --episodes 0
```

## Other experiment lines (optional, one env each)

| Line | Install | Notes |
|---|---|---|
| Waypoint / Hybrid (`*_wp`, `*_hybrid`) | see `coding-agent/ac_wp_predictor_shim/README.md` | SmartWay predictor server; checkpoints via `scripts/data/fetch_ckpt_smartway.sh` |
| HM-EQA (`hmeqa_*`) | `bash scripts/install/install_ac_hmeqa.sh` | HM3D scenes + questions; split manifest in `coding-agent/bridges/splits/` |
| VLNVerse (`vlnverse_*`) | `bash scripts/install/install_ac_vlnverse.sh` | Isaac Sim 5.1 (own bundled python); data via `scripts/data/fetch_{episodes,scenes}_vlnverse.py` |
| RxR-CE | `scripts/data/fetch_data_vlnce.sh --rxr` + `scripts/data/make_rxr_rand100.py` | builds the scene-matched RxR `rand100` |
| Canvas UI (optional) | `bash scripts/install/install_core.sh` | the visual platform on `:8000`/`:5173` — not needed for the board |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `conda not found` inside scripts | conda isn't on PATH in that shell — `source ~/miniforge3/etc/profile.d/conda.sh` or run from a login shell |
| env server import errors | re-run `install_ac_vlnce.sh` (idempotent) and check its trailing wire-stack probe |
| `reset` fails with a scene/dataset error | data layout incomplete — `bash scripts/data/fetch_data_vlnce.sh --status`, and check the `data/scene_datasets/mp3d` symlink |
| Server-mode call returns 500 | the matching `ac-*` env isn't installed |
| Port in use (`:9200`) | stop the holder or pass another `--port` |

## Uninstall

```bash
conda env remove -n agentcanvas
conda env remove -n ac-vlnce      # plus any other ac-* envs you created
```
