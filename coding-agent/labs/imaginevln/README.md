# ImagineVLN — world model as an on-demand `imagine` tool

Claude-SDK multi-turn navigation harness (auto-observe, waypoint `goto`) where a
learned world model is a **tool the model calls only when it wants to**:
`imagine([1,3,4])` renders a predicted walk-through for the named waypoints.
Three arms live here — `imagine` / `base` (same harness, no tool) / `--no-refine`.
Design + results: blackboard :8100 → ImagineVLN.

## Layout

```
run_imagine_sdk.py     runner: SDK session, refine skill, arms, artifact contract
imagine_tools.py       toolset: imagine / goto / stop, view-aligned rollouts, auto-observe
merge_summaries.py     merge sharded summary_w*.json when running N workers in parallel
service/mw_service.py  resident world-model service (:9270), T5 dropped (mw_notext.py)
cache/prompt_embed.pt  cached constant text embedding (1.8 MB, checked in)
wp_shim/               sitecustomize shim so smartway predictor loads its depth encoder
                       in an env without habitat_sim (see below)
test_*.py              view-alignment math (offline) + depth/A-B checks (needs services)
```

Weights are NOT in git: put MemoryWorld's export at `<repo>/data/mw_export`
(`CogVideoX-2b/`, `ckpt/`, `code/`) or set `MW_EXPORT=/path/to/mw_export`.

## Services (all `auto_host`, run from `agentcanvas/backend`)

```bash
# habitat env (ac-vlnce)
python -m app.server.auto_host --file ../../workspace/nodesets/env/env_habitat.py --class EnvHabitatNodeSet --port 9200
# waypoint predictor — MUST load the DDPPO depth encoder or every candidate collapses
# to a constant fan (see below). Use an env with habitat_sim (ac-smartway), or:
PYTHONPATH=<repo>/coding-agent/imaginevln/wp_shim:<repo>/third_party/habitat-lab \
python -m app.server.auto_host --file ../../workspace/nodesets/method/smartway_waypoint/__init__.py --class SmartWayWaypointNodeSet --port 9210
# world model (env "mw": torch cu128, diffusers==0.32.2, transformers==4.46.3)
python coding-agent/imaginevln/service/mw_service.py --port 9270
```

Sanity check for the predictor: predict twice from two headings — candidates
must be spread (>60°) and differ between headings. Identical `[−132°..−12°]×5`
every time = depth encoder failed to load, results are garbage.

## Run

```bash
# agentcanvas env; claude CLI logged in (subscription). Output → outputs/beta-imaginevln/<run>
python coding-agent/imaginevln/run_imagine_sdk.py --arm imagine --episodes 0-99 --split rand100 --model claude-opus-5 --out sdk_opus5_imagine
python coding-agent/imaginevln/run_imagine_sdk.py --arm base    --episodes 0-99 ... --out sdk_opus5_base
python coding-agent/imaginevln/run_imagine_sdk.py --arm base --no-refine ...
# N-way parallel (base arm only — the world model is single-instance):
#   one habitat per worker (ENV_URL=http://127.0.0.1:920k), --worker-tag wk, then
#   python coding-agent/imaginevln/merge_summaries.py outputs/beta-imaginevln/<run>
```

Runs show up in the AgentCanvas coding-agent monitor under the **ImagineVLN** board
(`SOURCE_ROOTS["imagine"]` → `outputs/beta-imaginevln`).

## Context contract

Past rounds' imagine images never re-enter context: the session is rebuilt
(`interrupt()`) after any imagine→goto, and the rebuilt opening carries journey +
panorama history only. Panoramas are kept forever (image_window=0 semantics).
`context` events log `stale_imagine_sheets` per session — must stay 0.
