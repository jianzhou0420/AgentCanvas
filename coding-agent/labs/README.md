# labs — parked harness lines (off the std board)

The `refactor(eval)` restructure (`fc0661b`) settled coding-agent into four
active bands — `core/` (shared machinery the running board imports),
`exp_workspace/<arm>/` (self-contained experiment arms, each with an `exp.py`
the loader picks up), `reporting/`, and `splits/` + `scripts/`. As part of that
it **retired the eharness / vlaharness / ImagineVLN lines** from the board.

This directory is where those lines are **parked, not deleted** — kept whole,
with their git history, for experiments that will want them again (e.g. an
ObjectNav arm that puts SAM landmarks on the map). It is a deliberate fifth
band, held apart from the four active ones.

```
labs/
├── eharness/    embodied-harness shell + the SAM3 engine (depthmap.py mask
│                projection / metric-depth decode, landmarks.py LandmarkOrgan)
├── vlaharness/  VLA agent line (agent_loop / state / judge / planner / toolset)
├── imaginevln/  ImagineVLN world-model agent (imagine_model / _tools / _toolset,
│                service/ world-model RPC, wp_shim/)
└── sam_arms/    the SAM3-on-map experiment arms, parked with their engine:
    slamsam_r2r_01 · slamsam_rxr_01 · slamsam_ovon_01 · slamsam_objnav_02
```

## The invariant that keeps the board green

**Nothing under `core/` or `exp_workspace/` imports anything in `labs/`.**
The board's arm loader globs `exp_workspace/*/exp.py` only, so a folder parked
here is never scanned — the std board loads, `stdrun.py board` renders, and the
Monitor lists + displays every run under `outputs/` exactly as before, all
independent of this code. The only trace of these lines on the active side is a
path constant (`OUTPUT_ROOTS["imagine"] → outputs/beta-imaginevln`) so their
historical runs still show in the Monitor, plus a couple of provenance comments
citing `eharness.depthmap` in the bare/hybrid/wp nodesets. Neither imports labs.

## Running a parked line in place (the common case)

You don't have to move anything back to tweak-and-run eharness / vlaharness /
ImagineVLN. `run.py` establishes the path + the old-name aliases (`driver`,
`prompts`, `toolset`, `harnesses.*` → their `core.*` homes; see `_bootstrap.py`)
and runs the target as `__main__`, so a line runs exactly as it did before the
restructure — while `core/` and the std board stay decoupled from it.

```bash
PY=~/miniconda3/envs/agentcanvas/bin/python   # the driver interpreter

# ImagineVLN
$PY coding-agent/labs/run.py coding-agent/labs/imaginevln/run_imagine_sdk.py --episodes 0-4 --arm imagine
# VLA harness
$PY coding-agent/labs/run.py coding-agent/labs/vlaharness/run.py --n 5 --mode agent
# eharness smoke / a script-style test
$PY coding-agent/labs/run.py coding-agent/labs/eharness/smoke.py
# eharness pytest suites (conftest.py bootstraps the same path+aliases)
$PY -m pytest coding-agent/labs/eharness/test_depthmap.py
```

Runtime services are unchanged: ImagineVLN still needs its world-model RPC
(`imaginevln/service/mw_service.py`, :9270) up; an eharness line that drives a
sim still needs its env auto_host. `run.py` only re-establishes the import
seams — it doesn't start servers. (A few eharness `test_*.py` are script-style
and `sys.exit()` at import, so run those through `run.py`, not `pytest`.)

## Reactivating an arm

`sam_arms/<arm>/exp.py` is import-clean (only `importlib`/`pathlib` at module
top), so the engine is pulled in only when the arm actually runs — through its
`bridge.py` / `toolset.py`, which do `from eharness import depthmap` /
`from eharness.landmarks import landmark_phrases`. To bring one back:

1. Move the arm into `exp_workspace/` (the loader then registers its cells), and
2. make `eharness` importable — either add `labs/` to `PYTHONPATH`, or, to honor
   the self-contained-arm rule, vendor the eharness modules the arm needs into
   its own `nodeset/` (the way `slam_01` carries its own `_slam.py`).

## Provenance

- `eharness/`, `vlaharness/`, `sam_arms/*` — lifted from the pre-restructure
  working tree, snapshotted on branch `wip/pre-jian-restructure` before the
  reset onto the restructure.
- `imaginevln/` — restored from `dev/coding-agent` (first landed at `7204ddb`,
  "bring ImagineVLN agent into the repo"). Its `cache/*.pt` prompt-embedding
  blob is regeneratable and kept local-only (gitignored).
