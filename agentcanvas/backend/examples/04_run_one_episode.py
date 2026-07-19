"""Example 04 — run ONE MapGPT-MP3D episode in-process.  [EXPERIMENT]

    PYTHONPATH=. python examples/04_run_one_episode.py

⚠ This is a real env run: g.run(load_nodesets=True) scans the workspace
registry, auto-loads the mapgpt (local) + env_mp3d (server-mode) nodesets —
spawning an env_mp3d subprocess in the ac-mp3d conda env — drives the executor
for one episode, then tears the env server down. Needs the ac-mp3d env, MP3D
data, and a gpt-5-mini key. It uses GPU + the LLM, so it is an *experiment*;
a real multi-episode sweep belongs behind /experiment:run (see example 05 for
the batch entry point, still an experiment).

A single g.run() drives the executor once = one episode; it resets to
episode 0 of the split the env panel defaults to. Metrics land in
result.metrics (from the evaluate node) and result["metrics"] (the graphOut).
"""

from __future__ import annotations

from app.mapgpt_mp3d_sdk import build

if __name__ == "__main__":
    g = build()
    print("running one episode in-process (load_nodesets=True) — spawning env_mp3d server…")
    result = g.run(load_nodesets=True, validate=True)
    print("metrics:", result.metrics)
    print("graphOut['metrics']:", result.outputs.get("metrics"))
