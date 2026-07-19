"""Example 05 — batch-eval MapGPT-MP3D over N episodes.  [EXPERIMENT]

    PYTHONPATH=. python examples/05_batch_eval.py            # 2 episodes
    PYTHONPATH=. python examples/05_batch_eval.py 10         # 10 episodes
    PYTHONPATH=. python examples/05_batch_eval.py 10 val_unseen 0

⚠ Real env run. g.eval(...) drives the same BatchEvalRunner the backend eval
API uses — one executor run per episode, metrics averaged over completed
episodes — but in-process, no FastAPI. It auto-loads env_mp3d (spawns the
sim server in the ac-mp3d env), runs the batch, then tears it down. Needs the
ac-mp3d env, MP3D data, and a gpt-5-mini key; uses GPU + the LLM. This is an
*experiment* — a large sweep belongs behind /experiment:run.

Returns an EvalResult: .metrics (aggregate, e.g. success_rate / spl),
.episodes (per-episode), .by_task.
"""

from __future__ import annotations

import sys

from app.mapgpt_mp3d_sdk import build

if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    split = sys.argv[2] if len(sys.argv) > 2 else "val_unseen"
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    print(f"batch eval: {episodes} episode(s) of R2R/{split} from index {start} …")
    result = build().eval(episodes=episodes, dataset="R2R", split=split, start_index=start)

    print("\n=== per episode ===")
    for e in result.episodes:
        sr = e["metrics"].get("success_rate", e["metrics"].get("success", "?"))
        print(f"  ep{e['index']:>3} [{e['status']:>9}] scene={e['scene'][:12]:<12} "
              f"steps={e['steps']:>3} success={sr}")
    print("\n=== aggregate (mean over completed) ===")
    for k, v in sorted(result.metrics.items()):
        print(f"  {k:<22} {v}")
