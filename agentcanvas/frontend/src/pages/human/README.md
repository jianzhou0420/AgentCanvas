# Human tab — human-performance test on env_habitat

An interactive **human baseline** for VLN-CE. A person drives one episode by
keyboard through the *same* `env_habitat` auto_host the coding-agent runs use,
so the metrics come from habitat's own ruler (`env_habitat__evaluate`) — SR /
OSR / NE / nDTW / SPL identical to the agent experiments. Default target is the
`rand100` split (100 episodes), rendered at **800 px** and shown in a large
square frame that fills the left panel (sized by viewport height so it stays
fully visible).

Open it from the header **Human** tab (next to *Coding Agent*).

## Pieces

| File | Role |
|---|---|
| `agentcanvas/backend/app/services/human_runner.py` | `HumanRunner`: owns one `env_habitat` auto_host (spawn/reuse), one live session; `load_episode` / `step` / `stop` drive it; records trajectory + summary under `outputs/human/{split}/`. |
| `agentcanvas/backend/app/api/execution/human.py` | `/api/human/*` router — thin shell over `HumanRunner`. Frames ride back inline as base64 PNG. |
| `agentcanvas/frontend/src/pages/human/HumanPage.tsx` | The page: large square frame + instruction, keyboard control loop, episode nav, 100-cell status grid, live metrics. |

Wiring: `main.py` (router + lifespan init/shutdown), `state.py` (`human_runner`),
`store.ts` (`human` app mode), `App.tsx`, `components/Header.tsx`.

## Controls

| Key | Action |
|---|---|
| `↑` | forward 0.25 m (action 1) |
| `←` | turn left 15° (action 2) |
| `→` | turn right 15° (action 3) |
| `Enter` | STOP (action 0) — opens a confirm dialog; press `Enter` again to confirm (or `Esc` to cancel), then the episode is scored |

On-screen D-pad buttons mirror the keys. Keys are ignored while typing in an
input. If the step budget (500) is exhausted, a **Finish & Score** button
appears so the episode is still recorded.

## Flow

1. **Start Session** → spawns (or reuses) the `env_habitat` auto_host. Cold
   scene load is a few seconds; the pill shows `env: starting → ready`.
2. Pick an episode (0–99, prev/next/type-a-number) → **Load** / **Re-test**.
3. Follow the instruction with the arrow keys; `Enter` to STOP.
4. Metrics appear; the grid cell turns green (success) / red (fail). Re-test
   any episode to overwrite its record.
5. **End Session** tears the env down and frees the GPU.

## Multiple testers (auto-save + Clear)

Built for running the same 100 episodes across several people:

- **Auto-save on completion** — the moment a tester finishes all 100, the run is
  automatically copied to `outputs/human/archive/{split}_{timestamp}/` (with a
  `RESULTS.md` scorecard). A green banner confirms it. Nothing to click.
- **Clear** (grid header, **double confirmation**) — wipes the live grid so the
  next person starts empty. It archives first as a safety net *unless* the exact
  data is already saved (so the normal complete→clear flow leaves one copy, and
  a discarded partial run is still never lost). Archives live in a sibling
  `archive/` dir, so clearing a split never touches saved runs.

Rename an archive folder to the tester's real name whenever you like.

## HTTP API (`/api/human`)

| Method + path | Purpose |
|---|---|
| `POST /start-server` `{split}` | Spawn/reuse the auto_host. **Blocking** until ready (runs on a pooled thread — see gotcha). |
| `POST /stop-server` | Tear down the auto_host. |
| `GET /server-status` | `{state, error, split, url, session}`; proactively flips to `error` if the env died. |
| `POST /episode/{i}/load` `{rgb_resolution}` | Place + arm episode `i`; returns instruction + first frame. |
| `POST /step` `{action}` | One discrete move (1/2/3); returns new frame + `done`. |
| `POST /stop` | STOP (if live) + evaluate; persists the record. Auto-archives on the run reaching all 100. |
| `GET /status?split=` | Per-episode records + aggregate + `{complete, archived_to}` (from `summary.json`). |
| `POST /clear` | Wipe the live split for the next tester (archives first unless already saved). |

## Outputs

Everything lands under `outputs/human/{split}/` (e.g. `outputs/human/rand100/`):

- `episode_{i}.jsonl` — full trajectory: `episode_meta`, one `step` per action
  **with `position` / `orientation` coordinates**, `stop`, `metrics`.
- `summary.json` — per-episode records (tested / success / all metrics / step
  count) + running aggregate. Re-testing overwrites that episode's record.

## Gotchas (important)

- **Run the backend WITHOUT `uvicorn --reload` for real sessions.** `--reload`
  restarts the worker on any `.py` save, and the habitat auto_host (a child
  process armed with `PR_SET_PDEATHSIG`) dies with it → the next action fails
  with `HTTPConnectionPool: Max retries exceeded`. Data/artifact writes
  (`*.jsonl`, `*.json`) are safe — only `.py` edits trigger it. Recovery mid-
  session: click **Start Session** again (it detects the dead env and respawns).

  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```

- **One GPU, one session at a time.** The Human tab and the Coding-Agent
  Monitor tab each spawn their own `env_habitat` auto_host — don't run both at
  once.

- **Env auto_hosts must be spawned from a long-lived thread** (`asyncio.
  to_thread`, not a transient `threading.Thread`): `PR_SET_PDEATHSIG` fires
  when the *spawning thread* exits, so a short-lived thread would kill the env
  the instant it returns.
