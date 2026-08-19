# wp — the waypoint-selection surface (MIP §4.4 interface arm)

Folder = method arm: depth-predicted candidate waypoints numbered on a
4-view panorama; the agent picks goto(n) or stop (wp_bridge copy). Migrated
from bridges/wp_bridge.py + prompts.py wp branch 2026-08-18, byte-parity
(specs exp_dir-only / briefing incl. the wp_max_moves=45 variant / tool
surface).

## Profiles
- **r2r** — historical `std_{sdk,codex,mini}_*_wp` names (sdk×4 · codex×2 ·
  mini local 4b/9b + API plus×3), STD_FROZEN fallback, WP_MAX_TURNS 100 /
  wp_max_moves 30 from orchestration. Batches W · WQ.
- **rxr** — `rxr_sdk_{model}_wp` ×4 (batch RXW), frozen from the paper's
  tab:long-horizon fable run (rxr_wp_fable_rand100): RxR-CE / rand100_en /
  max_turns 300 / wp_max_moves 45 / auto_observe off (cell extra). The
  opus-5/sonnet RxR-wp archives (mt150/moves50) are a DIFFERENT protocol —
  off-board.

## Dependencies
Env auto_host = this folder's nodeset (byte copy of exp_workspace/bare's,
ac-vlnce python) + the smartway_waypoint predictor auto_host (:9210, ac-wp
env over the SHARED coding-agent/ac_wp_predictor_shim tree — infra asset).
mini cells ride harnesses/mini/toolset.py WaypointToolSet (byte gate
check_equivalence.py vs the SHARED bridges/wp_bridge.py — folder copy and
shared file must not drift).

## Boards
r2r wp column ran pre-exp_workspace from the code this folder froze; rxr
seats are new (live smoke pending before the first board).

Rule: NEVER edit — fork instead.
