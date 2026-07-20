---
name: mapgpt-nav
description: Map-guided navigation strategy (MapGPT, arXiv 2401.07314) adapted to raw egocentric control — build a textual topological map of Places while navigating, plan over it, backtrack through it, and stop deliberately. Use for any indoor instruction-following navigation episode.
---

# MapGPT navigation (map-guided prompting, adapted)

You navigate with only two tools: `observe()` (one egocentric RGB frame) and
`step(actions)` (0=STOP, 1=forward 0.25 m, 2=left 15°, 3=right 15°). This skill
gives you the MapGPT discipline: instead of wandering frame-by-frame, you build
and maintain a **textual topological map** of the places you explore, plan
multi-step routes over that map, and backtrack through it when you go wrong.

Upstream MapGPT chooses among simulator-provided candidate viewpoints; here
**you create the Places yourself** at decision points. Everything else — the
map bookkeeping, the planning discipline, the backtracking rule — is the same.

## The five memory blocks

Repeat these five blocks, updated, in EVERY reply. The conversation is your
only memory — if it is not written down, you have forgotten it.

```
Trajectory: Place 0 1 3 3 5          ← visit sequence, IDs in discovery order
Map:
  Place 0 is connected with Places 1, 2
  Place 1 is connected with Places 0, 3
  Place 3 is connected with Places 1, 4, 5
Supplementary Info:                   ← seen but never visited = backtrack targets
  Place 2 (60° right of Place 0): dark hallway toward kitchen
  Place 4 (left of Place 3): glass door to patio
Previous Planning: reach bar area (Place 5?), then find its far corner
Progress: "past the pool" DONE · "between bar and chairs" DOING · "stop at corner" TODO
```

- A **Place** is a decision point: a room entry, junction, doorway, or any spot
  where multiple ways forward exist. Number them in discovery order, starting
  at Place 0 (your start). Give each a 3-8 word visual tag when first seen.
- **Map** lines use exactly the upstream shape: `Place i is connected with
  Places j, k`. Add a connection when you travel it or clearly see it.
- **Supplementary Info** lists places you observed but never entered, with
  their rough bearing from the Place you saw them from. These are your only
  recovery options when navigation goes wrong — never let this list rot.
- **Progress** aligns the instruction clauses against your History: mark each
  clause DONE / DOING / TODO. You may have already executed more of the
  instruction than you think; judge from images, not from optimism.

## Per-Place decision loop

1. **Scan** when you arrive somewhere new or feel lost: repeat 4×
   { `observe()` → `step([3,3,3,3,3,3])` } — four views, 90° apart, one full
   turn. At plain corridors a single `observe()` is enough; scan at real
   decision points only (scans are expensive: ~4 turns each).
2. **Update the map**: new Places for new openings; new connections; move a
   Place from Supplementary to Trajectory when you enter it.
3. **Think in options**, exactly like MapGPT's action list:
   ```
   Action options:
   A. stop
   B. go forward to Place 6 (the archway ahead)
   C. turn left to Place 2 (dark hallway, from Supplementary)
   D. turn around and backtrack to Place 1 via Place 3
   ```
   Pick ONE letter, justify it in one Thought sentence against Instruction +
   Map + Progress, update Previous Planning → New Planning.
4. **Execute in bulk**: translate the chosen option into ONE `step()` call —
   e.g. face the target (n×15° turns) then advance (m× forward, ~2-4 m at a
   time), then `observe()` to confirm. Never spend a whole turn on a single
   15° twitch; batch every leg.

## Backtracking (the point of the map)

If two consecutive observations contradict the instruction (wrong room, dead
end, landmark missing), declare a navigation error in Thought. Then choose the
most promising Supplementary place, read the route to it off the Map
(`current → … → target`), and execute the whole return leg as batched steps.
Do not re-explore Places already in Trajectory unless the Map route passes
through them.

## Stopping — decide it, don't drift into it

Success means calling `step([0])` within 3 m of the goal. Two hard rules:

- When Progress shows every clause DONE and the current view matches the final
  clause's landmark, **walk up to it until it fills a good part of the frame
  (~2 m) and stop immediately**. Lingering "to be sure" only moves you off the
  goal; a near-miss stop scores exactly zero, and so does never stopping.
- Budget discipline: you have a limited number of decisions. If two full
  scan-plan-move cycles produce no Progress change, pick between (a) the best
  Supplementary backtrack and (b) stopping at the best goal-matching spot seen
  so far — continuing to wander is the only guaranteed-zero choice.

## Worked micro-example

> Instruction: "Walk past the pool. Walk between the bar and chairs. Stop at
> the corner of the bar."

Turn 1: observe → pool ahead. Map: Place 0 (start, pool view). Options: B. go
forward past the pool. Execute `step([1,1,1,1,1,1,1,1])`, observe.
Turn 2: bar room entrance = Place 1, connected 0-1. Bar counter left, chairs
right → the "between" corridor is a new Place 2; a dark hallway right goes to
Supplementary as Place 3. Plan: traverse Place 2, find bar's far corner.
Turn 3: advance through Place 2 alongside the counter — do not stop at the
NEAR corner; the instruction's corner is where the counter ENDS. At the far
corner, counter edge ~1.5 m away: Progress all DONE → `step([0])`.
