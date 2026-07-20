---
name: wp-ledger-nav
description: Waypoint-selection navigation algorithm — a dead-reckoned pose ledger over metric goto() moves, an anti-circling guard that refuses to re-tread visited ground, instruction sub-goal ticking, and a guaranteed-termination stop protocol. Use for indoor instruction-following with the observe/goto/stop panorama tools.
---

# WaypointLedgerNav — navigate the panorama like you code

You have three tools: `observe()` (a Left/Front/Right/Back panorama with
numbered green waypoints, each listed with `direction`, `angle_deg` — degrees
left of your heading, negative = right — and `distance_m`), `goto(n)` (turn to
waypoint n and walk to it), and `stop()`. Three facts make you different from
prior VLN methods, and this algorithm is built on all three:

1. **Your moves are metric.** `goto(n)` turns by that waypoint's `angle_deg`
   and walks its `distance_m`. You can compute your own pose by arithmetic —
   and a pose ledger is exactly how you know you are **not going in circles**.
2. **Every panorama you observed is still in your context.** Don't re-describe
   what you saw; keep an INDEX that lets you find and re-look at the right one.
3. **You can verify.** Treat navigation like code: every `goto` is a change,
   every `observe` is a test run, `stop()` is a release — releases require
   passing tests.

## The Ledger (update after EVERY tool call, keep in every reply)

```
Pose: x=+3.25 y=-1.40 θ=105°   moves_used=6/40
Visited (dead-reckoned):
  P0 (0.0, 0.0)    start: pool ahead, bar to the left      LANDMARK "pool"
  P1 (2.4, 0.0)    junction: hallway Left, kitchen Front   JUNCTION
  P2 (3.25,-1.4)   kitchen entrance, counters on Right     LANDMARK "kitchen"
Came-from: the waypoint in my Back arc (angle_deg near ±180°) returns to P1
Best-stop: P2 "kitchen entrance", score 0.6
Plan: take the Front waypoint deeper in, EXPECT the sink/stove to appear
Progress: "into the kitchen" ✓ · "by the sink" DOING · "stop at sink" TODO
```

**Pose arithmetic** (do it silently, exactly): for `goto(n)` whose waypoint
reads `angle_deg=a, distance_m=d`, update `θ += a` (degrees, CCW-positive =
Left), then `x += d·cos θ`, `y += d·sin θ`, with θ=0° along your initial
facing. Round to 0.1 m. The predictor reports angles relative to your CURRENT
heading at each `observe()`, so always add `a` to the running θ.

**Visited** gets one line per position you stood at: id, dead-reckoned (x,y),
a ≤6-word gist, and a flag — `LANDMARK` when the panorama shows something the
instruction names, `JUNCTION` when several ways lead onward, `GOAL-CANDIDATE
★score` when this spot could be the endpoint. The gist is a retrieval key, not
a description: when you need detail, scroll up and RE-READ that panorama image.

**Best-stop** is the most important register. The moment any spot looks like
the instruction's endpoint, score it 0–1 and record it. You are never lost
while it is full — worst case is "return there and stop".

## Anti-circling — the rule that matters most

You go in circles by taking waypoints that undo your last move. Two guards,
both mechanical, checked BEFORE every `goto`:

1. **Never take the Back waypoint that points where you came from.** After a
   move, the candidate in your Back arc (`direction: "Back"`, `angle_deg` near
   ±180°) almost always returns to the previous position. Take it ONLY if the
   instruction explicitly says turn around / go back, OR you hit a dead end and
   Best-stop is behind you.
2. **Pose-check the candidate.** Compute where it lands (arithmetic above). If
   that is within ~1.5 m of a Visited entry that is NOT your intended goal, you
   are about to re-tread — pick a different, UNVISITED direction instead.

If your dead-reckoned pose returns near the same Visited entry **twice**, you
are oscillating in a pocket: stop exploring it, commit to Best-stop, approach,
stop. If EVERY candidate re-treads, it is a dead end — go to Best-stop, or take
the single least-recently-visited exit once and re-evaluate.

## The loop: predict → execute → verify

One decision per turn, always this shape:

1. **Predict**: pick ONE waypoint that advances the CURRENT instruction
   sub-goal, toward an UNVISITED direction, passing both anti-circling guards.
   Write `EXPECT: after goto(n) I see <specific thing>`.
2. **Execute**: `goto(n)`, then `observe()` to see the new surroundings.
3. **Verify**: grade your expectation ✓/✗ against the new panorama; update
   Pose, Visited, Progress. **Two consecutive ✗** = you are off-route: return
   to the last good Visited entry (compute the waypoint heading back to it) and
   take a different exit, or commit to Best-stop if its score ≥ 0.6.

## Matching the instruction to the panorama

The instruction is a sequence of sub-goals — tick them off ONE at a time.
For the current sub-goal, read the four labeled views: the waypoint whose
`direction` and view match the sub-goal's landmark is your pick. `action_options`
lists which numbered waypoints fall in each of Left/Front/Right/Back. When no
view matches, take a **Front** waypoint (keep going straight) rather than
turning back — turning back is how you re-tread.

## Stopping — approach first, then a checklist

**SEEING IS NOT ARRIVING.** When the goal first appears it is often one or two
waypoints away; the `distance_m` of the waypoint heading toward it is how far.
Between "spotted" and "stop" there is always an approach: take the waypoint
toward it, `observe()`, repeat. Stop only when the CURRENT panorama passes:

- object/furniture goal: its landmark sits in the CENTER (Front) of the
  panorama AND the nearest Front waypoint's `distance_m` ≤ ~2 m; or
- room / area / doorway goal: you have already crossed into it — its described
  surroundings are around you in the panorama.

**Placement** (endpoint disambiguation, in order): several similar spots (two
corners, either side of a table) → the one FARTHER along your travel direction,
the far one you reach last. "All the way / to the end" → keep taking Front
waypoints until no Front waypoint remains (you physically cannot continue).
"Door / doorway / entrance" → stand AT the opening, not in the room beyond.

**Termination is guaranteed by budget law.** Every result reports
`moves_remaining`. Obey it mechanically:

- `moves_remaining ≤ 5` (or a `MOVE_WARNING` appears): exploration is OVER.
  Head to Best-stop and `stop()`.
- Ending an episode without your own `stop()` scores ZERO even if you are
  standing on the goal. If Best-stop is empty (rare), `stop()` at the most
  instruction-consistent place you can reach in ONE move.

Two symmetric ways to lose: stopping on sight (too far — the common failure)
and drifting past the goal (overshoot, usually from circling back). The
approach protocol prevents the first; the anti-circling guard prevents the
second.

## Micro-example

> "Walk into the kitchen. Walk by the sink and oven. Stop by the sink."

t1: `observe()` → pool behind me, kitchen doorway in Front = wp 2 (2.3 m). P0
logged, LANDMARK "pool". Predict: `goto(2)`, EXPECT kitchen interior.
t2: `observe()` → counters on Right, sink ahead-left = wp 1 (Front, 1.8 m); wp
3 is Back, 2.3 m, angle ≈ 180° → returns to P0, do NOT take it. P1 kitchen,
"into the kitchen" ✓. Predict: `goto(1)`, EXPECT sink close and centered.
t3: `observe()` → sink centered in Front, nearest Front waypoint 1.4 m,
moves_used 3/40. "by the sink" reached, sink centered and ≤ 2 m → `stop()`.
Total: 3 moves, one Back waypoint refused, no circle.
