---
name: ledger-nav
description: Navigation algorithm designed for long-context coding agents — dead-reckoning odometry from metric actions, a frame ledger that indexes (not copies) past observations, a predict-execute-verify loop, and a guaranteed-termination stop protocol. Use for indoor instruction-following navigation with observe/step tools.
---

# LedgerNav — navigate like you code

You have two tools: `observe()` (egocentric RGB) and `step(actions)` (0=STOP,
1=forward **0.25 m**, 2=left **15°**, 3=right **15°**). Three facts make you
different from every prior VLN prompting method, and this algorithm is built
on all three:

1. **Your actions are metric.** You can compute your own position by
   arithmetic. Do it — you are good at arithmetic.
2. **Every frame you ever observed is still in your context.** You never need
   to describe-and-store what you saw; you need an INDEX that lets you find
   and re-look at the right frame later.
3. **You can verify.** Treat navigation like code: every move is a change,
   every observation is a test run, and STOP is a release — releases require
   passing tests.

## The Ledger (update after EVERY tool call, keep in every reply)

```
Pose: x=+2.50 y=-1.25 θ=90°   steps_used=68/500  decisions_used=9/~35
Frames:
  #0  t0  (0.0,0.0,0°)    start: pool ahead              LANDMARK "pool"
  #3  t4  (2.5,0.0,0°)    bar room entrance, junction    JUNCTION J1
  #5  t6  (2.5,-1.0,90°)  counter left, chairs right     LANDMARK "bar","chairs"
  #7  t9  (4.0,-1.0,90°)  counter END visible ~2m        GOAL-CANDIDATE ★0.8
Junctions:
  J1 @(2.5,0.0): exits 0°(taken) · 90°(taken) · 270° dark hallway (UNTRIED)
Best-stop: frame #7, pose (4.0,-1.0), score 0.8 — "corner where counter ends"
Plan: advance 1.5m along counter, EXPECT counter edge fills left frame
Progress: "past the pool" ✓ · "between bar and chairs" ✓ · "corner" DOING
```

**Pose arithmetic** (do it silently, exactly): left(2) adds 15° to θ,
right(3) subtracts 15°; forward(1) adds (0.25·cosθ, 0.25·sinθ) with θ=0°
pointing along your initial facing. Round to centimeters. The simulator
slides along obstacles — if a view barely changes after forward steps,
mark `[BLOCKED]`, trust the pose less, and re-anchor: match the current
view against a ledger frame and reset your confidence from there.

**Frames** get one line each: id, turn, pose, ≤6-word gist, plus flags —
`LANDMARK` when it shows something the instruction names, `JUNCTION` when
multiple ways forward exist, `GOAL-CANDIDATE ★score` when it could be the
final stopping place. The gist is a retrieval key, not a description: when
you need detail, scroll up and RE-READ the actual image. Never re-observe
what you can re-look at.

**Junctions** list untried exits with bearings. With pose coordinates you can
always compute the turn-and-walk sequence back to any junction — backtracking
is arithmetic, not luck.

**Best-stop** is the single most important register. The moment any frame
looks like the instruction's endpoint, score it (0-1) and record it. You are
never lost while this register is full: the worst case is "return and stop".

## The loop: predict → execute → verify

One decision per turn, always in this shape:

1. **Predict**: pick ONE leg (advance to X / turn to bearing β and cross to
   J2 / return to best-stop) and write the expectation:
   `EXPECT: after step([...]) I see <specific thing>`.
2. **Execute in one batch**: full legs, not twitches — face the bearing
   (n×15°), walk it (m×0.25 m, usually 8-16 forwards), then `observe()`.
   A 15°-turn-then-look turn is a wasted decision; you have ~35, spend them
   on legs.
3. **Verify**: grade your own expectation ✓/✗ against the new frame, update
   Ledger + Progress. **Two consecutive ✗ = navigation error**: stop
   predicting forward progress, pick the best UNTRIED junction exit (or
   best-stop if score ≥0.6), compute the return leg from poses, execute it.

Scan policy: corridors get zero scans — just look ahead. At a JUNCTION, a
room entry, or whenever the right way forward is not obvious, call
`look_around()`: ONE tool call returning four labeled views (ahead / right
+90° / behind +180° / left +270°) with your heading restored automatically
(θ unchanged; costs ~24 low-level turn steps, which you can afford). Log
every exit it reveals as a Junction entry. Never burn tool calls turning
15° at a time to scan — that is the single most common way to waste your
budget.

## Stopping — approach first, then a release checklist

**SEEING IS NOT ARRIVING.** When you first recognize the goal object/place it
is usually 4-8 m away; stopping there scores exactly zero, the same as never
finding it. You do not have to guess the distance: every observation and
every step result reports `clearance_m` — the REAL distance in meters to the
nearest obstacle in the left/center/right thirds of the view (10.0 = open,
≥10 m). Between "spotted" and "stop" there is ALWAYS an approach phase:

1. Turn until the goal sits in the CENTER third of the frame. Now
   `clearance_m.center` IS your distance to it — trust the number over your
   eyes.
2. Declare `GOAL APPROACH: <center clearance> m` and walk most of it in one
   leg: 4×(clearance − 1.5) forward steps (0.25 m each), then re-observe.
3. Close the rest in short legs (2-4 steps + observe), re-centering the goal
   whenever it drifts off-center, until the CLOSE-ENOUGH test passes.

**CLOSE-ENOUGH test — metric, on the CURRENT observation, never from memory:**
- goal is an object/furniture: it is in the center third and
  `clearance_m.center ≤ 2.0`;
- goal is a room / doorway / open area: you have already crossed its
  threshold — you stand one or two steps INSIDE with the described
  surroundings around you (clearance may legitimately read large here);
- sanity check: if `clearance_m.center > 3.0` toward your believed goal
  object, you are NOT close enough, no matter how big it looks.

**Placing the final stop — close-enough means the NEIGHBORHOOD, not the
point.** Success is measured to one exact spot; a plausible stop 4 m from
the right spot scores the same zero as never arriving. When the test passes
and `tool_calls_remaining > 15`, spend two of them on placement before
stopping:

1. `look_around()` once, and re-read the instruction's FINAL clause word by
   word. Ask: within 2-3 m of me, is there a spot that matches that wording
   better — the other corner, the doorway itself, right beside the named
   furniture?
2. Endpoint disambiguation rules, in order:
   - Instructions describe a path; when several similar spots qualify (two
     corners of the same bar, either side of a table), the endpoint is the
     one FARTHER ALONG your direction of travel — the far corner you reach
     last, not the near one you meet first.
   - "All the way" / "to the end" wording means keep going until
     `clearance_m.center < 1.5` in that corridor or room — the end is where
     you physically cannot continue, not where the view got repetitive.
   - Endpoints named "door / doorway / entrance" mean stand AT the opening
     (clearance jumps open right through it), not in the room beyond and
     not meters in front.
3. Walk the 1-3 steps to the better spot if one exists, then `step([0])`.

The environment enforces this: while budget is rich, your first `step([0])`
is WITHHELD and echoes this checklist back. That is not an error — do the
check honestly (a rushed re-confirm just converts a fixable near-miss into
a zero), then `step([0])` again to execute.

Only when Progress is fully ✓ AND the test passes (plus placement check if
budget allows): `step([0])`. Two symmetric ways to lose: stopping on sight
(too far — the most common failure) and drifting past the goal "to be sure"
(overshoot). The approach protocol prevents the first; stopping once
placement is checked prevents the second.

**Termination is guaranteed by budget law.** The environment reports your
real budget in every `step()` result: `tool_calls_remaining` counts down to
the moment the session is killed. Obey it mechanically:

- `tool_calls_remaining ≤ 20` (or a `BUDGET_WARNING` appears): exploration is
  OVER. Commit to Best-stop, start the approach.
- `BUDGET_WARNING` says CRITICAL: execute the terminal protocol THIS turn —
  navigate to Best-stop by dead reckoning (or stay, if the current spot scores
  higher) and `step([0])`. No further observe calls, no second thoughts.

Ending an episode without your own STOP is a bug, never an outcome — it
scores zero even if you are standing on the goal. If Best-stop is empty
(rare), stop at the most instruction-consistent place you can reach in two
legs.

## Micro-example

> "Walk past the pool. Walk between the bar and chairs. Stop at the corner
> of the bar."

t1: observe → pool ahead → frame #0, LANDMARK. Predict: 12 forwards, EXPECT
room entrance. Execute `step([1×12])`, observe.
t2: entrance = #3 JUNCTION J1 (dark hallway at 270° logged UNTRIED). Predict:
turn to 90°, 8 forwards, EXPECT counter on left, chairs right. Execute, ✓.
t3: #5 confirms both LANDMARKs; counter END visible ahead, centered, and the
step result says `clearance_m: {center: 4.1}` → #7 GOAL-CANDIDATE ★0.8,
Best-stop set. `GOAL APPROACH: 4.1 m` → 4×(4.1−1.5) ≈ 10 forwards, re-observe.
t4: corner centered, `clearance_m.center = 1.6` ≤ 2.0 — CLOSE-ENOUGH passes,
Progress all ✓ → `step([0])`. Total: 5 decisions, ~30 env steps, one junction
banked untried and never needed.
