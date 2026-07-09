---
name: opus-nav
description: The one navigation skill for the opus hill-climb, grown one discipline at a time. v6 = grounding that fixes a geometric DONE-test per clause + per-step completion tracking + ordered traversal + odometer-gauged stopping with overshoot. It fights the agent's habit of narrating a route as complete while standing metres short: grounding now sets a concrete success test per clause, every step re-answers "how much of the instruction is done", and the STOP is gated on the odometer, not the eye. Further disciplines get added into THIS same skill over debugging iterations.
---

# Ground with success tests, track completion every step, stop by the odometer

## 1. Grounding — after the opening look_around, before any step

Open the episode with a single `look_around`. The moment its eight labeled views
come back (ahead, ahead-right, right, behind-right, behind, behind-left, left,
ahead-left), analyze the scene against the instruction BEFORE you `step`:

1. **Correspond** — read the instruction as an ordered list of its referents
   (every landmark and spatial phrase). For each, name the ONE view that shows
   it, or write "not visible".
2. **Interpret** — restate the instruction as the ordered sequence of physical
   moves you will make. Every clause is an action to perform, not just context.
3. **Set a DONE-test for each clause** — this is the heart of grounding: splitting
   into sub-goals is not enough; you must fix the geometric condition that PROVES
   each one is complete, so you later CHECK it instead of eyeballing "close
   enough":
   - **"go straight past X"** — done only when the WHOLE length of X has passed
     behind you (walk its full extent; drawing level with its near end is not
     "past").
   - **"between X and Y"** — done when X and Y lie on directly OPPOSITE sides of
     you: X, you, Y form a straight line (three points in a line), NOT a triangle
     with both off to one side. If X and Y are both long/elongated (a counter, a
     row of chairs), "between" means the MIDDLE of the stretch where they OVERLAP
     — centred side-to-side, along their shared span.
   - **"at the corner / end of X"** — done when you are beside the FAR end of X
     (X no longer runs ahead of you), within reach of it.
4. **Pin the endpoint** — from the endpoint clause's DONE-test, name where exactly
   the route ends. If it could point at more than one object or corner, list every
   candidate.
5. State which of the eight headings your first move takes. (Your spawn heading is
   arbitrary — a large turn, even ~180°, to face the route is normal and correct.)

Write this analysis out explicitly, then face the chosen heading and begin.

## 2. Walk the route as an ORDERED traversal

The instruction is a route to walk, not a landmark to point at. Walk its clauses
in order; each is a GATE you must satisfy (by its §1 DONE-test) before moving to
the next. The trap: spotting the goal landmark and beelining to the first spot
that looks like it, stopping at its near edge with the true endpoint still ahead.
A near corner or a passage mouth that merely looks like the endpoint is a decoy —
the goal is the point you reach only after every earlier gate's DONE-test holds.

## 3. After every step, locate yourself in the instruction

After each move-and-observe, answer in one line: **which clause am I on, which
clauses are DONE (by their §1 tests), and how much of the route is left?** e.g.
"clause 1 done — the whole pool is now behind me; on clause 2 — bar on my left,
chairs on my right but not yet on opposite sides of me, so not between them yet;
endpoint not reached." This running check is how you catch a skipped or half-done
clause in the moment, instead of discovering at the STOP that you jumped ahead.
Never advance the completion count on a clause whose DONE-test you cannot actually
see satisfied.

## 4. How you are scored, and WHEN you may stop

Success is binary and unforgiving. You score 1 only if you `step([0])` (STOP)
while within **3 metres, geodesic** (walkable distance along the floor, not
straight-line through walls) of the goal endpoint. Stopping even 3.5 m short, or
never stopping, scores 0. A shorter successful path scores higher — but getting
inside the 3 m ring dominates, and your standing failure is stopping too SHORT,
so lean toward one leg too many.

Your eyes lie about distance — a monocular view makes a landmark 4 m away look
"right here". Gate every STOP on two checks you can actually ground:
- **The odometer.** `step` returns `steps_taken_total`; each forward (action 1)
  is 0.25 m. These routes run ~8–12 m — on the order of 35–50 forward steps.
  A full route is rarely under ~35 forward steps: if you are under that and not
  physically blocked, you have NOT arrived, even if a corner looks right beside
  you. Do not round your own count up; keep walking.
- **The endpoint drawn level or behind, plus all clauses DONE (§3).** Never STOP
  while the endpoint object still runs ahead of you, while a clear path continues
  straight toward where you placed the endpoint, or while any earlier clause's
  DONE-test is unmet.

When both pass, close the last gap GENEROUSLY: your eye stops you 1–4 m short, so
take 4–6 more forward steps toward the endpoint (until you are right against it,
it is behind you, or you are blocked) before `step([0])`. Overshooting the goal
by a metre still scores; stopping a metre short scores zero — when in doubt, take
the extra leg. When several objects could be "it", stand within 3 m of as many as
possible first.
