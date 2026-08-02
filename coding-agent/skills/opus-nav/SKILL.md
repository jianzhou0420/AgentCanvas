---
name: opus-nav
description: Minimal decision-rule skill for strong models — route-verified grounding, orient-before-commit, systematic backtrack, decisive endpoint placement. No bookkeeping. Use for indoor instruction-following navigation with observe/step/look_around tools.
---

# Navigate by verifying the ROUTE, not the landmark

You already know how to move, look, and stop — no procedure here. First read
the instruction into a completion test (Step 0); then four decision rules
cover the four ways strong navigators actually fail.

## Step 0 — Turn the instruction into a completion test (BEFORE you move)

Read the whole instruction once and split it into two things, then keep both
in view the entire episode:

- **Waypoints** — the ordered path constraints (made-up example: "up the
  stairs", then "between the sofa and the bookshelf"). Things you must be
  SEEN TO PASS THROUGH, in order.
- **The completion test** — a concrete, EGOCENTRIC snapshot of what you will
  see standing exactly at the endpoint: what is at your LEFT hand, what is at
  your RIGHT hand, what is directly AHEAD, and how close (clearance). Phrase it
  so a decoy nearby FAILS at least one slot. Resolve ambiguity NOW with
  walker's-logic (Rule 4). At Step 0 this is a PREDICTION built from the
  instruction ALONE — you have not seen the endpoint, so you do not yet know
  which named thing ends up on which side; leave those slots BLANK to fill from
  observation. Mark it coarse and sharpen it as you approach (next section).

Draft it as left / right / ahead, never as an abstract summary. Take a made-up
instruction, "walk between the sofa and the bookshelf, then stop at the
fireplace on the far wall" — the FORM of its test is:

> DONE = I have passed BETWEEN the sofa and the bookshelf — both now BEHIND me,
> no longer one on each side of me — and I have reached the FAR wall (the one
> reached last, Rule 4) with the fireplace directly AHEAD, within reach.
> LEFT = ⟨fill from view⟩ · RIGHT = ⟨fill from view⟩ · AHEAD = ⟨the fireplace⟩, clearance ⟨read⟩.

The decoy this form kills is general, not tied to any scene: while the two
flankers are still ONE ON EACH SIDE of you, you are at the gap's ENTRANCE, not
through it — "between them" as a place you STAND is the trap that stops you
short; you have passed through only once both are BEHIND you. A left/right/ahead
test catches the entrance decoy because a side slot still reads a flanker; an
abstract "I'm near the fireplace" cannot.

## Two registers update every step — by OPPOSITE rules

Every observation brings new information. At each step update two things — but
they update differently, and conflating them is the mistake:

1. **Grounding — where am I on the route?** Bias hard toward STABILITY. Keep
   your current reading unless the new view genuinely CONTRADICTS it: an
   expected waypoint is absent, a place you thought was ahead is now behind
   you, a candidate endpoint fails the test up close. A grounding that still
   fits needs no revisiting — re-litigating one that holds is how you thrash,
   burn budget, and never commit. Change your mind only when evidence forces it.

2. **The completion test — what will "done" concretely look like?** This one
   you SHARPEN every step. The Step-0 draft was coarse — built from the
   instruction before you could see the endpoint. As that area comes into view,
   fill and rewrite the left / right / ahead slots with what is ACTUALLY in
   front of you now. Sharpening is NOT thrashing: the semantic target (the
   corner reached last) stays fixed; only its concrete signature — which object
   is on which side, what is ahead, how far — gets filled in from observation.
   You are done only when the sharpened, observed test matches the CURRENT view
   slot for slot: left, right, and ahead all confirmed on THIS frame, not from
   memory.

The trap the stability rule alone cannot catch: a coarse test ("I'm at a
corner that fits") stays true at the decoy AND the goal, so nothing
contradicts it and you stop early. Only a test that gets more concrete as you
see more starts FAILING at the decoy — a flank slot still shows the object you
were meant to pass — and holds you to the real endpoint.

## Rule 1 — The endpoint only counts if you reached it VIA the route

Instructions are a path, read them as an ORDERED checklist of clauses.
A final landmark reached without passing the intermediate ones is almost
always the WRONG INSTANCE — houses have two sinks, two bedrooms, two
hallways. Before you treat anything as the goal, ask: "did I visibly pass
every landmark named BEFORE it, in order?" If no — this is a decoy; keep
following the clause you are actually on.

When the environment withholds your first STOP and echoes a checklist, do
the audit FOR REAL: first restate your Step-0 completion test verbatim and
answer it yes/no on the CURRENT view; then list each waypoint and name the
specific past frame where you saw it satisfied. "I traced the full route"
without frame evidence is exactly how you stop at the wrong sink. If the
DONE test reads false, or any waypoint has no frame, you are at a decoy —
go find the missed landmark instead of stopping.

## Rule 2 — Orient before you commit (the first move decides the episode)

The single most common total failure is walking confidently in the wrong
initial direction. At the very start: `look_around()` once, match clause 1
against ALL four views, and only then move. If two directions both fit
clause 1, prefer the one that also fits clause 2. Re-apply this rule at any
junction where the instruction's next clause does not obviously select an
exit.

But orienting is one decision, not a mode: once a direction matches,
COMMIT — walk full legs and stop auditing a route that keeps matching.
Caution has a budget too: at most TWO look_around calls before your first
long leg, and never re-scan a spot you already scanned. A route that is
confirming clause after clause needs no second opinions.

## Rule 3 — When lost, rewind — never improvise

Two signs you are off-route: (a) two consecutive legs in a row failed to
show what the instruction predicts next, or (b) you recognize a place you
have already been. The moment either fires: stop exploring forward. Return
to the LAST position where the instruction still matched what you saw
(you have every past frame in context — find it), and take the best exit
you have not tried. A wrong route corrected early costs 10 steps; a wrong
route followed to a confident stop costs the episode.

Rewinding also has a limit: after TWO rewinds, sweeping the floor plan is
no longer buying information. Take the strongest partial match you have
seen, walk to it, and place your stop there — a best-guess endpoint on a
half-verified route beats a perfect audit that never stops.

Know the scale: these routes are SHORT — typically 8-15 m across 2-4
rooms, done in ~40 forward steps. The endpoint is never on the far side
of the house. If you have crossed MORE rooms than the instruction names,
you have left the route and are hunting decoys — rewind to where it last
matched instead of pressing outward. This is a reason to rewind, never a
reason to withhold a stop you are otherwise ready to make.

## Rule 4 — Decide the endpoint like a walker, not a philosopher

Endpoint wording is written by a human who walked this route once. When it
is ambiguous (made-up examples: "the end of the counter", "the covered
porch", "the door before the stairs"), do not deliberate over semantics
across multiple turns — apply these tie-breakers immediately and go:

- Among similar candidates, the endpoint is the one FARTHER ALONG your
  direction of travel (the one you reach last, not first).
- "Before X" means on the near side of X's threshold, one step short —
  not inside X, not meters away.
- "Wait at / near X" means within arm's reach of X: center X in view and
  close until `clearance_m.center ≤ 1.5`. Success is judged at 3 m from an
  exact spot you cannot see — your visual "close enough" is reliably ~1 m
  too generous, so always take the extra 2-3 forward steps while the path
  is clear. Nobody ever failed by standing too close to the right thing.
- Room-or-outside contradictions (made-up: "the covered porch, but stop
  before stepping outside"): stand AT the boundary — the doorway or
  threshold itself.

Seeing the goal is not arriving: when you first recognize it, it is
usually 4-8 m away. Center it, read `clearance_m.center`, walk the gap,
then stop. If the budget report turns CRITICAL, go to your best candidate
and `step([0])` this turn — an unstopped episode scores zero from any
position.
