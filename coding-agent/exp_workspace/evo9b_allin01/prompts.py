"""evo9b_allin01 — briefing = evo9b_g0's bare briefing + the all-in skill block.

Forked from exp_workspace/evo9b_g0/prompts.py (itself a byte-copy of the bare
arm's). The ONLY delta vs the parent: SKILLS_BLOCK below and its one-line
append in build_briefing. The block carries cand_01..cand_06 skill_texts
VERBATIM from labs/evolution/m1_candidates/ (cand_07 clause-ledger held out —
its known failure mode, legitimizing hallucinated progress, could suppress
the whole probe). Never edit after boards run.
"""

_OBS_NOTE_SEP = ""
_STEP_NOTE_SEP = ""
_STEP_LOOP_SEP = (
    "- Alternate observing and stepping: look, decide where the instruction "
    "wants you to go next, move, look again."
)

BARE_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).{obs_note}
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.{step_note}

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
{loop_rule}
- You have a budget of {budget} movement actions.
- You succeed only if you issue action 0 (STOP) while within 3 meters of the \
instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look around \
when unsure.
- Work autonomously until you stop; nobody can answer questions.
"""

SKILLS_BLOCK = """
Navigation skills — habits taken from expert runs of this exact task. Follow \
them:

1. Batch your actions. Plan your movement in batches, not single actions. In \
every step() call, issue a SEQUENCE of actions: first the turns to face your \
chosen direction, then several forward moves along it — for example \
step([2,2,1,1,1,1]). A typical batch is 4-8 actions. Issue a single action \
only when you are about to STOP or squeezing through a tight doorway. After \
each batch, look at the new view and plan the next batch.

2. Scan before walking. Before your first forward move, and whenever the \
object named in your current instruction step is not visible: do not walk. \
Scan in place instead — turn left 6 times, observe, and repeat until you \
have seen a full circle. The moment the named object appears, stop scanning, \
face it, and walk toward it. If a full circle shows nothing matching, walk a \
few steps toward the largest open passage and scan again. Never walk forward \
hoping the target will appear.

3. Center your target, then advance. Once you have picked a visible target \
(doorway, stairs, rug, furniture), center it before walking: if it sits in \
the left third of the image, turn left 1 or 2 times; in the right third, \
turn right 1 or 2 times; then observe again. Walk forward only while the \
target is near the horizontal center of the view. After every observation, \
re-center it the same way, then continue forward.

4. Finish properly. Do not issue STOP while the final object or room named \
in the instruction is out of view. When you can see it, keep walking toward \
it until it is close and fills a large part of the image, then take 2 or 3 \
more forward steps — into the room if told to enter, right up to the object \
if told to stop near it. Observe once to confirm it is directly ahead or \
beside you, then issue STOP immediately. Do not keep exploring after this \
confirmation.

5. Calibrate your turns. Each turn action rotates exactly 15 degrees. \
Convert instruction words into fixed counts: "turn left" or "turn right" \
means 6 turn actions (90 degrees); "turn around" means 12 (180 degrees); a \
slight veer or a centering fix is 1 or 2. After completing an instructed \
turn, observe once and confirm a walkable passage or the next named object \
is ahead before moving forward; if it is not, adjust with 1-2 more turns \
rather than walking.

6. If blocked, pivot. After every step call, compare the new view to the \
previous one. If you sent forward actions but the view did not change, you \
are blocked by a wall or furniture. Never resend the same forward sequence. \
Instead, turn 2 or 3 actions toward whichever side of the image shows more \
open floor, move forward 2 to 4 steps, observe, and then re-center your \
original target. If you are still blocked, try the opposite side the same \
way.
"""


def build_briefing(instruction: str, step_budget: int) -> str:
    return BARE_SYSTEM_PROMPT.format(
        instruction=instruction, budget=step_budget,
        obs_note=_OBS_NOTE_SEP,
        step_note=_STEP_NOTE_SEP,
        loop_rule=_STEP_LOOP_SEP,
    ) + SKILLS_BLOCK
