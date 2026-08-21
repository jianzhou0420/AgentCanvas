"""Solo runner — one NavAgent over the wrapped toolset for the whole episode.

No planner, no receipts, no heartbeat channel: the model sees everything
itself; the harness contributes guards, the pre-stop gate, keyframes/recall,
the state block (as an ephemeral message via SoloModel), and L2 compaction.
This is the +S1+S2+S3(+L2) arm — the solo paradigm of solo-harness.html.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from eharness.executor import SubgoalEnvironment, _jinja_raw
from eharness.wrapper import HarnessedToolset

SOLO_ADDENDUM = (
    "\n\n--- HARNESS NOTES ---\n"
    "A [CURRENT STATE] header arrives with every turn — it is your shared "
    "memory. The instruction is parsed into route SEGMENTS (each a landmark "
    "plus a stretch of walk) and one TERMINATION condition (the point where "
    "you STOP — shown at the top; it is what the verifier checks at the "
    "end). Update the state via the optional fields on your action calls: "
    "progress_note / advance_subgoal / landmark / dead_end (one short phrase "
    "each). Set advance_subgoal=true each time you complete one segment.\n"
    "Older turns' narration is cleared from your context automatically — "
    "your CURRENT thought belongs in progress_note/expect, not in prose "
    "you hope to re-read later.\n"
    "recall(query) shows stored keyframes by landmark name or frame number "
    "([frame#N] tags mark each view) — free, costs no steps.\n"
    "Before a move you may fill expect (where it should land you) — the "
    "state header holds you to it next turn. Pace by distance: nothing "
    "important nearby → 4-6 forward steps per call; a landmark at medium "
    "distance → 2-3 steps; final approach or a turn coming → single "
    "steps. Never more than 6 forwards in one call — the harness truncates "
    "the rest.\n"
    "Your robot HAS A BODY (a ~0.2m-radius cylinder): it needs clearance "
    "on both sides and cannot squeeze past furniture. If an obstacle sits "
    "in the CENTER of your view a few steps ahead, veer a notch around it "
    "BEFORE pushing forward — walking straight at it will jam you.\n"
    "Keep the state honest as you look: surroundings = what is around you "
    "(your understanding of the environment), updated whenever the scene "
    "changes.\n"
    "If you get physically stuck, look at the view, pick the side that "
    "looks open and call veer('left'/'right') — the harness makes the small "
    "turn + forward step FOR you and notes the outcome in your journey "
    "line. Read 'journey so far' in the state header to know what your "
    "walk has amounted to.\n"
    "Set near_goal=true the MOMENT you believe the final goal is in sight "
    "or close — it lights the NEAR-STOP flag and arms arrival verification "
    "on every move.\n"
    "Your first STOP request is checked by a verifier; if it is withheld, "
    "look again, then request STOP a second time to execute it."
)

# The depth-waypoint surface keeps every primitive AND adds goto(). So the
# pacing paragraph is not dropped — it is re-aimed: the real skill on this
# surface is choosing WHICH tool fits the distance, and the failure mode to
# pre-empt is a model that discovers goto() and then never takes a small step
# again, over-shooting the last metre where precision decides success.
DWP_ADDENDUM = (
    "You have no look action and need none: EVERY action returns a fresh view "
    "with the places you can walk to numbered on it, a line about the space "
    "around you, and what the detector recognises. Looking costs nothing and "
    "happens for you — spend your turns deciding, not looking.\n"
    "goto(place) walks to one numbered place: the middle of open floor, a few "
    "metres away, straight line already measured clear and wide enough for "
    "your body. THIS is how you cover ground. step([...]) is for turning, "
    "lining up and the final metre (1 = forward 0.25 m, 2/3 = turn 15° "
    "left/right, 0 = STOP); turns batch freely but only 2 forward moves per "
    "call are carried out, so pushing forward blind is not an option.\n"
    "Decide by putting the three readouts together: the PLACES (where you "
    "could go), AROUND YOU (how far and which side is open), and the DETECTED "
    "LANDMARKS (what is recognised, which side, how far). Then pick the place "
    "that carries you toward the landmark the CURRENT segment names — not the "
    "roomiest, not the furthest.\n"
    "The places only cover what the camera sees. If none goes the right way, "
    "or none is offered, turn with step turns — step([2,2,2]) is 45° left, "
    "step([3]*6) a quarter turn right — and read the fresh view. An empty "
    "list means look elsewhere, not dead end.\n"
    "Close the last metre with single step([1])s, not another hop — the goal "
    "is a 3 m circle and precision decides it.\n"
    "Places marked as a gap are ways THROUGH between two things — usually "
    "what an instruction means by 'walk between'.\n"
    "Your robot HAS A BODY (a ~0.2m-radius cylinder). goto and step both go "
    "through the same per-primitive sensing and brakes (collision, closing "
    "corridor) — but do not lean on the brake: prefer a measured place over "
    "pushing at clutter.\n"
    "Keep the state honest as you look: surroundings = what is around you, "
    "updated whenever the scene changes.\n"
    "Set near_goal=true the MOMENT you believe the final goal is in sight or "
    "close — it lights the NEAR-STOP flag and arms arrival verification.\n"
    "Your first STOP request is checked by a verifier; if it is withheld, "
    "look again, then request STOP a second time to execute it."
)

# everything before the step()-specific pacing block, shared by both surfaces
_SHARED_HEAD = SOLO_ADDENDUM.split("Before a move you may fill expect")[0]

# The lean step face (slamstep on the mini executor, user 2026-08-18: step +
# get_map + get_pose, no recall / veer / look_around, guards and the arrival
# bell off). Same discipline as SOLO_ADDENDUM minus every tool it does not
# hold and every mechanism that is switched off — the harness must not
# describe a veer it will not perform or an arrival check it will not run.
LEAN_STEP_ADDENDUM = (
    "\n\n--- HARNESS NOTES ---\n"
    "A [CURRENT STATE] header arrives with every turn — it is your shared "
    "memory. The instruction is parsed into route SEGMENTS (each a landmark "
    "plus a stretch of walk) and one TERMINATION condition (the point where "
    "you STOP — shown at the top; it is what the verifier checks at the "
    "end). Update the state via the optional fields on your action calls: "
    "progress_note / advance_subgoal / landmark / dead_end (one short phrase "
    "each). Set advance_subgoal=true each time you complete one segment — "
    "YOU decide which segment you are on; nothing decides it for you.\n"
    "Older turns are cleared from your context automatically: you see your "
    "last few views and the state header{footer} — your CURRENT thought "
    "belongs in progress_note/expect, not in prose you hope to re-read "
    "later.\n"
    "get_map and get_pose cost no steps and are never pushed to you: pull "
    "the map whenever you need to check where you have been, whether you "
    "are circling, or which way is still unexplored.\n"
    "Before a move you may fill expect (where it should land you). Pace by "
    "distance: nothing important nearby → 4-6 forward steps per call; a "
    "landmark at medium distance → 2-3 steps; final approach or a turn "
    "coming → single steps. Never more than 6 forwards in one call — the "
    "harness truncates the rest.\n"
    "Your robot HAS A BODY (a ~0.2m-radius cylinder): it needs clearance on "
    "both sides and cannot squeeze past furniture. If an obstacle sits in "
    "the CENTER of your view a few steps ahead, turn a notch (one 15° turn) "
    "around it BEFORE pushing forward — walking straight at it will jam "
    "you. If a forward step is blocked, the result says so: look at the "
    "view, turn toward the open side yourself, and step again.\n"
    "Keep the state honest as you look: surroundings = what is around you, "
    "updated whenever the scene changes.\n"
    "Set near_goal=true the MOMENT you believe the final goal is in sight "
    "or close — it marks NEAR-STOP in your state.\n"
    "Your first STOP request is checked by a verifier; if it is withheld, "
    "look again, then request STOP a second time to execute it."
)


def addendum_for(tool_names: set[str], *, instruction_footer: bool = False) -> str:
    """Pick the addendum that matches the action space actually on offer.

    Self-selecting rather than configured: a prompt that describes step() to a
    model holding goto() is a silent, expensive lie, and the tool list is the
    ground truth about which surface is live."""
    if "goto" in tool_names:
        return _SHARED_HEAD + DWP_ADDENDUM
    if "recall" not in tool_names and "veer" not in tool_names:
        return LEAN_STEP_ADDENDUM.format(   # slamstep: step + get_map + get_pose
            footer=(", with your full instruction restated under it"
                    if instruction_footer else ""))
    return SOLO_ADDENDUM


def run_solo(
    wrapper: HarnessedToolset,
    *,
    briefing: str,
    task: str,
    model_name: str,
    model_knobs: dict[str, Any],
    turn_cap: int,
    output_path: Path,
    event_hook: Callable[[str, dict], None],
    opening_parts: list | None = None,
) -> tuple[dict[str, Any], Any]:
    """Run the whole episode as one harnessed loop. Returns (exit_info, agent)."""
    from nav_agent import NavAgent  # harnesses/mini on sys.path via mini_swe

    from eharness.compactor import SoloModel

    env = SubgoalEnvironment(wrapper)   # subgoal_over never fires in solo
    model = SoloModel(
        harness=wrapper,
        emit=event_hook,
        model_name=model_name,
        tools=wrapper.tool_schemas(),
        image_window=model_knobs.get("image_window", 6),
        compact_at=model_knobs.get("compact_at", 12000),
        tail_msgs=model_knobs.get("tail_msgs", 24),
        reattach_k=model_knobs.get("reattach_k", 3),
        # §12 typed context (the default; legacy_l2 is the ablation arm)
        history_frames=model_knobs.get("history_frames", 12),
        recent_turns=model_knobs.get("recent_turns", 6),
        legacy_l2=bool(model_knobs.get("legacy_l2", False)),
        instruction_footer=bool(model_knobs.get("instruction_footer", False)),
        manifest=bool(model_knobs.get("manifest", True)),
        set_cache_control=model_knobs.get("set_cache_control"),
        model_kwargs=model_knobs.get("model_kwargs", {"drop_params": True}),
        cost_tracking=model_knobs.get("cost_tracking", "default"),
    )
    agent = NavAgent(
        model, env,
        system_template=_jinja_raw(
            briefing + addendum_for(
                {t["name"] for t in wrapper.tool_schemas()},
                instruction_footer=bool(model_knobs.get("instruction_footer", False)))),
        instance_template=_jinja_raw(task),
        step_limit=turn_cap,
        cost_limit=model_knobs.get("cost_limit", 5.0),
        output_path=output_path,
        event_hook=event_hook,
        opening_parts=opening_parts,
    )
    exit_info: dict[str, Any] = {}
    try:
        exit_info = agent.run(task=task) or {}
    except Exception as exc:  # noqa: BLE001 — surfaced by the adapter
        exit_info = {"exit_status": f"solo_error:{type(exc).__name__}",
                     "error": repr(exc)}
        event_hook("driver_error", {"error": repr(exc), "where": "solo"})
    return exit_info, agent
