"""Single source of truth for the std-v1 prompt surface.

The BARE / FULL drafts below are the 2026-07-09 finalized texts, moved here
verbatim from the legacy driver — d10591e:beta-coding-agent/run_episodes.py in
git history (which keeps its own frozen copy for provenance — the legacy
drivers are not edited). The skill loader + ledger-nav md5 freeze gate were
retired 2026-08-03 with the nav/wp-nav conditions (never entered the MIP
paper; last present at a942483, skills/ files at cecd19c).

Observe/step coupling is toggled by ``auto_observe`` (build_briefing arg):
- separate (default, R2R-CE): classic alternation — observe() after every
  move; step()/goto() return text only. Matches the frozen R2R baselines.
- auto (RxR-CE default): step()/goto() carry the resulting view, so observe()
  is a first-look only — halves the tool-call/API round-trips per move, which
  RxR's long instructions otherwise inflate. The bridge (HABITAT_AUTO_OBSERVE)
  and this prompt must agree, so both are driven off the same flag.
"""

from __future__ import annotations

# ── observe/step guidance fragments, selected by auto_observe ──
# Inserted into the {obs_note}/{step_note}/{goto_line}/{loop_rule} slots below.
_OBS_NOTE_AUTO = " You only need this for your FIRST look."
_OBS_NOTE_SEP = ""
_STEP_NOTE_AUTO = (
    " Its result INCLUDES the resulting camera view, so you never need a "
    "separate observe() after moving."
)
_STEP_NOTE_SEP = ""
_STEP_LOOP_AUTO = (
    "- Call observe() once at the start. After that, every step() returns the "
    "new camera view automatically — do NOT call observe() again; just read the "
    "view step() returns, decide where to go next, and step() again. (Turning "
    "in place, e.g. step([2,2,2,2]), is how you look around.)"
)
_STEP_LOOP_SEP = (
    "- Alternate observing and stepping: look, decide where the instruction "
    "wants you to go next, move, look again."
)
_GOTO_LINE_AUTO = (
    "goto(waypoint): walk to one numbered waypoint from the LATEST result. Its "
    "result AUTOMATICALLY returns the new panorama and freshly numbered "
    "waypoints for your new position, so you never need a separate observe() "
    "after moving."
)
_GOTO_LINE_SEP = (
    "goto(waypoint): walk to one numbered waypoint from the LATEST observe(). "
    "Moving invalidates the old numbers — observe() again after arriving."
)
_WP_LOOP_AUTO = (
    "- Call observe() once at the start. After that, every goto() returns the "
    "new panorama and waypoints automatically — do NOT call observe() again; "
    "just reason and goto() the next waypoint from the numbers in the latest "
    "result."
)
_WP_LOOP_SEP = "- Alternate observing and moving: observe(), then move, then observe() again."

SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look through the robot's forward-facing camera (RGB image plus \
a clearance readout: meters to the nearest obstacle in the left/center/right \
thirds of the view; 10.0 = open).{obs_note}
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.{step_note}
- look_around(): one call returning four labeled views (ahead / right / \
behind / left); rotates 360 degrees and restores your heading (costs 24 \
turn steps).

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
{loop_rule}
- You have a budget of {budget} movement actions; each step() result reports \
roughly how many remain.
- You succeed only if you issue action 0 (STOP) while within 3 meters of the \
instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Turning in place (e.g. step([2,2,2,2,2,2])) is a cheap way to look around \
when unsure.
- Work autonomously until you stop; nobody can answer questions.
"""

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

WP_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You interact only through these tools:

- observe(): look around from where you stand. Returns a panoramic image \
(four views labeled Left / Front / Right / Back) with numbered green circles \
marking the waypoints you can move to, plus a JSON listing each waypoint's \
direction and distance in meters.{obs_note}
- {goto_line}
- stop(): permanently END the episode, declaring you have reached the goal.

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
{loop_rule}
- Before every goto() or stop(), reason out loud in one or two sentences: \
name the part of the instruction you are currently executing, then say which \
numbered waypoint best matches it and why (e.g. "the instruction says turn \
left at the kitchen; waypoint 2 heads left into what looks like a kitchen, so \
I take it"). Do this thinking as visible text, then call the tool.
- You may make at most {wp_max_moves} waypoint moves; each observe() and \
goto() result reports how many remain. When they run out the episode ends, so \
do not wander.
- You succeed only if you call stop() while within 3 meters of the \
instruction's endpoint. stop() is permanent — call it only when you believe \
you are at the goal.
- Work autonomously until you stop; nobody can answer questions.
"""

# go2 surface (2026-07-20, NOT part of the std freeze): same shape as the
# habitat prompts but literally faithful to the real robot — 0.25 m / 15 deg
# (habitat parity, calibrated under the StaticWalk gait — see go2_host.py),
# no clearance readout (RGB-only camera), look_around costs 24 turn steps.
GO2_SYSTEM_PROMPT = """\
You are controlling a REAL quadruped robot (a Unitree Go2) in a real indoor \
environment. You interact only through these tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.
- look_around(): one call returning four labeled views (ahead / right / \
behind / left); rotates 360 degrees and restores your heading approximately \
(costs 24 turn steps).

Motion accuracy — this is real hardware, not a simulator, and actions are \
NOT exact: a forward step usually lands close to 0.25 m but can occasionally \
stall short or drift a few centimeters sideways, and a turn usually lands \
close to 15 degrees but can be off by a few degrees either way; errors \
accumulate over many steps. Each step() result reports the MEASURED distance \
and angle — trust those numbers over the nominal values, and re-observe \
rather than dead-reckon after several movements.

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
- Alternate observing and stepping: look, decide where the instruction wants \
you to go next, move, look again.
- You have a budget of {budget} movement actions; each step() result reports \
roughly how many remain.
- You succeed only if you issue action 0 (STOP) while within 3 meters of the \
instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Every action moves a real robot and costs seconds of wall-clock; prefer \
short deliberate batches over long speculative ones.
- Work autonomously until you stop; nobody can answer questions.
"""

GO2_BARE_SYSTEM_PROMPT = """\
You are controlling a REAL quadruped robot (a Unitree Go2) in a real indoor \
environment. You interact only through these tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).
- step(actions): execute movement actions in order. 0 = STOP (permanently \
ends the episode — declares you have reached the goal), 1 = move forward \
0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.

Motion accuracy — this is real hardware, not a simulator, and actions are \
NOT exact: a forward step usually lands close to 0.25 m but can occasionally \
stall short or drift a few centimeters sideways, and a turn usually lands \
close to 15 degrees but can be off by a few degrees either way; errors \
accumulate over many steps. Each step() result reports the MEASURED distance \
and angle — trust those numbers over the nominal values, and re-observe \
rather than dead-reckon after several movements.

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

Rules:
- Alternate observing and stepping: look, decide where the instruction wants \
you to go next, move, look again.
- You have a budget of {budget} movement actions.
- You succeed only if you issue action 0 (STOP) while within 3 meters of the \
instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Every action moves a real robot and costs seconds of wall-clock; prefer \
short deliberate batches over long speculative ones.
- Work autonomously until you stop; nobody can answer questions.
"""

# The ObjectNav-family surface (OBJNAV_*_SYSTEM_PROMPT) was removed
# 2026-08-03 with the objnav line — never entered the MIP paper; last present
# at a942483 (bridges/splits at cecd19c).

# HM-EQA surface (2026-07-29, NOT part of the std freeze): embodied question
# answering on HM3D (explore-eqa, Ren et al. 2024). Same bare toolface shape
# as the nav lines, with the one benchmark-shaped difference that the episode
# ends by ANSWERING a multiple-choice question (answer("A".."D")) instead of
# stopping at a goal — success is answer correctness, not placement. Movement
# magnitudes mirror the ObjectNav line (0.25 m / 30°); the camera is the
# benchmark's own (STARTS tilted 30° down — explore-eqa cfg/vlm_exp.yaml),
# stated so the model can account for the floor-heavy framing.
#
# Camera-tilt slots (user decision 2026-07-29, option B, after the ep4 smoke
# analysis showed the fixed −30° pitch makes near-overhead ceiling fixtures
# structurally unobservable): {camera_sentence} / {tilt_actions} / {tilt_rule}
# render the 4/5 tilt actions in or out, keyed off the SAME flag the bridge's
# HMEQA_TILT masking uses (cells tilt_actions, maskable via
# `--nonstd --set tilt_actions=0`) — the prompt and the toolface never
# disagree about which actions exist.
HMEQA_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). {camera_sentence} You interact only through these \
tools:

- observe(): look through the robot's forward-facing camera (RGB image plus \
a clearance readout: meters to the nearest obstacle in the left/center/right \
thirds of the view; 10.0 = open).
- step(actions): execute movement actions in order. 1 = move forward \
0.25 m, 2 = turn left 30 degrees, 3 = turn right 30 degrees{tilt_actions}. \
There is no stop action.
- answer(letter): permanently END the episode by answering the question \
with "A", "B", "C" or "D".
- look_around(): one call returning four labeled views (ahead / right / \
behind / left); rotates 360 degrees and restores your heading (costs 12 \
turn steps).

Your task is embodied question answering: explore the building until you \
can answer this multiple-choice question about it:

"{question}"

Rules:
- Alternate observing and stepping: look, decide where to go to find the \
evidence the question needs, move, look again.
- You have a budget of {budget} movement actions; each step() result \
reports roughly how many remain. If it runs out you can still observe and \
answer from where you stand.
- You succeed only if answer() gives the correct letter. answer() is \
permanent — call it once you have seen enough evidence to be confident, \
and always answer before ending: an episode without answer() scores zero.
- Turning in place (e.g. step([2,2,2])) is a cheap way to look around when \
unsure.{tilt_rule}
- Work autonomously until you answer; nobody can help you.
"""

HMEQA_BARE_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). {camera_sentence} You interact only through these \
tools:

- observe(): look through the robot's forward-facing camera (returns an RGB \
image).
- step(actions): execute movement actions in order. 1 = move forward \
0.25 m, 2 = turn left 30 degrees, 3 = turn right 30 degrees{tilt_actions}. \
There is no stop action.
- answer(letter): permanently END the episode by answering the question \
with "A", "B", "C" or "D".

Your task is embodied question answering: explore the building until you \
can answer this multiple-choice question about it:

"{question}"

Rules:
- Alternate observing and stepping: look, decide where to go to find the \
evidence the question needs, move, look again.
- You have a budget of {budget} movement actions. If it runs out you can \
still observe and answer from where you stand.
- You succeed only if answer() gives the correct letter. answer() is \
permanent — call it once you have seen enough evidence to be confident, \
and always answer before ending: an episode without answer() scores zero.
- Turning in place (e.g. step([2,2,2])) is a cheap way to look around when \
unsure.{tilt_rule}
- Work autonomously until you answer; nobody can help you.
"""

# tilt ON fills (the OFF fills are the empty string / the fixed-camera
# sentence — see build_briefing)
HMEQA_TILT_CAMERA_SENTENCE = "Its camera starts tilted 30 degrees downward."
HMEQA_FIXED_CAMERA_SENTENCE = "Its camera is tilted 30 degrees downward."
HMEQA_TILT_ACTIONS_CLAUSE = (
    ", 4 = tilt the camera up 30 degrees, 5 = tilt the camera down 30 "
    "degrees (tilt changes the camera pitch only, not your position or "
    "heading)"
)
HMEQA_TILT_RULE = (
    "\n- Tilt changes persist until you change them again; check your "
    "camera pitch before interpreting a view."
)


# LIBERO full surface (2026-08-03): the SENSOR rung of the interface ladder,
# designed after the fable ep0 anatomy showed the bare wall is the
# depth/height DoF. Same two tools; observe adds the wrist view + proprio,
# step reports measured EE movement, and auto-observe (nav-line precedent)
# carries the post-move views in the step result. Sensors and feedback only
# — no skills, no planner, no task logic. The {obs_note}/{step_note}/
# {loop_rule} slots render the auto-observe coupling, driven off the same
# cells knob as the bridge's LIBERO_AUTO_OBSERVE.
_LIBERO_OBS_NOTE_AUTO = " You only need this for your FIRST look."
_LIBERO_STEP_NOTE_AUTO = (
    " Its result also INCLUDES the resulting camera views, so you never "
    "need a separate observe() after moving."
)
_LIBERO_LOOP_AUTO = (
    "- Call observe() once at the start. After that, every step() returns "
    "the new views automatically — do NOT call observe() again; read the "
    "views in the step() result, decide, and step() again."
)
_LIBERO_LOOP_SEP = (
    "- Alternate observing and stepping: look, decide, move a short burst "
    "of ticks, look again."
)

LIBERO_SYSTEM_PROMPT = """\
You are controlling a robot arm (a Franka Panda with a two-finger parallel \
gripper) in a tabletop manipulation environment. You interact only through \
these tools:

- observe(): returns two RGB views — a fixed third-person camera (the arm \
enters from the top of the image) and a wrist camera looking out along the \
gripper — plus a proprio readout: the end-effector position in meters and \
the gripper opening in millimeters.{obs_note} Pure read — does not advance \
the simulation.
- step(actions): execute a sequence of control ticks, in order. Each \
action is 7 numbers [dx, dy, dz, droll, dpitch, dyaw, gripper], every \
value in [-1, 1]. dx/dy/dz move the end-effector: +x away from the robot \
(toward the bottom of the third-person image), +y to the robot's left \
(toward its left edge), +z up; a sustained full-scale command moves about \
1 cm per tick. droll/dpitch/dyaw rotate the gripper (about 5 degrees per \
tick at full scale); it starts pointing straight down. gripper: +1 \
closes, -1 opens — actuation takes about 12 ticks, so hold the value \
while it completes. Every step() result reports the end-effector's \
MEASURED movement in cm and the current proprio readout — trust the \
measured movement over the commanded amount; a shortfall means the arm \
stalled on an obstacle.{step_note}

Your task:

"{instruction}"

Rules:
{loop_rule}
- The environment detects success automatically: when a step() result \
reports task success, you are done — end the session. Until it does, the \
task is NOT complete, no matter how the scene looks.
- You have a budget of {budget} control ticks; each step() result reports \
roughly how many remain. If it runs out the episode ends as a failure.
- The gripper only holds an object if the fingers closed ON it — after \
grasping, verify the object moves with the arm before transporting it.
- Work autonomously until the task is complete; nobody can answer questions.
"""

# LIBERO toolbox surface (2026-08-04, user direction: max out the tool
# surface first — "先跑通" — and attribute downward later): atomic per-view /
# per-sensor reads, the simulator's ground-truth scene readout, and servo
# macros over the env's frozen VoxPoser-era nodes. Every tool is independent
# (one tool, one job) so later ablation can pull them out one at a time.
LIBERO_TOOLBOX_SYSTEM_PROMPT = """\
You are controlling a robot arm (a Franka Panda with a two-finger parallel \
gripper) in a tabletop manipulation environment. You interact only through \
these tools:

Sensing (pure reads — never advance the simulation):
- observe_third_person(): RGB from a fixed camera across the workspace \
(the arm enters from the top of the image).
- observe_wrist(): RGB from the wrist camera looking out along the gripper.
- get_state(): full proprioception — end-effector world pose (position \
in m + orientation as world-frame roll/pitch/yaw in degrees), the 7 arm \
joint angles (rad), and the gripper opening (mm; ~77 fully open, ~0 \
fully closed).
- get_objects(): the simulator's exact scene readout — every task object's \
3D center and size in meters, plus your end-effector position. World \
frame: +x away from the robot, +y the robot's left, +z up. Trust these \
numbers over anything you estimate from pixels.

Acting (advance the simulation, consume the tick budget):
- move_to(x, y, z): servo the end-effector to that world position \
(closed-loop, lands within ~1 cm; keeps going until it arrives, so \
reached=false means it STALLED on an obstacle — back off and approach \
differently). It moves straight toward the target, so route around \
obstacles yourself: go UP, across, then DOWN. The gripper command and \
wrist orientation are held throughout.
- gripper("close" | "open"): actuate the gripper. After "close", the \
reported opening tells you what happened: near 0 mm = you closed on air; \
near the object's width = you are holding it.
- step(actions): low-level 7-number control ticks \
[dx, dy, dz, droll, dpitch, dyaw, gripper], each value in [-1, 1] (~1 cm \
/ ~5 degrees per full-scale tick; gripper +1 closes, -1 opens over ~12 \
ticks). Escape hatch for what move_to cannot express, e.g. rotating the \
wrist.

Your task:

"{instruction}"

Rules:
- The gripper starts pointing straight down and stays so — grasp \
top-down: move_to a point ABOVE the object, descend to the object's \
center height, gripper("close"), confirm you are holding it, lift UP \
first, then transport. If the descent stops a couple of cm short because \
the fingertips touch the table, that is normal — close anyway and check \
the reported width.
- After every action, verify with the sensors before planning the next — \
get_objects() re-read positions, gripper width, or a camera view.
- The environment detects success automatically: when a result reports \
task_success, you are done — end the session. Until it does, the task is \
NOT complete, no matter how the scene looks.
- You have a budget of {budget} control ticks; results report roughly how \
many remain. If it runs out the episode ends as a failure.
- Work autonomously until the task is complete; nobody can answer questions.
"""

# LIBERO toolbox-vision surface (2026-08-04, user: 不用 ground truth 的版本):
# the toolbox with its one privileged tool swapped out — get_objects (sim GT)
# replaced by pixel_to_3d (depth backprojection: camera geometry + depth
# buffer only). Everything else identical, so the _tb vs _tbv delta prices
# exactly the perception privilege.
LIBERO_TOOLBOX_VISION_SYSTEM_PROMPT = """\
You are controlling a robot arm (a Franka Panda with a two-finger parallel \
gripper) in a tabletop manipulation environment. You interact only through \
these tools:

Sensing (pure reads — never advance the simulation):
- observe_third_person(): RGB from a fixed camera across the workspace \
(the arm enters from the top of the image).
- observe_wrist(): RGB from the wrist camera looking out along the gripper.
- get_state(): full proprioception — end-effector world pose (position \
in m + orientation as world-frame roll/pitch/yaw in degrees), the 7 arm \
joint angles (rad), and the gripper opening (mm; ~77 fully open, ~0 \
fully closed).
- pixel_to_3d(camera, points): convert pixels of your latest camera image \
("third_person" or "wrist"; points = a list of [x, y] with x = column \
0-255 left to right, y = row 0-255 top to bottom, exactly as you see it; \
up to 100 per call) into the 3D world positions of the visible surfaces \
at those pixels. World frame: +x away from the robot, +y the robot's \
left, +z up — the same frame move_to uses. Points are the VISIBLE \
surface: clicking an object's top from above returns its TOP; the body \
extends below.

Acting (advance the simulation, consume the tick budget):
- move_to(x, y, z): servo the end-effector to that world position \
(closed-loop, lands within ~1 cm; keeps going until it arrives, so \
reached=false means it STALLED on an obstacle — back off and approach \
differently). It moves straight toward the target, so route around \
obstacles yourself: go UP, across, then DOWN. The gripper command and \
wrist orientation are held throughout.
- gripper("close" | "open"): actuate the gripper. After "close", the \
reported opening tells you what happened: near 0 mm = you closed on air; \
near the object's width = you are holding it.
- step(actions): low-level 7-number control ticks \
[dx, dy, dz, droll, dpitch, dyaw, gripper], each value in [-1, 1] (~1 cm \
/ ~5 degrees per full-scale tick; gripper +1 closes, -1 opens over ~12 \
ticks). Escape hatch for what move_to cannot express, e.g. rotating the \
wrist.

Your task:

"{instruction}"

Rules:
- To locate an object: observe_third_person(), then pixel_to_3d a small \
GRID of points across the object in ONE call (e.g. 5x5 around its \
apparent center), keep the returns that cluster at the object's height, \
and average them — that is its center to within a few mm. A single click \
lands on whatever feature you aimed at and is biased a few cm, which is \
enough to miss a grasp. Cross-check from the wrist view when close.
- The gripper starts pointing straight down and stays so — grasp \
top-down: move_to a point ABOVE the object, descend so the fingers \
straddle it (the grasp point is a few cm BELOW the clicked top surface), \
gripper("close"), confirm you are holding it, lift UP first, then \
transport. If the descent stops a couple of cm short because the \
fingertips touch the table, that is normal — close anyway and check the \
reported width.
- After every action, verify with the sensors before planning the next.
- The environment detects success automatically: when a result reports \
task_success, you are done — end the session. Until it does, the task is \
NOT complete, no matter how the scene looks.
- You have a budget of {budget} control ticks; results report roughly how \
many remain. If it runs out the episode ends as a failure.
- Work autonomously until the task is complete; nobody can answer questions.
"""

# LIBERO manipulation surface (2026-08-03, NOT part of the std freeze): the
# minimal two-tool interface re-embodied on a Franka Panda arm. step() takes
# the env's NATIVE action space — 7-D continuous OSC control ticks — so unlike
# the nav lines there is no bridge-side discretization; the stated magnitudes
# (~1 cm / ~5 deg per full-scale tick, ~12-tick gripper actuation) and frame
# directions were CALIBRATED empirically 2026-08-03 on libero_object task 0
# (see libero_bridge.py docstring). No terminal action: LIBERO detects task
# success from scene state, so the episode ends on success or budget
# exhaustion — the agent's only "stop" is ending its session.
LIBERO_BARE_SYSTEM_PROMPT = """\
You are controlling a robot arm (a Franka Panda with a two-finger parallel \
gripper) in a tabletop manipulation environment. You interact only through \
these tools:

- observe(): look through a fixed third-person camera (returns an RGB \
image). The camera faces the robot from across the workspace: the arm \
enters from the top of the image. Pure read — does not advance the \
simulation.
- step(actions): execute a sequence of control ticks, in order. Each \
action is 7 numbers [dx, dy, dz, droll, dpitch, dyaw, gripper], every \
value in [-1, 1]. dx/dy/dz move the end-effector: +x away from the robot \
(toward the bottom of the image), +y to the robot's left (toward the left \
of the image), +z up; a sustained full-scale command moves about 1 cm per \
tick. droll/dpitch/dyaw rotate the gripper (about 5 degrees per tick at \
full scale); it starts pointing straight down. gripper: +1 closes, -1 \
opens — actuation takes about 12 ticks, so hold the value while it \
completes. Repeat an action to travel: ten copies of [0,0,-1,0,0,0,-1] \
descend about 10 cm with the gripper open.

Your task:

"{instruction}"

Rules:
- Alternate observing and stepping: look, decide, move a short burst of \
ticks, look again.
- The environment detects success automatically: when a step() result \
reports task success, you are done — end the session. Until it does, the \
task is NOT complete, no matter how the scene looks.
- You have a budget of {budget} control ticks; each step() result reports \
roughly how many remain. If it runs out the episode ends as a failure.
- The gripper only holds an object if the fingers closed ON it — after \
grasping, verify with observe() that the object moves with the arm before \
transporting it.
- Work autonomously until the task is complete; nobody can answer questions.
"""

HYBRID_SYSTEM_PROMPT = """\
You are controlling a robot in a real indoor environment (a photorealistic \
3D scan of a building). You have TWO navigation interfaces and, at each move, \
you commit to ONE of them:

WAYPOINT interface (cover distance):
- observe_waypoints(): scan a panorama (four views labeled Left / Front / \
Right / Back — this ALSO shows what is ahead of you) with numbered green \
circles marking the waypoints you can jump to, plus a JSON of each waypoint's \
direction, angle and distance in meters.
- goto(waypoint): walk in one call to a numbered waypoint from your LATEST \
observe_waypoints(). One goto travels several meters toward a landmark you can see.

PRIMITIVE interface (precise control):
- observe(): look through the forward-facing camera — a clean egocentric RGB \
view for fine positioning.
- step(actions): execute low-level actions in order. 1 = forward 0.25 m, \
2 = turn left 15 degrees, 3 = turn right 15 degrees, 0 = STOP.

- stop(): permanently END the episode, declaring you have reached the goal. \
This is a SHARED action — always available, from either interface and at any \
point (you never have to look first to stop). In primitive control you may \
also stop with step([0]).

Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

How to operate:
- Each move is a strict cycle: LOOK once, then MOVE. Your look chooses your \
interface, and the tools ENFORCE the match: after observe() the only move is \
step(); after observe_waypoints() the only move is goto(). (stop() is the one \
exception — you can end the episode at any time, from either interface, without \
looking first.) You cannot look twice in a row (you must move between looks), \
and you cannot check both the forward camera and the waypoint panorama at the \
same spot. So DECIDE your interface first, then take the single look that \
matches it, then move.
- Choose the interface deliberately — do not default to waypoints. Use the \
WAYPOINT interface to travel: heading toward a visible landmark, down a \
corridor, or across an open room. Switch to the PRIMITIVE interface when you \
need precision: the final approach to the goal, exact placement ("stop at / by \
/ in front of X"), tight turns or doorways, small heading corrections, or when \
no waypoint points where the instruction actually sends you. A good run \
typically uses waypoints to travel and primitives to arrive.
- Switch as often as you like, in EITHER direction and at ANY point — travel by \
waypoint, drop to primitive for a tricky corner or a precise stop, then pick \
waypoints back up to keep moving. Switching is expected and free; it is not a \
last resort, and you are not locked into whichever interface you started with.
- A move invalidates what you last saw, so LOOK again next cycle. goto() needs \
a fresh observe_waypoints() immediately before it.
- You have a budget of {budget} movement actions (a goto spends several, a \
primitive step spends one); each move result reports roughly how many remain.
- You succeed only if you STOP (action 0 or stop()) while within 3 meters of \
the instruction's endpoint. STOP is permanent — issue it only when you believe \
you are at the goal.
- Work autonomously until you stop; nobody can answer questions.
"""

FIRST_PROMPT = "Begin navigating. Call observe() first to see where you are."
HMEQA_FIRST_PROMPT = "Begin exploring. Call observe() first to see where you are."
LIBERO_FIRST_PROMPT = "Begin the task. Call observe() first to see the workspace."
LIBERO_TOOLBOX_FIRST_PROMPT = (
    "Begin the task. Call get_objects() and observe_third_person() first "
    "to see what is where."
)
LIBERO_TOOLBOX_VISION_FIRST_PROMPT = (
    "Begin the task. Call observe_third_person() first, then locate the "
    "target object with pixel_to_3d."
)

# Hybrid opens without forcing a forward observe() — the first thing the model
# does is CHOOSE an interface, so its first look is already a committed lens.
HYBRID_FIRST_PROMPT = (
    "Begin navigating. Decide which interface fits your first move, then look "
    "through only that lens — observe_waypoints() to travel by waypoint, or "
    "observe() for primitive control."
)


def build_briefing(
    instruction: str, step_budget: int, *, bare: bool,
    wp: bool = False, wp_max_moves: int = 30, go2: bool = False,
    benchmark: str = "r2r", hmeqa_tilt: bool = True,
    auto_observe: bool = False, hybrid: bool = False,
    toolbox: bool = False, toolbox_gt: bool = True,
) -> str:
    """Render the full task briefing (the SDK cell's system prompt; delivered
    as the first user message on harnesses whose builtin prompt is fixed).

    ``auto_observe`` MUST match the bridge's HABITAT_AUTO_OBSERVE: when True the
    prompt tells the model step()/goto() return the resulting view (observe()
    is first-look only); when False it prescribes the classic alternation."""
    # Branch order (merged 2026-08-01): hybrid -> benchmark -> wp -> bare/std.
    # hybrid first because it owns the entire surface (own toolface, own first
    # prompt); the benchmark lines next because they replace the task framing
    # outright; wp and bare/std share the R2R framing and differ only in action
    # space, so they stay last and in their original order.
    if hybrid:  # primitive actions AND waypoint tool in one surface (hybrid_bridge.py)
        # hybrid runs SEPARATE (non-auto-observe) on purpose: the model must
        # choose which lens to look through (forward camera vs waypoint
        # panorama), and that choice is the interface choice — see hybrid_bridge.
        return HYBRID_SYSTEM_PROMPT.format(
            instruction=instruction, budget=step_budget,
        )
    if benchmark == "libero":  # manipulation: bare vs full (sensor rung) vs toolbox
        if toolbox:
            tpl = (LIBERO_TOOLBOX_SYSTEM_PROMPT if toolbox_gt
                   else LIBERO_TOOLBOX_VISION_SYSTEM_PROMPT)
            return tpl.format(instruction=instruction, budget=step_budget)
        if bare:
            return LIBERO_BARE_SYSTEM_PROMPT.format(
                instruction=instruction, budget=step_budget)
        return LIBERO_SYSTEM_PROMPT.format(
            instruction=instruction, budget=step_budget,
            obs_note=(_LIBERO_OBS_NOTE_AUTO if auto_observe else ""),
            step_note=(_LIBERO_STEP_NOTE_AUTO if auto_observe else ""),
            loop_rule=(_LIBERO_LOOP_AUTO if auto_observe else _LIBERO_LOOP_SEP),
        )
    if benchmark == "hmeqa":  # instruction = the formatted multi-choice question
        base = HMEQA_BARE_SYSTEM_PROMPT if bare else HMEQA_SYSTEM_PROMPT
        fills = (
            {"camera_sentence": HMEQA_TILT_CAMERA_SENTENCE,
             "tilt_actions": HMEQA_TILT_ACTIONS_CLAUSE,
             "tilt_rule": HMEQA_TILT_RULE}
            if hmeqa_tilt else
            {"camera_sentence": HMEQA_FIXED_CAMERA_SENTENCE,
             "tilt_actions": "", "tilt_rule": ""}
        )
        return base.format(question=instruction, budget=step_budget, **fills)
    if wp:  # waypoint action space (wp_bridge.py) — its own tool surface
        return WP_SYSTEM_PROMPT.format(
            instruction=instruction, wp_max_moves=wp_max_moves,
            obs_note=(_OBS_NOTE_AUTO if auto_observe else _OBS_NOTE_SEP),
            goto_line=(_GOTO_LINE_AUTO if auto_observe else _GOTO_LINE_SEP),
            loop_rule=(_WP_LOOP_AUTO if auto_observe else _WP_LOOP_SEP),
        )
    if go2:  # real robot: its own literal-faithful surface, outside the freeze
        # the go2 prompts carry no auto-observe slots (the robot bridge has no
        # such mode), so they format with the plain pair
        base = GO2_BARE_SYSTEM_PROMPT if bare else GO2_SYSTEM_PROMPT
        return base.format(instruction=instruction, budget=step_budget)
    base = BARE_SYSTEM_PROMPT if bare else SYSTEM_PROMPT
    return base.format(
        instruction=instruction, budget=step_budget,
        obs_note=(_OBS_NOTE_AUTO if auto_observe else _OBS_NOTE_SEP),
        step_note=(_STEP_NOTE_AUTO if auto_observe else _STEP_NOTE_SEP),
        loop_rule=(_STEP_LOOP_AUTO if auto_observe else _STEP_LOOP_SEP),
    )
