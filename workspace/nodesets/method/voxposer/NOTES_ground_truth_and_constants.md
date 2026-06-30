# VoxPoser-on-LIBERO — ground truth & fixed constants

Reference note for the `voxposer_libero_decomposed` port. Lists (A) the
**privileged ground-truth** the port consumes instead of perceiving, and (B) the
**hardcoded numbers** baked into the planner/env — the dials that, when wrong,
silently break grasping (the table-height bug being the canonical example).

Faithful to code as of 2026-06-29. Citations are `file:line`.

---

## A. Ground truth we rely on

VoxPoser here does **not** perceive — it is fed a privileged sim snapshot by
`env_libero__observe_objects` (`env/env_libero/__init__.py:890` ff). Everything
below is read straight out of MuJoCo / the BDDL, not from RGB-D or a VLM:

| GT input | Source | Note |
|---|---|---|
| `object_names` | active BDDL file (`_parse_bddl_objects`) | object identities, not detected |
| `object_pcs` | per-object **AABB** of MuJoCo geoms, sampled | `_GT_PC_DENSITY=200` pts/obj; +z normals only (occupancy, not real surface) |
| `scene_pc` | AABB sample over all named non-robot/gripper/hand bodies | collision proxy |
| `ee_pos` / `ee_quat` | MuJoCo EE site (`robot0_eef_site`…) | exact, not estimated |
| `gripper_open` | finger qpos sum `> 0.04` | |
| `bounds_min/max` | `_workspace_bounds(sim)` — fixed x/y, **z anchored to live robot base z** | see §B / table bug |
| robot base z | `robot0_base` body xpos (`_robot_base_z`, `:929`) | the per-scene support-surface anchor |

Implication: object **geometry is an axis-aligned box**, not a real point cloud —
fine for top-down pinches on isolated tabletop objects, weak for shaped objects
(wide bowls, the `libero_spatial` failure mode).

---

## B. Fixed / hardcoded numbers

### env side — `env/env_libero/__init__.py`

| Constant | Value | Meaning |
|---|---|---|
| `_GT_BOUNDS_X` | `(-0.7, 0.3)` | workspace x box (m, world) — fixed |
| `_GT_BOUNDS_Y` | `(-0.4, 0.5)` | workspace y box (m, world) — fixed |
| `_Z_BELOW_BASE` / `_Z_ABOVE_BASE` | `0.10` / `0.35` | z box **relative to robot base z** — see table bug |
| `_FALLBACK_BASE_Z` | `0.912` | base z if `robot0_base` unreadable (table frame) |
| `_GT_PC_DENSITY` | `200` | pts per object/body AABB sample |
| `_SUITE_MAX_STEPS` | `2500` (all suites) | per-episode env-step budget |
| `_STEP_POS_M` | `0.005` | max **5 mm** OSC advance per substep (→ ~1.25 mm/tick realised) |
| `_STEP_ROT_RAD` | `0.05` | max ~3° OSC rotation per substep |
| `_OUTPUT_MAX_POS` | `0.05` | robosuite scale: delta=1 → 5 cm goal shift |
| `_OUTPUT_MAX_ROT` | `0.5` | robosuite scale: delta=1 → 0.5 rad goal shift |
| `pos_tol_m` / `rot_tol_rad` | `0.01` / `0.10` (config defaults) | convergence tol; graph overrides pos to `0.005` |
| `_DEFAULTS` | resolution `256`, num_steps_wait `10`, seed `42` | render / reset |

**Gripper ACTION convention** (`:1238`): `libero_grip = 1.0 if grip_in>0.5 else -1.0`
→ **`+1` = CLOSE, `-1` = OPEN** (verified live). The `-1=close` comments
elsewhere in this file are **stale** — do not trust them.

### method side — `method/voxposer/_runtime.py`

| Constant | Value | Meaning |
|---|---|---|
| `_GRASP_APPROACH_M` | `0.08` | top-down pre-grasp height above grasp point; for grasp subtasks the overshooting LMP path is **discarded** and replaced by a vertical descent (`:40`) |
| `_LIFT_MARGIN_BELOW_CEIL` | `0.05` | lift target margin below workspace ceiling z |
| LMP `temperature` | `0.0` | all 7 LMPs |
| planner `stop_threshold` / `ee_local_radius` | `0.001` / `0.15` | composer/path knobs |
| `_grip_state` init | `0.0` (open) | downstream convention **`1.0`=closed / `0.0`=open**; gripper scheduled deterministically from the subtask string, NOT sampled from the gripper_map |

### graph — `graphs/vla/unverified/voxposer_libero_decomposed.json`

| Field | Value | Meaning |
|---|---|---|
| `step_pose.max_steps` | `400` | OSC ticks per waypoint (was 200; long pre-grasp move needs >200) |
| `voxposer__init.controller_num_samples` | `10000` | planner sampling |
| `voxposer__init.controller_horizon` | `1` | |
| `iterIn.step_budget` | `8` | outer-loop waypoint budget |

---

## The table bug (canonical example — diagnosed 2026-06-28)

Before the fix the z box was **one hardcoded constant `[0.85, 1.25]`**, calibrated
for the table-mounted `libero_spatial` (robot base z ≈ 0.912). `libero_object` is
**floor-mounted** (base z ≈ 0.0, objects at z ≈ 0.0–0.15), so the entire voxel
workspace sat **~0.9 m above the objects** → every grasp target clamped to the
unreachable top → `libero_object` was **0/50 for 29 AAS iterations** (misread as
an OSC step-budget problem). Fix (`:944`): derive the z box **per scene** from the
live robot base z (`base_z − _Z_BELOW_BASE`, `base_z + _Z_ABOVE_BASE`), keep the
proven below/above offsets. Suites differ in world-frame z anchoring by ~0.9 m —
so any z constant is a per-suite assumption; never re-hardcode it.

Result: `libero_object` 0/50 → **44/50 (0.88)** at 50-ep suite scale.
