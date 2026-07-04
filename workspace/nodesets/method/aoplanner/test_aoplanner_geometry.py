"""Isolation test for the AO-Planner M2 geometry/overlay nodes.

project_waypoints is validated against a SYNTHETIC flat-floor with analytic
ground truth (no Habitat needed): camera at height 1.25 m, level, +X right /
+Y up / -Z forward, HFOV 90 (fx=fy=256, cx=cy=256, 512x512). Depth is built so
each below-horizon pixel lands on the floor plane; the recovered (angle,
distance) is checked against the closed-form values, and the left/right SIGN is
asserted (right pixel -> negative/CW yaw, left -> positive/CCW, with the default
bearing_sign=-1).

Run (agentcanvas env):
  cd /path/to/vlnworkspace
  PYTHONPATH=agentcanvas/backend python \
      workspace/nodesets/method/aoplanner/test_aoplanner_geometry.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "agentcanvas/backend"))
sys.path.insert(0, _REPO)
import workspace.nodesets.method.aoplanner as m

FX = FY = 256.0
CX = CY = 256.0
W = H = 512
CAM_H = 1.25
fails = []


def approx(a, b, tol=1e-2):
    return abs(a - b) <= tol


def make_floor_depth() -> np.ndarray:
    """Planar Z-depth so pixel (u,v>cy) lands on the floor (y_cam=-CAM_H)."""
    depth = np.zeros((H, W), dtype=np.float32)
    for v in range(H):
        if v > CY:
            depth[v, :] = CAM_H * FY / (v - CY)
    return depth


def check(name, cond):
    print(("  PASS" if cond else "  FAIL"), name)
    if not cond:
        fails.append(name)


def test_projection():
    print("[project_waypoints] synthetic flat-floor")
    depth = make_floor_depth()
    # pixels: center-bottom, right-bottom, left-bottom (all v=400), and a sky pixel (v=100)
    pts = [[256, 400], [400, 400], [112, 400], [256, 100]]
    res = m._project_pixels(pts, depth, FX, FY, CX, CY, view_heading_deg=0.0)
    by_idx = {r["idx"]: r for r in res}

    d_expect = CAM_H * FY / (400 - CY)  # 2.2222
    # center: angle 0, distance == d
    c = by_idx[0]
    check("center valid", c["valid"])
    check(f"center angle~0 (got {c['angle']:.4f})", approx(c["angle"], 0.0))
    check(
        f"center distance~{d_expect:.3f} (got {c['distance']:.3f})",
        approx(c["distance"], d_expect, 0.02),
    )

    x_r = (400 - CX) / FX * d_expect
    beta = math.atan2(x_r, d_expect)
    dist_r = math.hypot(x_r, d_expect)
    r = by_idx[1]
    check(f"right angle<0 (got {r['angle']:.4f})", r["angle"] < -0.1)
    check(f"right angle~{-beta:.4f}", approx(r["angle"], -beta))
    check(
        f"right distance~{dist_r:.3f} (got {r['distance']:.3f})",
        approx(r["distance"], dist_r, 0.02),
    )

    lft = by_idx[2]
    check(f"left angle>0 (got {lft['angle']:.4f})", lft["angle"] > 0.1)
    check(f"left angle~{beta:.4f}", approx(lft["angle"], beta))

    sky = by_idx[3]
    check("sky pixel (v<cy, depth 0) invalid", not sky["valid"])


def test_depth_decode():
    print("[_decode_depth_m] un-normalize (depth_scale) + resize to intrinsics res")
    from PIL import Image

    # 256x256 'mm' PNG encoding normalized 0.4 (env stores normalized*1000 as uint16 mm)
    raw = np.full((256, 256), 400, dtype=np.uint16)
    buf = io.BytesIO()
    Image.fromarray(raw, mode="I;16").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # legacy decode (no scale, no resize): 400/1000 = 0.4 "m" (the M6 bug — 10x too small)
    d_legacy = m._decode_depth_m(b64, True)
    check(
        f"legacy decode ~0.4 (the bug, got {d_legacy.mean():.3f})",
        approx(float(d_legacy.mean()), 0.4),
    )
    # fixed decode: *10 un-normalize -> 4.0 m, resized to 224x224 (RGB/intrinsics res)
    d_fix = m._decode_depth_m(b64, True, depth_scale=10.0, target_wh=(224, 224))
    check("fixed decode resized to 224x224", d_fix.shape == (224, 224))
    check(f"fixed decode ~4.0 m (got {d_fix.mean():.3f})", approx(float(d_fix.mean()), 4.0, 0.05))


def test_view_heading():
    print("[project_waypoints] view_heading composition")
    depth = make_floor_depth()
    res = m._project_pixels([[256, 400]], depth, FX, FY, CX, CY, view_heading_deg=90.0)
    a = res[0]["angle"]
    check(f"heading=90 center angle~pi/2 (got {a:.4f})", approx(a, math.pi / 2, 0.02))


def test_sample_waypoints():
    print("[sample_waypoints] grid in mask")
    from PIL import Image

    def _mask_b64(row0: int) -> str:
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[row0:, :] = 255
        buf = io.BytesIO()
        Image.fromarray(mask, mode="L").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    node = m.SampleWaypointsNode()
    node.config = {"grid_px": 50, "include_start": True}
    # bottom 150 rows navigable → 2 grid rows x 9 cols + foot = 19 candidates
    out = asyncio.run(node.forward({"ground_mask_b64": _mask_b64(362)}))
    pts = json.loads(out["candidate_pixels"])
    check("returns candidates", out["count"] > 1)
    check("index 0 = foot point (W//2, H-5)", pts[0] == [W // 2, H - 5])
    # every grid (non-start) point is inside the mask (v>=362)
    grid_in_mask = all(p[1] >= 362 for p in pts[1:])
    check("all grid points in mask (v>=362)", grid_in_mask)
    # Upstream early-abort (process_image:354-362): a bottom-half mask on 512
    # yields 46 candidates > 40 → the view is skipped (face-wall heuristic).
    out_big = asyncio.run(node.forward({"ground_mask_b64": _mask_b64(256)}))
    check(">40 candidates → early-abort (upstream cap)", out_big["count"] == 0)


def test_annotate_markers():
    print("[annotate_markers] overlay")
    from PIL import Image

    rgb = np.full((120, 160, 3), 200, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    node = m.AnnotateMarkersNode()
    node.config = {"radius": 8, "font_size": 16}
    out = asyncio.run(
        node.forward(
            {
                "image_b64": b64,
                "points": json.dumps([[40, 40], [120, 80]]),
                "labels": json.dumps([1, 2]),
            }
        )
    )
    ann = out["annotated_b64"]
    check("annotated non-empty", bool(ann))
    dec = np.asarray(Image.open(io.BytesIO(base64.b64decode(ann))).convert("RGB"))
    check("annotated dims preserved", dec.shape == (120, 160, 3))
    check("markers changed pixels", not np.array_equal(dec, rgb))

    # Upstream vis_candidates(multi_start=True): the foot anchor (points[0]) is
    # never drawn and grid labels start at 1 (grounded_sam_Gemini.py:183-203).
    node_up = m.AnnotateMarkersNode()
    node_up.config = {"radius": 8, "font_size": 16, "skip_first": True, "label_start": "1"}
    out_up = asyncio.run(
        node_up.forward({"image_b64": b64, "points": json.dumps([[80, 115], [40, 40], [120, 80]])})
    )
    dec_up = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(out_up["annotated_b64"]))).convert("RGB")
    )
    foot_region = dec_up[105:120, 70:90]
    check("skip_first: foot anchor not drawn", np.all(foot_region == 200))
    check("skip_first: grid markers drawn", not np.array_equal(dec_up, rgb))


def test_prompts():
    print("[_prompts] verbatim assembly + parsers")
    td = m._prompts.build_task_description()
    check("task_description embodied-robot opener", td.startswith("You are an embodied robot"))
    check("task_description Action/Stop rule", "place 'Stop' into 'Action'" in td)
    sys_i = m._prompts.build_proposer_system("walk to the kitchen")
    check("proposer system has Waypoints def", "'Waypoints' refer to locations" in sys_i)
    check("proposer system embeds instruction", "walk to the kitchen" in sys_i)
    sys_n = m._prompts.build_proposer_system("")
    check(
        "proposer (no instr) omits instr_des",
        "is a step-by-step detailed guidance for\nnavigation" not in sys_n
        and "Instruction':" not in sys_n,
    )
    pa = m._prompts.parse_pathagent('```json\n{"Thought":"t","New Planning":"p","Action":"3"}\n```')
    check("pathagent action_id=3", pa["action_id"] == 3 and not pa["is_stop"])
    st = m._prompts.parse_pathagent('{"Thought":"t","New Planning":"p","Action":"Stop"}')
    check("pathagent Stop → is_stop", st["is_stop"] and st["action_id"] == -1)
    # §3E #4: digit-first stop — 'Stop in 2 m' moves to id 2 (not a stop), like upstream parse_num
    mixed = m._prompts.parse_pathagent('{"Action":"Stop in 2 m"}')
    check("'Stop in 2 m' → id 2, not stop (#4)", mixed["action_id"] == 2 and not mixed["is_stop"])
    # §3E #5: missing 'New Planning' → upstream sentinel, not ''
    nokey = m._prompts.parse_pathagent('{"Thought":"t","Action":"1"}')
    check(
        "missing New Planning → sentinel (#5)",
        nokey["new_planning"] == m._prompts.NO_PLANNING_FALLBACK,
    )
    # ok flag: unparseable → ok False (caller skips planning append, #6); valid → ok True
    bad = m._prompts.parse_pathagent("not json at all")
    check("unparseable → ok False + stop (#6)", bad["ok"] is False and bad["is_stop"])
    check("parsed → ok True", m._prompts.parse_pathagent('{"Action":"2"}')["ok"] is True)
    pr = m._prompts.parse_proposal('```json\n{"Waypoints":[2,5],"Paths":[[1,2],[3,5]]}\n```')
    check("proposal waypoints [2,5]", pr["waypoints"] == [2, 5])
    check("proposal paths [[1,2],[3,5]]", pr["paths"] == [[1, 2], [3, 5]])
    # C (model-forced, 2026-06-17): gpt-5-mini emits string/object IDs — tolerated.
    # (The earlier D-6 'strict' trim wrongly dropped these and suppressed D-2
    # multi-hop; re-filed C. Pure labels with no digit are still dropped.)
    pr_obj = m._prompts.parse_proposal('{"Waypoints":[{"id":"2"},{"id":5}],"Paths":[]}')
    check("C: object-shaped waypoint IDs → [2,5]", pr_obj["waypoints"] == [2, 5])
    pr_str = m._prompts.parse_proposal('{"Waypoints":["id_2","6"],"Paths":[]}')
    check("C: 'id_2'/'6' string IDs → [2,6]", pr_str["waypoints"] == [2, 6])
    # object-form Paths {"Waypoint":id,"Path":[ids]} mapped by waypoint id + aligned
    pr_objp = m._prompts.parse_proposal(
        '{"Waypoints":["6","7"],"Paths":[{"Waypoint":"7","Path":["15","11","7"]},{"Waypoint":"6","Path":["13","6"]}]}'
    )
    check(
        "C: object Paths aligned by waypoint id (not position)",
        pr_objp["waypoints"] == [6, 7] and pr_objp["paths"] == [[13, 6], [15, 11, 7]],
    )
    pr_lbl = m._prompts.parse_proposal(
        '{"Waypoints":["id_2"],"Paths":[{"Waypoint":"id_2","Path":["bottom_center","id_1","id_2"]}]}'
    )
    check("C: route labels → [1,2] (pure label dropped)", pr_lbl["paths"] == [[1, 2]])
    # Upstream count-mismatch branch (llm/utils.py:43-56): list-form Paths whose
    # count differs from Waypoints → 'use path[-1] as waypoint instead'.
    pr_mm = m._prompts.parse_proposal('{"Waypoints":[2],"Paths":[[1,2],[1,3]]}')
    check(
        "count mismatch → waypoints from path[-1] (upstream branch)",
        pr_mm["waypoints"] == [2, 3] and pr_mm["paths"] == [[1, 2], [1, 3]],
    )


def _small_rgb_b64(w=80, h=60):
    from PIL import Image

    arr = np.full((h, w, 3), 180, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_aggregate_and_decider():
    print("[decider] aggregate + assemble + build_images + parse + resolve + history")
    rgb = _small_rgb_b64()
    bundles = [
        {
            "view_dir": "front",
            "rgb_b64": rgb,
            "candidates": [
                {"u": 20, "v": 40, "angle": 0.1, "distance": 2.0},
                {"u": 60, "v": 45, "angle": -0.3, "distance": 1.5},
            ],
        },
        {
            "view_dir": "left",
            "rgb_b64": rgb,
            "candidates": [{"u": 30, "v": 50, "angle": 1.5, "distance": 3.0}],
        },
    ]
    agg = m.AggregateCandidatesNode()
    agg.config = {"radius": 6, "font_size": 12}
    out = asyncio.run(agg.forward({"view_bundles": bundles}))
    manifest = json.loads(out["candidates_dict"])
    check(
        "3 ghost candidates",
        out["count"] == 3 and set(manifest.keys()) == {"0", "1", "2", "_meta"},
    )
    check("_meta.total = ghosts ever minted", manifest["_meta"]["total"] == 3)
    check("action_space_text '0, 1, 2'", out["action_space_text"] == "0, 1, 2")
    vimgs = json.loads(out["view_images_b64"])
    vlabels = json.loads(out["view_labels"])
    check("2 view images", len(vimgs) == 2)
    check(
        "front label ids {0, 1}",
        "Locations {0, 1}" in vlabels[0] and vlabels[0].startswith("(front)"),
    )
    check(
        "manifest gid2 = left wp",
        approx(manifest["2"]["angle"], 1.5) and approx(manifest["2"]["distance"], 3.0),
    )
    # Upstream packs only CONTRIBUTING views; an empty view yields no image, no
    # label, and Image {i} counts contributing views (zero_shot_agent.py:727-731).
    out_sk = asyncio.run(
        agg.forward(
            {
                "view_bundles": [
                    {"view_dir": "front", "rgb_b64": rgb, "candidates": []},
                    bundles[1],
                ]
            }
        )
    )
    sk_labels = json.loads(out_sk["view_labels"])
    check(
        "empty view skipped + Image index by contributing order",
        len(json.loads(out_sk["view_images_b64"])) == 1
        and len(sk_labels) == 1
        and sk_labels[0].startswith("(left)")
        and "in Image 0" in sk_labels[0],
    )

    asm = m.AssemblePromptNode()
    ap = asyncio.run(
        asm.forward(
            {"instruction": "go to the sofa", "action_space_text": out["action_space_text"]}
        )
    )
    check("prompt has Instruction", "Instruction: go to the sofa" in ap["prompt"])
    check("prompt step0 init-history", m._prompts.INIT_HISTORY in ap["prompt"])
    check("prompt Options line", "Options (step 0): Locations {0, 1, 2}" in ap["prompt"])
    check(
        "task_description emitted", ap["task_description"].startswith("You are an embodied robot")
    )

    bi = m.BuildImagesNode()
    bo = asyncio.run(
        bi.forward({"view_images_b64": out["view_images_b64"], "view_labels": out["view_labels"]})
    )
    check("build_images 2 images", len(bo["images"]) == 2 and len(bo["image_labels"]) == 2)
    check(
        "build_images t=0: no Options prefix (Options in the text block)",
        not bo["image_labels"][0].startswith("Options"),
    )

    pr = m.ParseResponseNode()
    po = asyncio.run(pr.forward({"response": '{"Thought":"x","New Planning":"np","Action":"2"}'}))
    check("parse action_id 2", po["action_id"] == "2" and not po["is_stop"])
    # §3E #7: n_candidates=0 → force STOP even if the decider picked an id
    po_e = asyncio.run(
        m.ParseResponseNode().forward({"response": '{"Action":"1"}', "n_candidates": 0})
    )
    check("n_candidates=0 → force stop (#7)", po_e["is_stop"] and po_e["action_id"] == "-1")

    # §3E #6: planning append gated on parse success
    class _GS:
        def __init__(self):
            self.d = {"planning": ["seed"]}

        def read(self, k):
            return self.d.get(k)

        def write(self, k, v):
            self.d[k] = v

    class _Ctx:
        def __init__(self, gs):
            self.graph_state = gs
            self.step = 1

    gs = _GS()
    asyncio.run(m.ParseResponseNode().forward({"response": "garbage not json"}, _Ctx(gs)))
    check("parse-fail → planning NOT appended (#6)", gs.d["planning"] == ["seed"])
    gs2 = _GS()
    asyncio.run(
        m.ParseResponseNode().forward(
            {"response": '{"New Planning":"np2","Action":"1"}'}, _Ctx(gs2)
        )
    )
    check("parse-ok → planning appended", gs2.d["planning"] == ["seed", "np2"])

    rs = m.ResolveActionNode()
    ro = asyncio.run(
        rs.forward({"action_id": "2", "is_stop": False, "candidates_dict": out["candidates_dict"]})
    )
    check(
        "resolve angle~1.5 dist~3.0",
        approx(float(ro["angle"]), 1.5) and approx(float(ro["distance"]), 3.0),
    )
    ro_stop = asyncio.run(
        rs.forward({"action_id": "-1", "is_stop": True, "candidates_dict": out["candidates_dict"]})
    )
    check(
        "resolve STOP → (0,0)",
        approx(float(ro_stop["angle"]), 0.0) and approx(float(ro_stop["distance"]), 0.0),
    )
    # Upstream IndexError→STOP: an id beyond the episode's accumulated ghosts
    # (>= _meta.total) stops, exactly like waypoint_path_coord[id] raising
    # (zero_shot_agent.py:838-844).
    ro_oor = asyncio.run(
        rs.forward({"action_id": "99", "is_stop": False, "candidates_dict": out["candidates_dict"]})
    )
    check(
        "id >= _meta.total → STOP (upstream IndexError path)",
        approx(float(ro_oor["angle"]), 0.0)
        and approx(float(ro_oor["distance"]), 0.0)
        and json.loads(ro_oor["path_angles"]) == [],
    )
    # A PAST ghost id (< total, not in the current manifest) has no stored path
    # (D-3 deferral) → fallback to the most-forward current candidate.
    stale = json.loads(out["candidates_dict"])
    stale["_meta"] = {"total": 100}
    stale_json = json.dumps(stale)
    ro_fb = asyncio.run(
        rs.forward({"action_id": "50", "is_stop": False, "candidates_dict": stale_json})
    )
    check(
        "stale in-range id → fallback forward (gid0 ~0.1/2.0)",
        approx(float(ro_fb["angle"]), 0.1) and approx(float(ro_fb["distance"]), 2.0),
    )
    rs_noop = m.ResolveActionNode()
    rs_noop.config = {"fallback": "noop"}
    ro_noop = asyncio.run(
        rs_noop.forward({"action_id": "50", "is_stop": False, "candidates_dict": stale_json})
    )
    check(
        "fallback noop → (0,0)",
        approx(float(ro_noop["angle"]), 0.0) and approx(float(ro_noop["distance"]), 0.0),
    )

    # t>0: the Options line leaves the text block and rides the first option
    # image's label (upstream content order — prompt_manager.py:69-79).
    gs3 = _GS()
    ap1 = asyncio.run(
        asm.forward(
            {"instruction": "go on", "action_space_text": out["action_space_text"]}, _Ctx(gs3)
        )
    )
    check(
        "t>0 prompt ends 'History:\\n', no Options",
        ap1["prompt"].endswith("History:\n") and "Options" not in ap1["prompt"],
    )
    bo1 = asyncio.run(
        m.BuildImagesNode().forward(
            {
                "view_images_b64": out["view_images_b64"],
                "view_labels": out["view_labels"],
                "action_space_text": out["action_space_text"],
            },
            _Ctx(_GS()),
        )
    )
    check(
        "t>0 first option label carries Options line",
        bo1["image_labels"][0].startswith("Options (step 1): Locations {0, 1, 2}\n(front)")
        and not bo1["image_labels"][1].startswith("Options"),
    )

    uh = m.UpdateHistoryNode()
    ho = asyncio.run(uh.forward({"action_id": "2", "is_stop": False}))
    check("history step0 entry", "Step 0, move towards location 2." in ho["history"])


def test_proposer():
    print("[proposer] pick_ground_box + propose_prep + parse_proposal")
    det = json.dumps(
        {
            "boxes": [
                {"xyxy": [0, 200, 100, 300], "score": 0.42, "phrase": "ground"},
                {"xyxy": [0, 100, 512, 512], "score": 0.30, "phrase": "ground"},
            ],
            "count": 2,
            "image_w": 512,
            "image_h": 512,
        }
    )
    pgb = m.PickGroundBoxNode()
    pgb.config = {"strategy": "top_score"}
    o1 = asyncio.run(pgb.forward({"detections": det}))
    check("pick top_score box", json.loads(o1["box"]) == [0, 200, 100, 300] and o1["found"])
    pgb2 = m.PickGroundBoxNode()
    pgb2.config = {"strategy": "largest"}
    o2 = asyncio.run(pgb2.forward({"detections": det}))
    check("pick largest box", json.loads(o2["box"]) == [0, 100, 512, 512])
    o3 = asyncio.run(pgb.forward({"detections": json.dumps({"boxes": [], "count": 0})}))
    check("no boxes → not found", o3["box"] == "" and not o3["found"])

    pp = m.ProposePrepNode()
    op = asyncio.run(pp.forward({"instruction": "enter the bedroom"}))
    check("proposer prompt embeds instruction", "enter the bedroom" in op["system_prompt"])

    cand = json.dumps([[256, 507], [40, 300], [80, 320], [120, 280]])  # 0=foot, 1..3 grid
    resp = '```json\n{"Waypoints":[2,3],"Paths":[[1,2],[1,3]]}\n```'
    ppar = m.ParseProposalNode()
    osel = asyncio.run(ppar.forward({"response": resp, "candidate_pixels": cand}))
    check(
        "selected pixels ids 2,3",
        json.loads(osel["selected_pixels"]) == [[80, 320], [120, 280]] and osel["count"] == 2,
    )
    # D-2: selected_paths maps each waypoint's route ids -> pixels, ending at the dest
    check(
        "D-2: routes mapped + end at dest",
        json.loads(osel["selected_paths"]) == [[[40, 300], [80, 320]], [[40, 300], [120, 280]]],
    )
    osel2 = asyncio.run(
        ppar.forward({"response": '{"Waypoints":[1,99]}', "candidate_pixels": cand})
    )
    check("out-of-range id dropped", json.loads(osel2["selected_pixels"]) == [[40, 300]])
    # D-2: a waypoint with no route -> single-point route (the destination itself)
    check("D-2: no-route waypoint → [[dest]]", json.loads(osel2["selected_paths"]) == [[[40, 300]]])
    # M4b: foot point (id 0) excluded by default (it's the path-start anchor, not a destination)
    osel3 = asyncio.run(ppar.forward({"response": '{"Waypoints":[0,2]}', "candidate_pixels": cand}))
    check(
        "foot point id 0 excluded by default", json.loads(osel3["selected_pixels"]) == [[80, 320]]
    )
    ppar_keep = m.ParseProposalNode()
    ppar_keep.config = {"exclude_foot_point": False}
    osel4 = asyncio.run(
        ppar_keep.forward({"response": '{"Waypoints":[0,2]}', "candidate_pixels": cand})
    )
    check(
        "foot point kept when exclude_foot_point=False",
        json.loads(osel4["selected_pixels"]) == [[256, 507], [80, 320]],
    )


def test_view_glue():
    print("[view glue] extract_view + make_bundle + annotate IMAGE output")
    views = [
        {"rgb_base64": "RGB0", "depth_raw_base64": "D0", "heading_deg": 0.0},
        {"rgb_base64": "RGB1", "depth_raw_base64": "D1", "heading_deg": 90.0},
    ]
    ev = m.ExtractViewNode()
    ev.config = {"index": 1}
    eo = asyncio.run(ev.forward({"views": views}))
    check(
        "extract_view idx1",
        eo["rgb_b64"] == "RGB1" and eo["depth_b64"] == "D1" and approx(eo["heading_deg"], 90.0),
    )
    ev0 = m.ExtractViewNode()
    ev0.config = {"index": 5}
    check("extract_view OOB → empty", asyncio.run(ev0.forward({"views": views}))["rgb_b64"] == "")

    mb = m.MakeBundleNode()
    mb.config = {"view_dir": "left"}
    bundle = asyncio.run(
        mb.forward(
            {
                "rgb_b64": "RGBX",
                "candidates": json.dumps([{"u": 1, "v": 2, "angle": 0.5, "distance": 2.0}]),
            }
        )
    )["bundle"]
    check(
        "make_bundle shape",
        bundle["view_dir"] == "left"
        and bundle["rgb_b64"] == "RGBX"
        and len(bundle["candidates"]) == 1,
    )

    an = m.AnnotateMarkersNode()
    an.config = {"radius": 5, "font_size": 10}
    ao = asyncio.run(
        an.forward(
            {
                "image_b64": _small_rgb_b64(),
                "points": json.dumps([[10, 10]]),
                "labels": json.dumps([0]),
            }
        )
    )
    img = ao["annotated_image"]
    check(
        "annotate returns IMAGE np array",
        img is not None and hasattr(img, "shape") and img.ndim == 3,
    )


def test_path_execution():
    print("[D-2] path projection + manifest + resolve path output + env step_path geometry")
    # 1) project_waypoints attaches a polar route per candidate
    depth = make_floor_depth()
    proj = m.ProjectWaypointsNode()
    proj.config = {"depth_scale": 1.0, "heading_sign": "1", "bearing_sign": "-1"}
    intr = {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "width": W, "height": H}
    po = asyncio.run(
        proj.forward(
            {
                "candidate_pixels": json.dumps([[256, 400]]),
                "paths_pixels": json.dumps([[[256, 450], [256, 400]]]),  # 2-pt route, both on floor
                "depth": depth,
                "intrinsics": intr,
            }
        )
    )
    cands = json.loads(po["candidates"])
    check("project: 1 valid candidate", po["count"] == 1)
    check("D-2: candidate carries 2-hop route", len(cands[0]["path"]) == 2)
    d_near = CAM_H * FY / (450 - CY)
    d_far = CAM_H * FY / (400 - CY)
    check(
        f"D-2: route hop dists ~[{d_near:.2f},{d_far:.2f}]",
        approx(cands[0]["path"][0]["distance"], d_near, 0.02)
        and approx(cands[0]["path"][1]["distance"], d_far, 0.02),
    )
    po2 = asyncio.run(
        proj.forward(
            {
                "candidate_pixels": json.dumps([[256, 400]]),
                "paths_pixels": json.dumps([[[256, 100]]]),  # sky pixel, invalid depth
                "depth": depth,
                "intrinsics": intr,
            }
        )
    )
    c2 = json.loads(po2["candidates"])[0]
    check(
        "D-2: invalid route → single-hop fallback to waypoint",
        len(c2["path"]) == 1 and approx(c2["path"][0]["distance"], d_far, 0.02),
    )

    # 2) aggregate carries the route into the manifest
    agg = m.AggregateCandidatesNode()
    bundles = [
        {
            "view_dir": "front",
            "rgb_b64": _small_rgb_b64(),
            "candidates": [
                {
                    "u": 20,
                    "v": 40,
                    "angle": 0.5,
                    "distance": 2.0,
                    "path": [{"angle": 0.2, "distance": 1.0}, {"angle": 0.5, "distance": 2.0}],
                }
            ],
        }
    ]
    ao = asyncio.run(agg.forward({"view_bundles": bundles}))
    man = json.loads(ao["candidates_dict"])
    check("D-2: manifest gid0 carries 2-hop path", man["0"]["path"] == [[0.2, 1.0], [0.5, 2.0]])

    # 3) resolve emits the route as parallel path_angles/path_distances lists
    rs = m.ResolveActionNode()
    ro = asyncio.run(
        rs.forward({"action_id": "0", "is_stop": False, "candidates_dict": ao["candidates_dict"]})
    )
    check("D-2: resolve path_angles", json.loads(ro["path_angles"]) == [0.2, 0.5])
    check("D-2: resolve path_distances", json.loads(ro["path_distances"]) == [1.0, 2.0])
    check(
        "D-2: back-compat single hop = destination",
        approx(float(ro["angle"]), 0.5) and approx(float(ro["distance"]), 2.0),
    )
    ro_stop = asyncio.run(
        rs.forward({"action_id": "-1", "is_stop": True, "candidates_dict": ao["candidates_dict"]})
    )
    check(
        "D-2: STOP → empty path",
        json.loads(ro_stop["path_angles"]) == [] and json.loads(ro_stop["path_distances"]) == [],
    )

    # 4) env step_path geometry (calculate_vp_rel_pos, verbatim from upstream) + registration
    from workspace.nodesets.env import env_habitat as eh

    mgr = eh.HabitatEnvManager
    fwd = mgr._calculate_vp_rel_pos([0, 0, 0], [0, 0, -2.0], 0.0)
    left = mgr._calculate_vp_rel_pos([0, 0, 0], [-2.0, 0, 0], 0.0)
    right = mgr._calculate_vp_rel_pos([0, 0, 0], [2.0, 0, 0], 0.0)
    check("step_path geom: forward → (0, 2)", approx(fwd[0], 0.0) and approx(fwd[1], 2.0))
    check(
        "step_path geom: left(-X) → (pi/2, 2)",
        approx(left[0], math.pi / 2) and approx(left[1], 2.0),
    )
    check(
        "step_path geom: right(+X) → (3pi/2, 2)",
        approx(right[0], 3 * math.pi / 2) and approx(right[1], 2.0),
    )
    rel = mgr._calculate_vp_rel_pos([0, 0, 0], [0, 0, -2.0], math.pi / 2)
    check("step_path geom: base_heading subtracted", approx(rel[0], 3 * math.pi / 2))
    check(
        "env registers step_path tool",
        eh.StepPathHabitatTool().node_type == "env_habitat__step_path",
    )


def test_registration():
    print("[registration]")
    tools = [t.node_type for t in m.AoPlannerNodeSet().get_tools()]
    expect = {
        "aoplanner__sample_waypoints",
        "aoplanner__annotate_markers",
        "aoplanner__project_waypoints",
        "aoplanner__extract_view",
        "aoplanner__make_bundle",
        "aoplanner__pick_ground_box",
        "aoplanner__propose_prep",
        "aoplanner__parse_proposal",
        "aoplanner__aggregate",
        "aoplanner__assemble_prompt",
        "aoplanner__build_images",
        "aoplanner__parse_response",
        "aoplanner__resolve_action",
        "aoplanner__update_history",
        "aoplanner__emit_stop",
    }
    check("15 tools registered", set(tools) == expect and len(tools) == 15)


if __name__ == "__main__":
    test_projection()
    test_depth_decode()
    test_view_heading()
    test_sample_waypoints()
    test_annotate_markers()
    test_prompts()
    test_proposer()
    test_view_glue()
    test_aggregate_and_decider()
    test_path_execution()
    test_registration()
    print()
    if fails:
        print(f"RESULT: {len(fails)} FAILED -> {fails}")
        sys.exit(1)
    print("RESULT: ALL PASS")
