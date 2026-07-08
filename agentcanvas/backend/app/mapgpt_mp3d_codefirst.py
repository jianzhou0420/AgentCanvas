"""Reconstruct the verified MapGPT-MP3D graph purely in Python code.

    python -m app.mapgpt_mp3d_codefirst          # build + verify vs the JSON
    python -m app.mapgpt_mp3d_codefirst --save   # + write the code-built JSON

This takes the hand-authored, verified canvas graph
``workspace/graphs/vln/verified/mapgpt_mp3d.json`` (22 nodes, 49 wires, a
state container + access grants, an iterIn/iterOut episode loop) and rebuilds
the *same* topology through the code-first :class:`app.code_first.Graph`
builder — no JSON authoring, no canvas GUI.

Wiring is written explicitly as ``g.connect(source.out("x"), target.in_("y"))``
— one line per typed data dependency.

``main()`` proves faithfulness by comparing a semantic signature (node
types + configs, wire multiset, containers, grants, budget) of the code-built
graph against the verified JSON, ignoring UI-only noise (positions, edge ids,
synthesised iterIn ports). It does NOT run the episode — that needs the env
nodeset + GPU + LLM and must go through the sanctioned ``/experiment:run``
path on a hosted backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.code_first import Graph
from app.graph_def import GraphDefinition

VERIFIED_JSON = (
    Path(__file__).resolve().parents[3]
    / "workspace/graphs/vln/verified/mapgpt_mp3d.json"
)

# The planner system prompt is data, not structure — copied verbatim from the
# verified graph (JSON string escapes are valid Python string escapes).
PLANNER_SYSTEM_PROMPT = "You are an embodied robot that navigates in the real world. You need to explore between some places marked with IDs and ultimately find the destination to stop. At each step, a series of images corresponding to the places you have explored and have observed will be provided to you.\n'Instruction' is a global, step-by-step detailed guidance, but you might have already executed some of the commands. You need to carefully discern the commands that have not been executed yet.\n'History' represents the places you have explored in previous steps along with their corresponding images. It may include the correct landmarks mentioned in the 'Instruction' as well as some past erroneous explorations.\n'Trajectory' represents the ID info of the places you have explored. You start navigating from Place 0.\n'Map' refers to the connectivity between the places you have explored and other places you have observed.\n'Supplementary Info' records some places and their corresponding images you have ever seen but have not yet visited. These places are only considered when there is a navigation error, and you decide to backtrack for further exploration.\n'Previous Planning' records previous long-term multi-step planning info that you can refer to now.\n'Action options' are some actions that you can take at this step.\nFor each provided image of the places, you should combine the 'Instruction' and carefully examine the relevant information, such as scene descriptions, landmarks, and objects. You need to align 'Instruction' with 'History' (including corresponding images) to estimate your instruction execution progress and refer to 'Map' for path planning. Check the Place IDs in the 'History' and 'Trajectory', avoiding repeated exploration that leads to getting stuck in a loop, unless it is necessary to backtrack to a specific place.\nIf you can already see the destination, estimate the distance between you and it. If the distance is far, continue moving and try to stop within 1 meter of the destination.\nYour answer should be JSON format and must include three fields: 'Thought', 'New Planning', and 'Action'. You need to combine 'Instruction', 'Trajectory', 'Map', 'Supplementary Info', your past 'History', 'Previous Planning', 'Action options', and the provided images to think about what to do next and why, and complete your thinking into 'Thought'.\nBased on your 'Map', 'Previous Planning' and current 'Thought', you also need to update your new multi-step path planning to 'New Planning'.\nAt the end of your output, you must provide a single capital letter in the 'Action options' that corresponds to the action you have decided to take, and place only the letter into 'Action', such as \"Action: A\"."


def build() -> Graph:
    """Rebuild MapGPT-MP3D node-for-node, wire-for-wire, via explicit g.connect()."""
    g = Graph(name="MapGPT-MP3D", eval_graph=True, step_budget=50)

    # ── nodes (logic lives in the nodeset; here we only reference + configure) ──
    reset = g.add("env_mp3d__reset", id="reset", n_headings=12)
    iter_in = g.add(
        "iterIn", id="iter_in", version=3, pairedWith="iter_out",
        initPorts=[
            {"name": "instruction", "wire_type": "TEXT", "persist": True},
            {"name": "viewpoint_id", "wire_type": "TEXT", "persist": False},
            {"name": "heading", "wire_type": "TEXT", "persist": False},
            {"name": "navigable_json", "wire_type": "TEXT", "persist": False},
            {"name": "views", "wire_type": "LIST[IMAGE]", "persist": False},
            {"name": "view_meta", "wire_type": "TEXT", "persist": False},
        ],
    )
    observe = g.add("mapgpt__observe", id="observe")
    update_map = g.add("mapgpt__update_map", id="update_map")
    build_options = g.add("mapgpt__build_options", id="build_options", stop_after=3)
    render_prompt = g.add("mapgpt__render_prompt", id="render_prompt")
    planner = g.add(
        "llmCall", id="planner_llm",
        ports=[
            {"name": "system_prompt", "wire_type": "TEXT"},
            {"name": "plan", "wire_type": "TEXT", "required": True},
            {"name": "rgb", "wire_type": "LIST[IMAGE]"},
            {"name": "image_labels", "wire_type": "LIST[TEXT]"},
        ],
        temperature=1, max_tokens=6000, mode="single_turn",
        profile="gpt-5-mini", template="{plan}", system_prompt=PLANNER_SYSTEM_PROMPT,
    )
    parse = g.add("mapgpt__parse_action", id="parse_action")
    step = g.add("env_mp3d__step_waypoint", id="step", n_headings=12)
    update_history = g.add("mapgpt__update_history", id="update_history")
    image_budget = g.add("mapgpt__image_budget", id="image_budget", max_images=20)
    iter_out = g.add(
        "iterOut", id="iter_out", pairedWith="iter_in",
        ports=[
            {"name": "viewpoint_id", "wire_type": "TEXT", "persist": True},
            {"name": "heading", "wire_type": "TEXT", "persist": True},
            {"name": "navigable_json", "wire_type": "TEXT", "persist": True},
            {"name": "views", "wire_type": "LIST[IMAGE]", "persist": True},
            {"name": "view_meta", "wire_type": "TEXT", "persist": True},
        ],
    )
    evaluate = g.add("env_mp3d__evaluate", id="evaluate")
    viewer_thinking = g.add("textScroll", id="viewer_thinking")
    viewer_action = g.add("textScroll", id="viewer_action")
    viewer_metrics = g.add("metrics", id="viewer_metrics")
    viewer_panorama = g.add(
        "imageViewer", id="viewer_panorama", rows=1, cols=1,
        ports=[{"name": "views", "wire_type": "LIST[IMAGE]"}],
    )
    output_metrics = g.add(
        "graphOut", id="output_port__metrics", portName="metrics", wireType="METRICS"
    )
    seed_nav = g.add("env_mp3d__observe_navigable", id="seed_nav")
    seed_pano = g.add("env_mp3d__observe_panorama", id="seed_pano", n_headings=12)
    loop_nav = g.add("env_mp3d__observe_navigable", id="loop_nav")
    loop_pano = g.add("env_mp3d__observe_panorama", id="loop_pano", n_headings=12)

    # ── wires: one explicit, typed connection per data dependency ──
    # run-start seeds → loop entry (iterIn init side)
    g.connect(reset.out("instruction"), iter_in.in_("init_instruction"))
    g.connect(seed_nav.out("viewpoint_id"), iter_in.in_("init_viewpoint_id"))
    g.connect(seed_nav.out("heading"), iter_in.in_("init_heading"))
    g.connect(seed_nav.out("navigable_json"), iter_in.in_("init_navigable_json"))
    g.connect(seed_pano.out("views"), iter_in.in_("init_views"))
    g.connect(seed_pano.out("view_meta"), iter_in.in_("init_view_meta"))

    # loop entry → observe (both the init side and the carried side feed observe)
    g.connect(iter_in.out("init_viewpoint_id"), observe.in_("viewpoint_id"))
    g.connect(iter_in.out("iterout_viewpoint_id"), observe.in_("viewpoint_id"))
    g.connect(iter_in.out("init_heading"), observe.in_("heading"))
    g.connect(iter_in.out("iterout_heading"), observe.in_("heading"))
    g.connect(iter_in.out("init_navigable_json"), observe.in_("navigable_json"))
    g.connect(iter_in.out("iterout_navigable_json"), observe.in_("navigable_json"))
    g.connect(iter_in.out("init_views"), observe.in_("views"))
    g.connect(iter_in.out("iterout_views"), observe.in_("views"))
    g.connect(iter_in.out("init_view_meta"), observe.in_("view_meta"))
    g.connect(iter_in.out("iterout_view_meta"), observe.in_("view_meta"))
    g.connect(iter_in.out("init_instruction"), render_prompt.in_("instruction"))

    # perception → map → options → prompt
    g.connect(observe.out("current_vp"), update_map.in_("current_vp"))
    g.connect(observe.out("candidates_json"), update_map.in_("candidates_json"))
    g.connect(observe.out("candidate_tiles"), update_map.in_("candidate_tiles"))
    g.connect(observe.out("candidates_json"), build_options.in_("candidates_json"))
    g.connect(update_map.out("topo_snapshot"), build_options.in_("topo_snapshot"))
    g.connect(update_map.out("topo_snapshot"), render_prompt.in_("topo_snapshot"))
    g.connect(build_options.out("options_text"), render_prompt.in_("options_text"))
    g.connect(build_options.out("options_json"), parse.in_("options_json"))

    # prompt → planner LLM → parse action
    g.connect(render_prompt.out("prompt"), planner.in_("plan"))
    g.connect(render_prompt.out("image_list"), planner.in_("rgb"))
    g.connect(render_prompt.out("image_labels"), planner.in_("image_labels"))
    g.connect(render_prompt.out("image_count"), image_budget.in_("image_count"))
    g.connect(planner.out("response"), parse.in_("response"))
    g.connect(parse.out("viewpoint_id"), step.in_("viewpoint_id"))
    g.connect(parse.out("action_phrase"), update_history.in_("action_phrase"))
    g.connect(parse.out("thought"), update_history.in_("thought"))

    # step → observe next viewpoint → carry back to loop (iterOut)
    g.connect(step.out("info"), loop_nav.in_("trigger"))
    g.connect(step.out("info"), loop_pano.in_("trigger"))
    g.connect(loop_nav.out("viewpoint_id"), iter_out.in_("viewpoint_id"))
    g.connect(loop_nav.out("heading"), iter_out.in_("heading"))
    g.connect(loop_nav.out("navigable_json"), iter_out.in_("navigable_json"))
    g.connect(loop_pano.out("views"), iter_out.in_("views"))
    g.connect(loop_pano.out("view_meta"), iter_out.in_("view_meta"))

    # termination: image budget OR env terminated → stop the loop
    g.connect(image_budget.out("done"), iter_out.in_("stop"))
    g.connect(step.out("terminated"), iter_out.in_("stop"))

    # after-loop: iterOut final side → evaluate → metrics out
    g.connect(iter_out.out("final_stop"), evaluate.in_("trigger"))
    g.connect(evaluate.out("metrics"), viewer_metrics.in_("metrics"))
    g.connect(evaluate.out("metrics"), output_metrics.in_("value"))

    # live viewer taps
    g.connect(planner.out("response"), viewer_thinking.in_("text"))
    g.connect(parse.out("action_phrase"), viewer_action.in_("text"))
    g.connect(seed_pano.out("views"), viewer_panorama.in_("views"))
    g.connect(loop_pano.out("views"), viewer_panorama.in_("views"))

    # ── shared state ──
    g.container(
        "graph_state",
        label="MapGPT Navigation State",
        states={
            "topo_map": {
                "type": "lastWrite",
                "value_type": "ANY",
                "config": {
                    "initial_value": {
                        "nodes_list": [],
                        "graph": {},
                        "trajectory": [],
                        "node_imgs": [],
                    }
                },
                "lifetime": "run",
            },
            "history": {"type": "lastWrite", "value_type": "TEXT", "lifetime": "run"},
            "planning": {
                "type": "lastWrite",
                "value_type": "TEXT",
                "config": {"initial_value": "Navigation has just started, with no planning yet."},
                "lifetime": "run",
            },
        },
    )
    g.grant("update_map", "graph_state", id="ag_update_map")
    g.grant("render_prompt", "graph_state", id="ag_render_prompt")
    g.grant("parse_action", "graph_state", id="ag_parse_action")
    g.grant("update_history", "graph_state", id="ag_update_history")

    return g


# ── faithfulness check ────────────────────────────────────────────────────


def _clean_config(node_type: str, cfg: dict) -> dict:
    cfg = dict(cfg or {})
    cfg.pop("_expanded", None)  # UI-only fold state
    if node_type == "iterIn":
        cfg.pop("ports", None)  # synthesised from initPorts + paired iterOut
    return cfg


def signature(gd: dict) -> dict:
    """Semantic projection: everything that affects execution, nothing UI."""
    return {
        "nodes": {
            nd["id"]: {"type": nd["type"], "config": _clean_config(nd["type"], nd.get("config", {}))}
            for nd in gd["nodes"]
        },
        "edges": sorted(
            (e["source"], e.get("sourceHandle", ""), e["target"], e.get("targetHandle", ""))
            for e in gd["edges"]
        ),
        "containers": {c["id"]: c.get("states", {}) for c in gd.get("containers", [])},
        "grants": sorted((a["node_id"], a["container_id"]) for a in gd.get("access_grants", [])),
        "step_budget": gd.get("step_budget"),
        "eval_graph": gd.get("eval_graph"),
    }


def _diff(built: dict, orig: dict) -> list[str]:
    out: list[str] = []
    bn, on = built["nodes"], orig["nodes"]
    for nid in sorted(set(bn) | set(on)):
        if nid not in bn:
            out.append(f"node MISSING in code: {nid}")
        elif nid not in on:
            out.append(f"node EXTRA in code: {nid}")
        elif bn[nid] != on[nid]:
            if bn[nid]["type"] != on[nid]["type"]:
                out.append(f"node {nid}: type {bn[nid]['type']} != {on[nid]['type']}")
            else:
                bc, oc = bn[nid]["config"], on[nid]["config"]
                for k in sorted(set(bc) | set(oc)):
                    if bc.get(k) != oc.get(k):
                        out.append(f"node {nid}.config[{k!r}] differs")
    be, oe = set(map(tuple, built["edges"])), set(map(tuple, orig["edges"]))
    for e in sorted(oe - be):
        out.append(f"edge MISSING in code: {e}")
    for e in sorted(be - oe):
        out.append(f"edge EXTRA in code: {e}")
    for key in ("containers", "grants", "step_budget", "eval_graph"):
        if built[key] != orig[key]:
            out.append(f"{key} differs")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="PATH", nargs="?", const="__default__",
                    help="write the code-built graph JSON (default: alongside the verified graph)")
    args = ap.parse_args()

    g = build()
    built = g.to_dict()
    orig = GraphDefinition.from_dict(json.loads(VERIFIED_JSON.read_text())).to_dict()

    print(f"code-built: {len(built['nodes'])} nodes, {len(built['edges'])} edges")
    print(f"verified:   {len(orig['nodes'])} nodes, {len(orig['edges'])} edges")
    diffs = _diff(signature(built), signature(orig))
    if diffs:
        print(f"\nMISMATCH ({len(diffs)}):")
        for d in diffs:
            print(f"  - {d}")
        raise SystemExit(1)
    print("\nMATCH — code-built graph is semantically identical to the verified JSON.")

    if args.save is not None:
        path = (
            VERIFIED_JSON.parent / "mapgpt_mp3d_codefirst.json"
            if args.save == "__default__"
            else Path(args.save)
        )
        g.save(path)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
