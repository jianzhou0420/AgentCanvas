"""Equivalence checks: mini-swe-agent path vs the claude-SDK path.

Offline (no env server, no LLM). Three checks, each against the SDK path's
OWN source as the fixture (the frozen legacy drivers under
coding-agent/legacy/ — kept unedited exactly so they can serve here):

1. Tool schemas — HabitatToolSet's declared {name, description, input_schema}
   vs the MCP bridge introspected in-process (the exact schemas SDK sessions
   received, per the recorded session_inputs events). Bare and full variants.
2. Prompts — the legacy mini jinja templates rendered vs the legacy SDK
   driver's str.format prompts, plus the first user message; and the live
   prompt surface (coding-agent/prompts.py) byte-equal to the legacy SDK
   driver it claims to be moved verbatim from.
3. Clearance readout — same synthetic depth frame through both
   implementations.

Run:  ~/miniforge3/envs/agentcanvas/bin/python coding-agent/mini/check_equivalence.py
Exit code 0 = all equivalent.
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import importlib.util
import math
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

HERE = Path(__file__).resolve().parent            # coding-agent/mini
CODING_AGENT_DIR = HERE.parent
BRIDGE_PATH = CODING_AGENT_DIR / "bridges" / "mcp_bridge.py"
WP_BRIDGE_PATH = CODING_AGENT_DIR / "bridges" / "wp_bridge.py"
PROMPTS_PATH = CODING_AGENT_DIR / "prompts.py"
SDK_DRIVER_PATH = CODING_AGENT_DIR / "legacy" / "beta-coding-agent" / "run_episodes.py"
MINI_DRIVER_PATH = CODING_AGENT_DIR / "legacy" / "beta-react-harness" / "run_episodes.py"

sys.path.insert(0, str(HERE))
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from jinja2 import StrictUndefined, Template

from toolset import HabitatToolSet, WaypointToolSet

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)
        if detail:
            print(detail)


def _import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bridge_schemas(bare: bool) -> list[dict]:
    """Import the bridge with the same env gating the SDK driver used."""
    saved = os.environ.get("HABITAT_BARE")
    os.environ["HABITAT_BARE"] = "1" if bare else "0"
    try:
        mod = _import_from_path(f"_bridge_{'bare' if bare else 'full'}", BRIDGE_PATH)
        tools = asyncio.run(mod.mcp.list_tools())
        return [
            {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
            for t in tools
        ]
    finally:
        if saved is None:
            os.environ.pop("HABITAT_BARE", None)
        else:
            os.environ["HABITAT_BARE"] = saved


def _diff(a: str, b: str) -> str:
    return "\n".join(
        difflib.unified_diff(a.splitlines(), b.splitlines(), "sdk", "mini", lineterm="")
    )


def check_schemas() -> None:
    for bare in (False, True):
        variant = "bare" if bare else "full"
        sdk = {t["name"]: t for t in _bridge_schemas(bare)}
        mini = {
            t["name"]: t
            for t in HabitatToolSet("http://x", bare=bare).tool_schemas()
        }
        check(f"schema[{variant}] tool set", set(sdk) == set(mini),
              f"  sdk={sorted(sdk)} mini={sorted(mini)}")
        for name in sorted(set(sdk) & set(mini)):
            check(
                f"schema[{variant}] {name}.description",
                sdk[name]["description"] == mini[name]["description"],
                _diff(repr(sdk[name]["description"]), repr(mini[name]["description"])),
            )
            check(
                f"schema[{variant}] {name}.input_schema",
                sdk[name]["input_schema"] == mini[name]["input_schema"],
                f"  sdk={sdk[name]['input_schema']}\n  mini={mini[name]['input_schema']}",
            )


def check_prompts() -> None:
    sdk = _import_from_path("_sdk_driver", SDK_DRIVER_PATH)
    ours = _import_from_path("_mini_legacy_driver", MINI_DRIVER_PATH)
    live = _import_from_path("_live_prompts", PROMPTS_PATH)
    instruction = 'Walk past the sofa, then "turn left" at the door.'
    budget = 500
    pairs = [
        ("system_prompt[full]", sdk.SYSTEM_PROMPT.format(instruction=instruction, budget=budget),
         ours.SYSTEM_TEMPLATE),
        ("system_prompt[bare]",
         sdk.BARE_SYSTEM_PROMPT.format(instruction=instruction, budget=budget),
         ours.BARE_SYSTEM_TEMPLATE),
    ]
    for label, sdk_text, template in pairs:
        mini_text = Template(template, undefined=StrictUndefined).render(
            task=instruction, step_budget=budget
        )
        # jinja strips the template's single trailing newline (str.format keeps
        # it) — the prompts must be byte-identical modulo exactly that byte.
        check(f"{label} (modulo trailing \\n)", sdk_text == mini_text + "\n",
              _diff(sdk_text, mini_text))
    check("first_prompt", sdk.FIRST_PROMPT == ours.FIRST_PROMPT,
          _diff(sdk.FIRST_PROMPT, ours.FIRST_PROMPT))
    # the live prompt surface vs the frozen legacy source it was moved from
    for attr in ("SYSTEM_PROMPT", "BARE_SYSTEM_PROMPT", "FIRST_PROMPT"):
        sdk_text, live_text = getattr(sdk, attr), getattr(live, attr)
        check(f"prompts.py {attr} verbatim", sdk_text == live_text,
              _diff(sdk_text, live_text))


def check_clearance() -> None:
    bridge = _import_from_path("_bridge_clearance", BRIDGE_PATH)
    rng = np.random.default_rng(42)
    arr = rng.random((224, 224), dtype=np.float32)
    depth_field = {
        "__ndarray__": base64.b64encode(arr.tobytes()).decode(),
        "dtype": "float32",
        "shape": list(arr.shape),
    }
    sdk_val = bridge._clearance_m(depth_field)
    mini_val = HabitatToolSet._clearance_m(depth_field)
    check("clearance_m", sdk_val == mini_val, f"  sdk={sdk_val}\n  mini={mini_val}")
    check("clearance_m[None]",
          bridge._clearance_m(None) is None and HabitatToolSet._clearance_m(None) is None)


def _wp_bridge_schemas() -> list[dict]:
    mod = _import_from_path("_wp_bridge_schema", WP_BRIDGE_PATH)
    tools = asyncio.run(mod.mcp.list_tools())
    return [
        {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
        for t in tools
    ]


def _synthetic_pano(px: int = 512) -> list[dict]:
    """12 solid-color tiles so both annotators decode byte-identical PNGs.
    px > WP_VIEW_PX exercises the LANCZOS downscale path too."""
    views = []
    for dir_id in range(12):
        img = PILImage.new("RGB", (px, px), ((dir_id * 20) % 256, 90, 170))
        buf = BytesIO()
        img.save(buf, format="PNG")
        views.append({"dir_id": dir_id,
                      "rgb_base64": base64.b64encode(buf.getvalue()).decode()})
    return views


def check_wp() -> None:
    """wp_bridge.py (SDK/codex path) vs WaypointToolSet (mini path): tool
    schemas the model sees, the geometry helpers, and the annotated strip."""
    wpb = _import_from_path("_wp_bridge_mod", WP_BRIDGE_PATH)

    sdk = {t["name"]: t for t in _wp_bridge_schemas()}
    mini = {
        t["name"]: t
        for t in WaypointToolSet("http://x", wp_server_url="http://y").tool_schemas()
    }
    check("wp schema tool set", set(sdk) == set(mini),
          f"  sdk={sorted(sdk)} mini={sorted(mini)}")
    for name in sorted(set(sdk) & set(mini)):
        check(f"wp schema {name}.description",
              sdk[name]["description"] == mini[name]["description"],
              _diff(repr(sdk[name]["description"]), repr(mini[name]["description"])))
        check(f"wp schema {name}.input_schema",
              sdk[name]["input_schema"] == mini[name]["input_schema"],
              f"  sdk={sdk[name]['input_schema']}\n  mini={mini[name]['input_schema']}")

    # geometry helpers over an angle sweep + both candidate encodings
    angles = [i * math.pi / 6 for i in range(13)] + [-0.3, 3.5, 6.5, 0.0]
    check("wp _direction_of",
          all(wpb._direction_of(a) == WaypointToolSet._direction_of(a) for a in angles))
    check("wp _norm_pi",
          all(abs(wpb._norm_pi(a) - WaypointToolSet._norm_pi(a)) < 1e-12 for a in angles))
    raw = {"0": {"angle": 0.2, "distance": 1.5},
           "1": {"angle": 2.0, "distance": 3.0},
           "2": [3.14, 0.75]}  # opennav list-encoding path
    cands = wpb._normalize_candidates(raw)
    check("wp _normalize_candidates", cands == WaypointToolSet._normalize_candidates(raw))
    check("wp _action_options",
          wpb._action_options(cands) == WaypointToolSet._action_options(cands))

    # annotated strip: same views + candidates → byte-identical PNG
    views = _synthetic_pano()
    sdk_png = wpb._annotate_strip(views, cands)  # bridge reads WP_VIEW_PX global
    mini_png = WaypointToolSet._annotate_strip(views, cands, wpb.WP_VIEW_PX)
    check("wp _annotate_strip bytes", sdk_png == mini_png,
          f"  len sdk={len(sdk_png)} mini={len(mini_png)}")


if __name__ == "__main__":
    check_schemas()
    check_prompts()
    check_clearance()
    check_wp()
    print(f"\n{len(FAILURES)} failure(s)")
    sys.exit(1 if FAILURES else 0)
