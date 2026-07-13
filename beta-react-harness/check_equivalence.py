"""Equivalence checks: mini-swe-agent path vs the claude-SDK path.

Offline (no env server, no LLM). Three checks, each against the SDK path's
OWN source as the fixture:

1. Tool schemas — HabitatToolSet's declared {name, description, input_schema}
   vs the MCP bridge introspected in-process (the exact schemas SDK sessions
   received, per the recorded session_inputs events). Bare and full variants.
2. Prompts — our jinja templates rendered vs the SDK driver's str.format
   prompts, plus the first user message.
3. Clearance readout — same synthetic depth frame through both
   implementations.

Run:  ~/miniforge3/envs/agentcanvas/bin/python beta-react-harness/check_equivalence.py
Exit code 0 = all equivalent.
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
BRIDGE_PATH = REPO_ROOT / "beta-coding-agent" / "mcp_bridge.py"
SDK_DRIVER_PATH = REPO_ROOT / "beta-coding-agent" / "run_episodes.py"

sys.path.insert(0, str(HERE))
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from jinja2 import StrictUndefined, Template

import run_episodes as ours
from toolset import HabitatToolSet

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


if __name__ == "__main__":
    check_schemas()
    check_prompts()
    check_clearance()
    print(f"\n{len(FAILURES)} failure(s)")
    sys.exit(1 if FAILURES else 0)
