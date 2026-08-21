"""§14.9/§14.3 — the CapabilityManifest is the ONE source of the tool surface.

What broke without this: the SDK dwp cells were briefed with the BARE prompt
and told to "Call observe() first" (verified in std_sdkeh_opus-5_dwp's
recorded session_inputs) while the bridge registered observe/look_around the
model was never supposed to see. These tests pin: server-level registration
== manifest == SDK allowlist, prompt text carries none of the retired
teachings, and the driver hands dwp cells the dwp first prompt.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CA))

from eharness.capabilities import (  # noqa: E402
    DWP_FIRST_PROMPT,
    DWP_FIRST_PROMPT_NOIMG,
    MONITOR_TOOLS,
    action_space_prompt,
    mcp_visible_tools,
    sdk_allowed_tools,
)

PY = sys.executable
_BRIDGE_PROBE = r"""
import json, sys
sys.path.insert(0, {ca!r})
import importlib.util
spec = importlib.util.spec_from_file_location(
    "eh_bridge_probe", {bridge!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(json.dumps({{t.name: t.description or ""
                  for t in mod.mcp._tool_manager.list_tools()}}))
"""


def _bridge_surface(dwp: bool) -> dict[str, str]:
    """name → registered description, from a subprocess with EH_DWP pinned."""
    code = _BRIDGE_PROBE.format(ca=str(CA),
                                bridge=str(CA / "bridges" / "eharness_bridge.py"))
    env = {"PATH": "/usr/bin:/bin", "EH_DWP": "1" if dwp else "0",
           "HOME": str(Path.home())}
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                         env=env, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _bridge_tools(dwp: bool) -> list[str]:
    return sorted(_bridge_surface(dwp).keys())


def test_manifest_shapes():
    # face withdrawn 2026-08-12 evening (user ruling): goto + step ARE the
    # surface; step's own 2/3 turns cover direction changes
    assert mcp_visible_tools("dwp") == ("step", "goto", "veer", "recall")
    assert "observe" in mcp_visible_tools("bare")
    assert sdk_allowed_tools("dwp") == [
        "mcp__env__step", "mcp__env__goto",
        "mcp__env__veer", "mcp__env__recall"]
    assert MONITOR_TOOLS == mcp_visible_tools("dwp")


def test_bridge_registry_matches_manifest_dwp():
    tools = _bridge_tools(dwp=True)
    assert sorted(tools) == sorted(mcp_visible_tools("dwp")), tools
    assert "observe" not in tools and "look_around" not in tools


def test_bridge_registry_bare_keeps_observe():
    tools = _bridge_tools(dwp=False)
    for t in ("observe", "look_around", "step", "veer", "recall"):
        assert t in tools, tools
    assert "goto" not in tools and "face" not in tools


def test_bridge_descriptions_come_from_manifest():
    # §14.9: registered DESCRIPTIONS, not just names, must be the manifest's —
    # the bridge's old hand-written step text taught the bare surface a goto
    # its server never registered (review P1)
    from eharness.capabilities import (DWP_STEP_DESC, GOTO_DESC,
                                       STEP_DESC_BARE)
    dwp = _bridge_surface(dwp=True)
    assert dwp["step"].startswith(DWP_STEP_DESC), dwp["step"][:120]
    assert dwp["goto"].startswith(GOTO_DESC), dwp["goto"][:120]
    assert "face" not in dwp, "face must be OFF the model surface"
    bare = _bridge_surface(dwp=False)
    assert bare["step"].startswith(STEP_DESC_BARE), bare["step"][:120]
    assert "goto" not in bare["step"], "bare step teaches a missing verb"


def test_bridge_carries_reflect_answer_fields():
    # the [STOP AND ANSWER] gate needs these on the SDK arm too (review P2)
    import importlib.util as _ilu
    import inspect as _inspect
    import os as _os
    _os.environ["EH_DWP"] = "1"
    try:
        spec = _ilu.spec_from_file_location(
            "_eh_probe_sig", CA / "bridges" / "eharness_bridge.py")
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import asyncio as _aio
        tools = {t.name: t for t in _aio.get_event_loop_policy()
                 .new_event_loop().run_until_complete(mod.mcp.list_tools())}
        for fn_name in ("step", "goto"):
            schema = getattr(tools[fn_name], "inputSchema", {}) or {}
            props = schema.get("properties", {})
            for f in ("segment_done", "at_goal", "answer", "revoke_dead_end"):
                assert f in props, f"{fn_name} missing {f}: {sorted(props)}"
    finally:
        _os.environ.pop("EH_DWP", None)


def test_prompts_carry_no_retired_teachings():
    from prompts import DWP_SYSTEM_PROMPT, build_briefing
    rendered = build_briefing("go to the chair", 500, bare=True, dwp=True)
    for banned in ("observe(", "zero env steps", "always safe",
                   "are not checked"):
        assert banned not in rendered, banned
        assert banned not in DWP_FIRST_PROMPT
        assert banned not in DWP_FIRST_PROMPT_NOIMG
    # the action-space text is IN the rendered briefing, not a second copy
    assert action_space_prompt("dwp") in rendered
    # …and the real-turn cost is stated, without teaching the retired face
    assert "15 deg turned" in rendered
    assert "face(" not in rendered, "face left the model surface 2026-08-12"


def test_solo_addendum_matches():
    from eharness.solo import DWP_ADDENDUM
    assert "zero env steps" not in DWP_ADDENDUM
    assert "your own step()s do not" not in DWP_ADDENDUM


def test_driver_selects_dwp_first_prompt():
    src = (CA / "driver.py").read_text()
    assert "DWP_FIRST_PROMPT if dwp" in src
    # the dwp flag is read from extra (where cells put it), not cfg top-level
    assert 'cfg["extra"].get("dwp"' in src
    # session_inputs records the ACTUAL first prompt, not the constant
    assert '"first_prompt": ctx.first_prompt' in src


def test_sdk_adapter_derives_allowlist():
    src = (CA / "harnesses" / "claude_sdk.py").read_text()
    assert "sdk_allowed_tools(" in src
    assert '"mcp__env__face"' not in src  # no hand-written dwp list survives


def test_codex_identity_manifest_and_refine():
    # §18-6: codex must say WHO it is to the bridge, write an honest
    # unaudited context manifest, and expose the §17 refine hook
    src = (CA / "harnesses" / "codex_cli.py").read_text()
    assert '"EH_EXECUTOR": "codex"' in src
    assert '"label": "codex-managed-unaudited"' in src
    assert "context_manifest_" in src
    assert "def refine_instruction(" in src
    # §16.2: with no bootstrap artifact codex must not claim pictures exist
    assert "DWP_FIRST_PROMPT_NOIMG" in src


def test_two_tier_labelled_as_its_own_axis():
    # §18-7: the two-tier arm must never pass as the 12-cycle contract —
    # cells pin image_window=6 explicitly, so the label carries the truth
    from harnesses.eharness_agent import EharnessAdapter
    label = EharnessAdapter().inherent.get("two_tier_context", "")
    assert "NOT the 12-cycle contract" in label
    assert "NOT multiturn" in label


def test_hermetic_conftest_present():
    # §16 intro: pytest coding-agent/eharness/tests must collect without an
    # external PYTHONPATH — the conftest owns both required paths
    src = (CA / "eharness" / "tests" / "conftest.py").read_text()
    assert '"harnesses" / "mini"' in src and "sys.path.insert" in src


def test_map_contract_is_canonical_and_taught():
    # §20.4: the map is taught from ONE legend that both the system prompt
    # and the per-turn IMAGE 2 label quote — colour semantics included, so
    # the words match the pixels.
    from eharness.capabilities import MAP_LEGEND, MAP_USE_RULES
    from prompts import build_briefing
    b = build_briefing("walk to the kitchen", 500, bare=True, dwp=True)
    assert "the THREE things" in b and MAP_LEGEND in b and MAP_USE_RULES in b
    for word in ("GREEN", "RED", "AMBER", "UNKNOWN", "BLUE-PURPLE"):
        assert word in MAP_LEGEND, word
    # stage3 P0-1: the legend teaches the SINGLE anchor-fixed map — no
    # dual-panel composite, remembered places are dashed circles
    assert "ONE map" in MAP_LEGEND and "fixed to the WORLD" in MAP_LEGEND
    assert "TWO PANELS" not in MAP_LEGEND and "DASHED" in MAP_LEGEND
    src = (CA / "harnesses" / "mini" / "depth_toolset.py").read_text()
    assert 'text_part("IMAGE 2 — " + MAP_LEGEND)' in src
    assert "never walked" in MAP_LEGEND  # POTENTIAL is not walkable blind
    # …and the briefing teaches the compact facts contract + off-camera use
    assert "WALKABLE PLACES" in b and "silence" in b.lower()
    assert "remembered / off-camera place is a legitimate choice" in b


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} passed")
