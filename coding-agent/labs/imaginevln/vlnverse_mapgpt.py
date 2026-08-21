"""Load VLNVerse's MapGPT pieces from the VLNVerse source tree itself.

Nothing here is re-implemented. The prompt templates, the history formatter,
the action parser and the action-option binning are the ones VLNVerse actually
runs, loaded live from::

    ~/Desktop/Projects/VLNVerse/internnav/model/basemodel/mllm/prompt.py
                                                        /prompt_managers.py
                                                        /mllm_utils.py
    ~/Desktop/Projects/VLNVerse/internnav/trainer/mllm_isaac_trainer.py

Two loading styles, both deliberate:

* ``prompt.py`` / ``prompt_managers.py`` are imported by path — they only need
  numpy, so the real modules run as-is.
* ``mllm_utils.py`` imports cv2 (absent from the agentcanvas env) and
  ``mllm_isaac_trainer.py`` pulls in the whole habitat stack, so the three
  functions we need are lifted out of their ASTs and compiled on their own.
  Still the upstream source, byte for byte — an upstream edit is picked up on
  the next run instead of silently drifting from a copy.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import os
import re
import sys

import numpy as np

VLNVERSE = os.environ.get("VLNVERSE_PATH", "/home/xunyi/Desktop/Projects/VLNVerse")
MLLM_DIR = os.path.join(VLNVERSE, "internnav", "model", "basemodel", "mllm")
TRAINER = os.path.join(VLNVERSE, "internnav", "trainer", "mllm_isaac_trainer.py")


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _lift(path: str, want: dict[str, str]):
    """Compile named top-level / method functions out of a module we cannot
    import. ``want`` maps {exported_name: source_name}."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    found: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in want.values():
            found.setdefault(node.name, node)
    missing = set(want.values()) - set(found)
    if missing:
        raise ImportError(f"{path}: cannot find {sorted(missing)} — did VLNVerse move it?")
    ns: dict = {"np": np, "numpy": np, "math": math, "re": re, "__builtins__": __builtins__}
    for src_name in want.values():
        node = found[src_name]
        mod = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(mod)
        exec(compile(mod, path, "exec"), ns)          # noqa: S102 - upstream source
    return {export: ns[src_name] for export, src_name in want.items()}


# ── the real prompt templates + history formatter ──
prompt = _load_by_path("vlnverse_prompt", os.path.join(MLLM_DIR, "prompt.py"))
sys.modules["internnav.model.basemodel.mllm.prompt"] = prompt   # what prompt_managers imports
prompt_managers = _load_by_path("vlnverse_prompt_managers",
                                os.path.join(MLLM_DIR, "prompt_managers.py"))

get_prompt_manager = prompt_managers.get_prompt_manager
# the position-trace renderer — MapGPT's one distinctive ingredient, and the
# only thing separating it from NavGPT in VLNVerse's own code
format_mapgpt_history = prompt_managers.format_mapgpt_history
SYSTEM_MAPGPT = prompt.SYSTEM_MAPGPT
PROMPT_MAPGPT = prompt.PROMPT_MAPGPT

# ── the real parser + the revisit guard ──
_lifted = _lift(os.path.join(MLLM_DIR, "mllm_utils.py"),
                {"parse_action": "parse_action",
                 "has_repeated_location_visits": "has_repeated_location_visits"})
parse_action = _lifted["parse_action"]
has_repeated_location_visits = _lifted["has_repeated_location_visits"]

# ── the real Front/Left/Back/Right binning (a method, called with self=None) ──
_bin = _lift(TRAINER, {"get_action_options": "get_action_options"})["get_action_options"]


def get_action_options(radians_list, mask=None):
    return _bin(None, radians_list, mask=mask)


def provenance() -> dict:
    """What was loaded, so a run's artifacts record which upstream it ran against."""
    return {
        "vlnverse_path": VLNVERSE,
        "system_mapgpt_chars": len(SYSTEM_MAPGPT),
        "prompt_mapgpt_chars": len(PROMPT_MAPGPT),
        "sources": {name: round(os.path.getmtime(p), 0) for name, p in (
            ("prompt.py", os.path.join(MLLM_DIR, "prompt.py")),
            ("prompt_managers.py", os.path.join(MLLM_DIR, "prompt_managers.py")),
            ("mllm_utils.py", os.path.join(MLLM_DIR, "mllm_utils.py")),
            ("mllm_isaac_trainer.py", TRAINER))},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(provenance(), indent=2))
    fn = get_prompt_manager("mapgpt")
    demo = fn({"instruction": "Go past the pool.", "history": [], "positions": [np.zeros(3)],
               "collisions": [], "visited": [0], "dialogue_history": [""],
               "heading": 0.0, "action_options": get_action_options([0.0, 1.2, -1.2])})
    print("-" * 70)
    print(demo[-700:])
