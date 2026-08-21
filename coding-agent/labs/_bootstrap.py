"""Path + old-name aliases so the parked labs lines run in place.

The fc0661b restructure moved coding-agent's shared machinery into ``core/``
and retired the eharness / vlaharness / ImagineVLN lines here. The parked code
still imports that machinery by its old top-level names (``driver``,
``prompts``, ``toolset``, ``harnesses.*``). Rather than rewrite dozens of
import sites, this shim aliases those names onto their new ``core.*`` homes and
puts the labs roots on ``sys.path`` — so a parked line runs as it always did,
while ``core/`` and the std board stay decoupled. Nothing on the board imports
this; only ``labs/run.py`` and ``labs/conftest.py`` load it.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_LABS = Path(__file__).resolve().parent          # coding-agent/labs
_CODING = _LABS.parent                            # coding-agent
_REPO = _CODING.parent                            # repo root
_BACKEND = _REPO / "agentcanvas" / "backend"

# labs/ (import eharness|vlaharness), coding-agent/ (import core.*), backend
# (import app.*), and each line dir (imaginevln's bare `from imagine_tools`).
_PATHS = [
    _LABS, _CODING, _BACKEND,
    _LABS / "eharness", _LABS / "vlaharness", _LABS / "imaginevln",
]

# old top-level name -> new core home. Submodules of `harnesses` resolve via
# the aliased package's __path__, so only the package itself needs aliasing.
_ALIASES = {
    "driver": "core.driver",
    "prompts": "core.prompts",
    "toolset": "core.harnesses.mini.toolset",
    "harnesses": "core.harnesses",
}

_done = False


def setup() -> None:
    """Idempotent: install path roots + module aliases once."""
    global _done
    if _done:
        return
    for p in _PATHS:
        sp = str(p)
        if p.is_dir() and sp not in sys.path:
            sys.path.insert(0, sp)
    for old, new in _ALIASES.items():
        if old in sys.modules:
            continue
        try:
            sys.modules[old] = importlib.import_module(new)
        except Exception:
            # A line whose new home can't import in this env stays unaliased;
            # only that line fails, and only when actually run.
            pass
    _done = True
