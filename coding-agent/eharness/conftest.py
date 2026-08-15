"""Hermetic pytest entry (§16 intro): `pytest coding-agent/eharness` must
collect and run WITHOUT an externally exported PYTHONPATH.

The suites import three families by bare name — ``driver``/``prompts``
(coding-agent/), ``eharness.*`` (the package), and the mini executor's
modules (``model``, ``toolset``, ``depth_toolset`` under
harnesses/mini/). Test collection used to die on ``from model import
NavToolsModel`` unless the caller remembered both paths; this conftest is
the one place that knows them.
"""
from __future__ import annotations

import sys
from pathlib import Path

_CODING_AGENT = Path(__file__).resolve().parents[1]
for _p in (str(_CODING_AGENT), str(_CODING_AGENT / "harnesses" / "mini")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
