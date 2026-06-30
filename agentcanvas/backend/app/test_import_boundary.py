"""Framework import boundary guard (ADR-platform-001).

``agentcanvas/backend/app/`` is the *framework* and has zero domain knowledge:
all VLN / embodied-AI code (simulators, policies, agents) lives in
``workspace/`` and is discovered at runtime by ``WorkspaceComponentRegistry``.
The framework must therefore never import a domain simulator package.

This test imports the framework's entry-point modules in a *clean*
interpreter and asserts none of the forbidden top-level packages ended up in
``sys.modules`` — catching an accidental ``import habitat`` added anywhere in
the framework's transitive import graph.

Run: ``cd agentcanvas/backend && python -m pytest app/test_import_boundary.py -v``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Top-level packages the framework must never pull in (CLAUDE.md).
FORBIDDEN = ("habitat", "habitat_sim", "vlnce_baselines", "habitat_baselines")

# Framework surfaces that, between them, transitively import the bulk of
# ``app/`` — the FastAPI app, the eval subprocess entry, the registry, and the
# execution engine.
ENTRY_MODULES = (
    "app.main",
    "app.eval_subprocess_main",
    "app.components.registry",
    "app.agent_loop.graph_executor",
)

_BACKEND_DIR = Path(__file__).resolve().parents[1]  # agentcanvas/backend


def test_framework_never_imports_domain_simulators() -> None:
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in ENTRY_MODULES)
        + f"forbidden = {FORBIDDEN!r}\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in forbidden)\n"
        "assert not bad, 'framework imported forbidden modules: ' + ', '.join(bad)\n"
        "print('import boundary OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_BACKEND_DIR),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "framework import boundary violated:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
