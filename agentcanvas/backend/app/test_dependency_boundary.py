"""Graph SDK dependency-boundary guard.

The installable package (root ``pyproject.toml``) promises a tiered
dependency contract:

* ``import agentcanvas`` — **stdlib-only**. No third-party module at all.
* a pure-local ``g.run()`` — needs only the declared core dependency
  (``pydantic-settings`` and what it brings: pydantic, dotenv). Everything
  heavier (litellm, fastapi, httpx, msgpack, numpy, …) belongs to an extra
  and must stay behind a lazy import.

Nothing in the type system enforces this — one eager top-level import added
anywhere in the transitive graph silently breaks the core tier, and only a
user in a clean venv would ever notice. This test recreates the clean-venv
semantics in a subprocess: a meta-path finder *raises ModuleNotFoundError*
for every forbidden top-level package (active blocking, not passive
``sys.modules`` inspection), so soft imports (``try: import numpy``) degrade
exactly as they would in a bare venv while a hard import fails the test.

Run: ``cd agentcanvas/backend && python -m pytest app/test_dependency_boundary.py -v``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Heavy third-party packages that must never be *required* by the core tier.
# (They may be installed in the dev env — the blocker makes them unimportable.)
HEAVY = (
    "litellm",
    "fastapi",
    "uvicorn",
    "httpx",
    "msgpack",
    "numpy",
    "yaml",
    "torch",
    "PIL",
    "websockets",
    "psutil",
    "mcp",
)

# The declared core dependency chain — forbidden while importing `agentcanvas`
# (stdlib-only surface), allowed once a graph actually runs (Settings needs it).
CORE = ("pydantic_settings", "pydantic", "dotenv")

_BACKEND_DIR = Path(__file__).resolve().parents[1]  # agentcanvas/backend

_SUBPROCESS_CODE = f"""
import sys

FORBIDDEN = set({HEAVY!r}) | set({CORE!r})

class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise ModuleNotFoundError(
                f"blocked by test_dependency_boundary: {{fullname}}", name=fullname
            )
        return None

sys.meta_path.insert(0, Blocker())

# Tier 0 — the import surface is stdlib-only (even the core dep is blocked).
import agentcanvas
from agentcanvas import Graph, RunEvent, catalog, graph_to_code  # noqa: F401
print("tier 0 OK: import agentcanvas is stdlib-only")

# Tier 1 — a pure-local run needs only the declared core dependency.
for m in {CORE!r}:
    FORBIDDEN.discard(m)

from app.agent_loop.builtin_nodes import register_node
from app.components.bases import BaseCanvasNode, PortDef

class Seven(BaseCanvasNode):
    node_type = "dep_boundary_seven"
    input_ports: list = []
    output_ports = [PortDef("y", "ANY", "")]

    async def forward(self, inputs, ctx):
        return {{"y": self.config.get("x", 0) + 1}}

register_node(Seven)

g = Graph(name="dep-boundary")
a = g.add("dep_boundary_seven", id="a", x=7)
o = g.graph_out("r")
g.connect(a.out("y"), o.in_("value"))
events = []
r = g.run(on_event=events.append)
assert r["r"] == 8, r.outputs
assert any(e.kind == "graph_complete" for e in events)
assert "print(" in g.to_code()
print("tier 1 OK: pure-local run + on_event + to_code, heavy deps blocked")
"""


def test_sdk_core_tier_needs_no_heavy_deps() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_CODE],
        cwd=str(_BACKEND_DIR),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "SDK dependency boundary violated — a heavy dep became load-bearing "
        "for the core tier (check for a new eager top-level import):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # 3.10 (CI): fall back to pip's vendored parser
        from pip._vendor import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(path.read_text())


def _norm(req: str) -> str:
    return "".join(req.split()).lower()


def test_backend_extra_matches_requirements_txt() -> None:
    """`pip install agentcanvas[backend]` and `pip install -r requirements.txt`
    promise the same stack, declared in two hand-maintained lists (the
    pyproject comment says "keep in sync" — this is the check). The closure of
    core deps + server + llm + backend extras must equal requirements.txt."""
    repo_root = _BACKEND_DIR.parents[1]
    project = _load_toml(repo_root / "pyproject.toml")["project"]
    extras = project["optional-dependencies"]

    closure: set[str] = {_norm(r) for r in project["dependencies"]}
    for group in ("server", "llm", "backend"):
        closure |= {_norm(r) for r in extras[group] if not r.startswith("agentcanvas")}

    requirements = {
        _norm(line.split("#", 1)[0])
        for line in (_BACKEND_DIR / "requirements.txt").read_text().splitlines()
        if line.split("#", 1)[0].strip()
    }

    assert closure == requirements, (
        "pyproject [backend] closure and requirements.txt drifted:\n"
        f"only in pyproject:      {sorted(closure - requirements)}\n"
        f"only in requirements:   {sorted(requirements - closure)}"
    )
