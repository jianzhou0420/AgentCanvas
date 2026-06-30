"""Generic env panel protocol for nodesets that need a fixed UI panel.

An env panel is the "control plane" for a nodeset: episode selection, splits,
play/pause/stop/reset, or any other interaction that does not belong on the
canvas as a node. Each nodeset that wants a panel implements one
``BaseEnvPanel`` subclass and references it via ``BaseNodeSet.env panel``.

The frontend renders a single generic ``EnvPanel`` driven by the
env panel's declared ``fields`` and ``actions``. There is no env-specific
code in the frontend — adding a new env nodeset means writing one env panel
class inside that nodeset's file, nothing else.

Env panels are pure: they may touch their nodeset's manager singletons and
data, but they MUST NOT import the executor or runner. Run lifecycle stays in
``LoopRunner``; env panels communicate the desired follow-up action via the
``side_effect`` field on each ``EnvPanelAction``, which the frontend
interprets after the action handler returns.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal

log = logging.getLogger("agentcanvas.env_panel")


# ── Schema ──


FieldKind = Literal["select", "number", "text", "slider"]
SideEffect = Literal["run_start", "run_pause", "run_stop", "run_step", "signal", "none"]
# "signal" instructs the env panel router to forward a framework signal
# (e.g. "episode_reset") to the running executor's state containers.
# Env panels using this form return additional keys on the action result:
#   {"ok": True, "side_effect": "signal",
#    "signal_name": "episode_reset", "signal_payload": {...}}


@dataclass
class EnvPanelField:
    """Declarative description of a panel input widget."""

    name: str
    kind: FieldKind
    label: str
    options: list[str] | None = None  # static select options; dynamic via get_options()
    min: float | None = None
    max: float | None = None
    step: float | None = None
    placeholder: str | None = None


@dataclass
class EnvPanelAction:
    """Declarative description of a panel button."""

    name: str
    label: str
    side_effect: SideEffect = "none"
    enabled_when: Literal["always", "idle", "running", "paused"] = "always"


@dataclass
class EnvPanelInfo:
    """Schema sent to the frontend for rendering."""

    name: str
    display_name: str
    fields: list[dict]
    actions: list[dict]


# ── Base ──


class BaseEnvPanel(ABC):
    """Contract every nodeset env panel implements.

    Subclasses declare ``name``, ``display_name``, ``fields``, and ``actions``
    as ClassVars, then implement the four async hooks below. Instances are
    held by the env panel registry; one instance per loaded nodeset.

    The registry stamps ``_context`` after instantiation. It carries the
    deployment mode ("local" or "server") and (for server mode) the URL of
    the spawned auto-server subprocess. Env panels can read this in
    ``on_load`` to give an accurate status message when episode control
    cannot reach the env directly.
    """

    name: ClassVar[str]
    display_name: ClassVar[str]
    fields: ClassVar[list[EnvPanelField]]
    actions: ClassVar[list[EnvPanelAction]]

    _context: dict[str, Any] = {}

    def info(self) -> EnvPanelInfo:
        return EnvPanelInfo(
            name=self.name,
            display_name=self.display_name,
            fields=[asdict(f) for f in self.fields],
            actions=[asdict(a) for a in self.actions],
        )

    @abstractmethod
    async def on_load(self) -> dict[str, Any]:
        """Return the env panel's initial state for the panel.

        Should include current values for every declared field plus an
        ``available`` boolean and optional ``message``/``error`` for display.
        """

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        """Handle a panel field change. Return the updated state.

        Default: no-op that records the value but does nothing else. Override
        to perform side effects (e.g. a split change rebuilds the env).
        """
        return await self.on_load()

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle an action button click. Return ``{ok, side_effect, ...}``.

        ``side_effect`` tells the frontend which (if any) run-lifecycle call
        to make next: ``run_start`` / ``run_pause`` / ``run_stop`` /
        ``run_step`` / ``none``. The env panel never calls the runner itself.
        """
        return {"ok": True, "side_effect": "none"}

    async def get_options(self, field: str) -> list[dict[str, Any]]:
        """Populate dynamic select fields. Return ``[{value, label}, ...]``.

        Default: empty list. Override for fields whose options come from
        runtime state (e.g. the episode dropdown).
        """
        return []


# ── Remote proxy ──


class RemoteEnvPanelProxy(BaseEnvPanel):
    """Forwards every BaseEnvPanel call to an env panel running in an
    auto-hosted subprocess (server mode).

    Used by ``WorkspaceComponentRegistry._load_nodeset_as_server`` so the canvas
    panel works identically regardless of whether the env nodeset runs
    in-process or in a separate Python interpreter. The schema (fields,
    actions, display_name) is fetched once at registration time via
    ``GET /env-panel/info``; runtime calls hit ``/env-panel/state``,
    ``/env-panel/options/{field}``, ``/env-panel/field/{field}`` and
    ``/env-panel/action/{action}`` on the spawned subprocess.

    The proxy assigns ``name``/``display_name``/``fields``/``actions`` as
    instance attributes (rather than ClassVars) — Python attribute lookup
    treats both the same, so ``info()`` works unchanged.
    """

    def __init__(self, server_url: str, info: dict[str, Any]) -> None:
        self._server_url = server_url.rstrip("/")
        self.name = info.get("name", "")
        self.display_name = info.get("display_name", self.name)
        self.fields = [
            EnvPanelField(
                name=f.get("name", ""),
                kind=f.get("kind", "text"),
                label=f.get("label", ""),
                options=f.get("options"),
                min=f.get("min"),
                max=f.get("max"),
                step=f.get("step"),
                placeholder=f.get("placeholder"),
            )
            for f in info.get("fields", [])
        ]
        self.actions = [
            EnvPanelAction(
                name=a.get("name", ""),
                label=a.get("label", ""),
                side_effect=a.get("side_effect", "none"),
                enabled_when=a.get("enabled_when", "always"),
            )
            for a in info.get("actions", [])
        ]
        self._context = {
            "mode": "server",
            "server_url": server_url,
            "nodeset_name": self.name,
        }

    async def _get(self, path: str) -> Any:
        import httpx

        from ..server._loopback_proxy import loopback_httpx_kwargs

        async with httpx.AsyncClient(timeout=60.0, **loopback_httpx_kwargs()) as client:
            r = await client.get(self._server_url + path)
            r.raise_for_status()
            return r.json()

    async def _post(self, path: str, body: dict) -> Any:
        import httpx

        from ..server._loopback_proxy import loopback_httpx_kwargs

        async with httpx.AsyncClient(timeout=60.0, **loopback_httpx_kwargs()) as client:
            r = await client.post(self._server_url + path, json=body)
            r.raise_for_status()
            return r.json()

    async def on_load(self) -> dict[str, Any]:
        return await self._get("/env-panel/state")

    async def on_field_change(self, name: str, value: Any) -> dict[str, Any]:
        return await self._post(f"/env-panel/field/{name}", {"value": value})

    async def on_action(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._post(f"/env-panel/action/{name}", {"params": params})

    async def get_options(self, field: str) -> list[dict[str, Any]]:
        return await self._get(f"/env-panel/options/{field}")


# ── Registry ──
# Module-level dict — CPython GIL guarantees safe get/set without a lock.

_env_panels: dict[str, BaseEnvPanel] = {}


def register_env_panel(panel: BaseEnvPanel) -> None:
    """Register an env panel. Called by WorkspaceComponentRegistry on nodeset load."""
    _env_panels[panel.name] = panel
    log.info("Registered env panel: %s", panel.name)


def unregister_env_panel(name: str) -> None:
    """Unregister an env panel. Called by WorkspaceComponentRegistry on nodeset unload."""
    if _env_panels.pop(name, None) is not None:
        log.info("Unregistered env panel: %s", name)


def get_env_panel(name: str) -> BaseEnvPanel | None:
    return _env_panels.get(name)


def list_env_panels() -> list[BaseEnvPanel]:
    return list(_env_panels.values())
