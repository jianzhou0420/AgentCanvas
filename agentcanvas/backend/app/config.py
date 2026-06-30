"""AgentCanvas Backend configuration — loads from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

# Framework-level failsafe iteration cap, used when neither the env's per-episode
# hook nor the graph's authored ``step_budget`` field is set. See the resolver
# chain in ``agent_loop/eval_batch.py`` for the precedence rules.
DEFAULT_STEP_BUDGET: int = 1000

# Default workspace lives at <repo-root>/workspace. Computed from this file's
# location so the backend is portable across clones/checkout paths. Override
# via the WORKSPACE_DIR env var or a `workspace_dir` entry in `.env`.
# config.py → app → backend → agentcanvas → <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_WORKSPACE_DIR = str(_REPO_ROOT / "workspace")


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    vlm_max_steps: int = 20  # max ReAct iterations per run
    slam_backend: str = "mock"  # "mock" | "gaussian" | "habitat"
    env_backend: str = "habitat"  # "habitat" | "none" — simulator for Navigate page
    ws_heartbeat_sec: int = 15
    # When True (default), backend↔child-server httpx calls bypass HTTP_PROXY /
    # HTTPS_PROXY. Required when the user runs a local proxy (tinyproxy etc.)
    # since auto-host children listen on random high ports the proxy can't see.
    # See app/server/_loopback_proxy.py for the framework-side toggle.
    ignore_loopback_proxy: bool = True
    # Base URL a server-mode subprocess uses to call back to the executor's
    # /api/internal endpoints (cross-nodeset container access, reverse log/error
    # push). Default None → resolved from host/port via ``resolve_executor_url``.
    # An explicit ``AGENTCANVAS_EXECUTOR_URL`` env var still takes precedence at
    # the injection sites (it is what is actually handed to each subprocess).
    executor_url: str | None = None
    workspace_dir: str = _DEFAULT_WORKSPACE_DIR
    # Optional overlay loaded after frozen workspace_dir. When set, the
    # WorkspaceComponentRegistry scans both dirs: frozen first, then active
    # overrides by name (last-write-wins for nodesets/nodes/policies;
    # explicit active-first lookup for graph JSON / exp.yaml / hooks.json).
    # See registry.resolve_graph_path() for the resolver. Default None =
    # no overlay = behavior bit-identical to a single-dir scan.
    active_workspace_dir: str | None = None

    # Habitat VLN-CE settings
    habitat_exp_config: str = "vlnce_baselines/config/r2r_baselines/cma_pm_da.yaml"
    habitat_split: str = "val_unseen"
    habitat_gpu_id: int = 0

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Runtime mutations via setattr() persist for the process lifetime.
    They are intentionally ephemeral — restart resets to .env defaults.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def resolve_executor_url() -> str:
    """The base URL a subprocess uses to reach the executor's /api/internal.

    Explicit ``Settings.executor_url`` wins; otherwise built from the bound
    port over loopback (the bind host ``0.0.0.0`` is not a connect address, so
    we always dial ``127.0.0.1``). Replaces the ``http://localhost:8000``
    literal that was hardcoded at the prototype injection sites.
    """
    s = get_settings()
    if s.executor_url:
        return s.executor_url
    return f"http://127.0.0.1:{s.port}"
