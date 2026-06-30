"""Configuration endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ...config import get_settings
from ...llm import get_profile_store
from ...server._loopback_proxy import set_ignore_loopback_proxy

router = APIRouter()


@router.get("/")
async def get_config():
    settings = get_settings()
    store = get_profile_store()

    return {
        "vlm_max_steps": settings.vlm_max_steps,
        "slam_backend": settings.slam_backend,
        "env_backend": settings.env_backend,
        "ws_heartbeat_sec": settings.ws_heartbeat_sec,
        "ignore_loopback_proxy": settings.ignore_loopback_proxy,
        "debug": settings.debug,
        "active_profile": store.get_active(),
    }


@router.put("/")
async def update_config(updates: dict):
    """Update operational settings (ephemeral — resets on server restart).

    setattr() on the module-level singleton persists for the process lifetime.
    This is intentional: these are runtime knobs, not persistent configuration.
    Persistent config lives in .env; LLM credentials live in profiles.json.
    """
    settings = get_settings()
    allowed = {
        "vlm_max_steps",
        "slam_backend",
        "ws_heartbeat_sec",
        "ignore_loopback_proxy",
        "debug",
    }
    changed = {}
    for k, v in updates.items():
        if k in allowed and hasattr(settings, k):
            setattr(settings, k, v)
            changed[k] = v
            if k == "ignore_loopback_proxy":
                set_ignore_loopback_proxy(bool(v))
    return {"ok": True, "changed": changed, "note": "ephemeral — resets on server restart"}
