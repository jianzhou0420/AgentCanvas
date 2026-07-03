"""Profile CRUD endpoints — /api/profiles.

Profile-plane only: named {provider, model, overrides} CRUD + activation.
Provider-plane questions (registry, key status, live models, validation,
capabilities) live in ``providers.py`` under /api/providers. API keys are
never stored on profiles — they resolve via ``~/.agentcanvas/.keys`` →
env var (see ``app.llm.keystore``); ``api_key_set`` reflects that.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...llm import LLMProfile, get_profile_store
from ...llm.providers import get_provider_api_key

router = APIRouter()


class CreateProfileBody(BaseModel):
    name: str
    provider: str
    model: str
    base_url: str = ""
    api_type: str = ""


class UpdateProfileBody(BaseModel):
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_type: str | None = None


class ActivateBody(BaseModel):
    name: str  # "" = env fallback


def _profile_to_dict(p: LLMProfile) -> dict:
    return {
        "provider": p.provider,
        "model": p.model,
        "api_key_set": bool(get_provider_api_key(p.provider)) or p.provider == "ollama",
        "base_url": p.base_url,
        "api_type": p.api_type,
    }


@router.get("/")
async def list_profiles():
    store = get_profile_store()
    profiles = {n: _profile_to_dict(p) for n, p in store.list_profiles().items()}
    return {
        "active": store.get_active(),
        "profiles": profiles,
    }


@router.post("/")
async def create_profile(body: CreateProfileBody):
    store = get_profile_store()
    profile = LLMProfile(
        provider=body.provider,
        model=body.model,
        base_url=body.base_url,
        api_type=body.api_type,
    )
    try:
        store.create(body.name, profile)
    except ValueError:
        raise HTTPException(
            status_code=409, detail=f"Profile '{body.name}' already exists"
        ) from None
    return {"ok": True, "name": body.name}


@router.put("/{name}")
async def update_profile(name: str, body: UpdateProfileBody):
    store = get_profile_store()
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        store.update(name, **fields)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found") from None
    return {"ok": True}


@router.delete("/{name}")
async def delete_profile(name: str):
    store = get_profile_store()
    try:
        store.delete(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found") from None
    return {"ok": True}


@router.post("/activate")
async def activate_profile(body: ActivateBody):
    store = get_profile_store()
    try:
        store.set_active(body.name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Profile '{body.name}' not found") from None
    return {"ok": True, "active": body.name}
