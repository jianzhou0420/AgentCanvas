"""Profile CRUD endpoints — /api/profiles.

API keys are not stored in profiles. Each provider's key is read from
its standard env var (see ``app.llm.providers.PROVIDER_REGISTRY``). The
``api_key_set`` flag in responses reflects whether that env var is set.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...llm import PROVIDER_REGISTRY, LLMProfile, get_profile_store, resolve_provider_config
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


class BatchUpsertBody(BaseModel):
    # ``keys`` is accepted but ignored — kept for backward-compat with
    # older frontends. Real keys live in env vars.
    keys: dict[str, str] = {}
    active: str  # provider to activate ("" = none)
    active_model: str  # model for active provider
    overrides: dict[str, dict] = {}  # provider_id -> {base_url, model, api_type}


def _profile_to_dict(p: LLMProfile) -> dict:
    return {
        "provider": p.provider,
        "model": p.model,
        "api_key_set": bool(get_provider_api_key(p.provider)) or p.provider == "ollama",
        "base_url": p.base_url,
        "api_type": p.api_type,
    }


def _registry_to_dict() -> dict:
    return {
        k: {
            "label": v.label,
            "base_url": v.base_url,
            "api_type": v.api_type,
            "default_model": v.default_model,
        }
        for k, v in PROVIDER_REGISTRY.items()
    }


@router.get("/")
async def list_profiles():
    store = get_profile_store()
    profiles = {n: _profile_to_dict(p) for n, p in store.list_profiles().items()}
    return {
        "active": store.get_active(),
        "profiles": profiles,
        "registry": _registry_to_dict(),
    }


@router.put("/batch")
async def batch_upsert(body: BatchUpsertBody):
    store = get_profile_store()
    # ``body.keys`` is intentionally ignored — keys live in env vars.
    # We still ensure a profile exists for each provider mentioned, so
    # the UI's per-provider config is preserved.
    for provider_id in body.keys:
        if store.get(provider_id) is None:
            reg = PROVIDER_REGISTRY.get(provider_id)
            store.create(
                provider_id,
                LLMProfile(
                    provider=provider_id,
                    model=reg.default_model if reg else "",
                ),
            )
    # Apply overrides (custom base_url, model, etc.)
    for provider_id, ov in body.overrides.items():
        existing = store.get(provider_id)
        if existing:
            store.update(provider_id, **ov)
        elif provider_id == "ollama":
            store.create(
                provider_id,
                LLMProfile(
                    provider=provider_id,
                    model=ov.get("model", "llama3.1"),
                    base_url=ov.get("base_url", ""),
                ),
            )
    # Set active + model
    if body.active:
        p = store.get(body.active)
        if p:
            store.update(body.active, model=body.active_model)
            store.set_active(body.active)
        else:
            store.set_active("")
    else:
        store.set_active("")
    return {"ok": True}


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


# ── Model discovery ──


_NON_CHAT_PREFIXES = (
    "text-embedding",
    "dall-e",
    "tts-",
    "whisper",
    "babbage",
    "davinci",
    "text-moderation",
    "omni-moderation",
    "chatgpt-image",
    "gpt-image",
    "sora-",
)
_NON_CHAT_SUBSTRINGS = ("-realtime", "-transcribe", "-tts", "-instruct", "-audio")


async def _fetch_provider_models(
    api_type: str,
    api_key: str,
    base_url: str,
) -> list[str]:
    """Query a provider's models endpoint and return sorted model IDs."""
    timeout = httpx.Timeout(15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if api_type == "ollama":
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            return sorted(m["name"] for m in resp.json().get("models", []))

        if api_type == "google":
            resp = await client.get(
                f"{base_url}/models",
                params={"key": api_key, "pageSize": 100},
            )
            resp.raise_for_status()
            return sorted(
                m["name"].removeprefix("models/")
                for m in resp.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            )

        if api_type == "anthropic":
            resp = await client.get(
                f"{base_url}/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []))

        # OpenAI-compatible (default)
        resp = await client.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        ids = [m["id"] for m in resp.json().get("data", [])]
        return sorted(
            i
            for i in ids
            if not any(i.startswith(p) for p in _NON_CHAT_PREFIXES)
            and not any(s in i for s in _NON_CHAT_SUBSTRINGS)
        )


@router.get("/{name}/models")
async def list_provider_models(name: str):
    """Fetch available models from a provider's API endpoint."""
    store = get_profile_store()
    profile = store.get(name)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    cfg = resolve_provider_config(profile)
    if not cfg["api_key"] and cfg["api_type"] != "ollama":
        raise HTTPException(status_code=400, detail="No API key configured")
    try:
        models = await _fetch_provider_models(cfg["api_type"], cfg["api_key"], cfg["base_url"])
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Provider returned {exc.response.status_code}",
        ) from None
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    return {"models": models, "provider": name}
