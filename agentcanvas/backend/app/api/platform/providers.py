"""Provider-plane endpoints — /api/providers.

One router per plane (RFC: llm-provider-api): everything that is a
*provider* question — the code-owned registry, key status, live model
lists, key validation, and the parameter rulebook — lives here, keyed by
provider id. Profile CRUD stays in ``profiles.py``.

Keys are persisted to ``~/.agentcanvas/.keys`` (outside the repo, 0600;
see ``app.llm.keystore``) and resolved file → env → none.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...llm import (
    PROVIDER_REGISTRY,
    get_capabilities,
    get_key_store,
    get_profile_store,
    get_provider_key_source,
)
from ...llm.key_validator import validate_api_key_sync
from ...llm.providers import get_provider_api_key

router = APIRouter()


class KeyBody(BaseModel):
    key: str


def _registry_entry(provider_id: str) -> dict:
    reg = PROVIDER_REGISTRY[provider_id]
    source = get_provider_key_source(provider_id)
    return {
        "label": reg.label,
        "api_type": reg.api_type,
        "base_url": reg.base_url,
        "default_model": reg.default_model,
        "key_env": reg.env_var,
        # Ollama needs no key — usability is reachability, probed by /validate.
        "key_set": source != "none" or provider_id == "ollama",
        "key_source": source,
    }


def _require_provider(provider_id: str):
    reg = PROVIDER_REGISTRY.get(provider_id)
    if reg is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider_id}'")
    return reg


def _provider_call_config(provider_id: str) -> dict:
    """base_url/api_type for provider-level calls (models, validate) —
    registry defaults, with a profile's base_url override honored when one
    exists for this provider (e.g. a relocated Ollama)."""
    reg = PROVIDER_REGISTRY[provider_id]
    base_url = reg.base_url
    for profile in get_profile_store().list_profiles().values():
        if profile.provider == provider_id and profile.base_url:
            base_url = profile.base_url
            break
    return {
        "api_type": reg.api_type,
        "base_url": base_url.rstrip("/"),
        "api_key": get_provider_api_key(provider_id),
        "litellm_prefix": reg.litellm_prefix,
        "default_model": reg.default_model,
    }


@router.get("/")
async def list_providers():
    return {pid: _registry_entry(pid) for pid in PROVIDER_REGISTRY}


@router.put("/{provider_id}/key")
async def set_provider_key(provider_id: str, body: KeyBody):
    reg = _require_provider(provider_id)
    if not reg.env_var:
        raise HTTPException(
            status_code=400, detail=f"Provider '{provider_id}' does not use an API key"
        )
    if not body.key.strip():
        raise HTTPException(status_code=400, detail="Key must be non-empty (DELETE to remove)")
    get_key_store().set(reg.env_var, body.key.strip())
    return {"ok": True, "key_env": reg.env_var, "key_source": "file"}


@router.delete("/{provider_id}/key")
async def delete_provider_key(provider_id: str):
    reg = _require_provider(provider_id)
    if not reg.env_var:
        raise HTTPException(
            status_code=400, detail=f"Provider '{provider_id}' does not use an API key"
        )
    removed = get_key_store().delete(reg.env_var)
    return {
        "ok": True,
        "removed": removed,
        "key_source": get_provider_key_source(provider_id),
    }


@router.post("/{provider_id}/validate")
def validate_provider_key(provider_id: str):
    """Minimal round trip against the provider (sync — FastAPI runs this
    in the threadpool). Uses the resolved key + registry defaults."""
    _require_provider(provider_id)
    cfg = _provider_call_config(provider_id)
    if not cfg["api_key"] and cfg["api_type"] != "ollama":
        return {"ok": False, "message": "no key configured"}
    ok, message = validate_api_key_sync(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["default_model"],
        api_type=cfg["api_type"],
        litellm_prefix=cfg["litellm_prefix"],
    )
    return {"ok": ok, "message": message}


@router.get("/{provider_id}/capabilities")
async def provider_capabilities(provider_id: str, model: str = ""):
    _require_provider(provider_id)
    return {
        "provider": provider_id,
        "model": model,
        "capabilities": get_capabilities(provider_id, model),
    }


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


@router.get("/{provider_id}/models")
async def list_provider_models(provider_id: str):
    """Fetch available models from a provider's API endpoint — keyed by
    provider id (the noun model discovery actually depends on)."""
    _require_provider(provider_id)
    cfg = _provider_call_config(provider_id)
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
    return {"provider": provider_id, "models": models}
