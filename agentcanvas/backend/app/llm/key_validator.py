"""API key validation — litellm-backed with Ollama health check fallback.

Sends a minimal request to each provider to confirm the key works.
Returns (True, "ok") on success or (False, error_message) on failure.
"""

from __future__ import annotations

import logging
import re

import httpx
import litellm

from .call import API_TYPE_TO_LITELLM_PREFIX

log = logging.getLogger("agentcanvas.key_validator")


def _to_litellm_model(api_type: str, model: str, litellm_prefix: str = "") -> str:
    """Build a litellm model string for validation."""
    prefix = litellm_prefix or API_TYPE_TO_LITELLM_PREFIX.get(api_type, "openai")
    return f"{prefix}/{model}"


def validate_api_key_sync(
    api_key: str,
    base_url: str,
    model: str,
    api_type: str,
    timeout: float = 10.0,
    litellm_prefix: str = "",
) -> tuple[bool, str]:
    """Synchronous API key validation for CLI use.

    Args:
        api_key: The API key to test.
        base_url: Provider base URL (e.g. ``"https://api.openai.com/v1"``).
        model: Model name to use in the test request.
        api_type: One of ``"openai"``, ``"anthropic"``, ``"google"``, ``"ollama"``.
        timeout: Request timeout in seconds.
        litellm_prefix: litellm model prefix (e.g. ``"openai"``). Falls back
            to deriving from *api_type* if empty.

    Returns:
        ``(True, "ok")`` on success, ``(False, error_message)`` on failure.
    """
    try:
        # Ollama: lightweight health check (no model load needed)
        if api_type == "ollama":
            url = f"{base_url.rstrip('/')}/api/tags"
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url)
            if resp.status_code == 200:
                return True, "ok"
            return False, f"{resp.status_code} error — {resp.text[:120]}"

        litellm_model = _to_litellm_model(api_type, model, litellm_prefix)

        litellm.completion(
            model=litellm_model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            api_key=api_key,
            api_base=base_url or None,
            timeout=timeout,
            num_retries=0,
        )
        # If we got a response, the key works
        return True, "ok"

    except litellm.AuthenticationError:
        return False, "401 Unauthorized — check your API key"
    except litellm.PermissionDeniedError:
        return False, "403 Forbidden — key lacks permission"
    except litellm.RateLimitError:
        # Rate limited means the key is valid
        return True, "ok (rate-limited, but key is valid)"
    except litellm.Timeout:
        return False, f"timeout after {timeout}s"
    except httpx.ConnectError as exc:
        return False, f"connection error: {exc}"
    except Exception as exc:
        # litellm wraps many errors — check if it's an auth-adjacent status
        exc_str = str(exc)
        if "401" in exc_str or "Unauthorized" in exc_str:
            return False, "401 Unauthorized — check your API key"
        if "403" in exc_str or "Forbidden" in exc_str:
            return False, "403 Forbidden — key lacks permission"
        if "429" in exc_str or "rate" in exc_str.lower():
            return True, "ok (rate-limited, but key is valid)"
        log.debug("validate_api_key_sync unexpected error", exc_info=True)
        # Sanitize error message to avoid leaking API keys
        msg = re.sub(r"(sk-|key-|Bearer\s+)\S+", r"\1****", str(exc))
        return False, f"error: {msg[:200]}"
