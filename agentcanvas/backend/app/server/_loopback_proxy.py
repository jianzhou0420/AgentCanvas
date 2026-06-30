"""Loopback HTTP proxy policy for in-process server↔child-server calls.

The backend spawns auto-host children on localhost (random high ports) and
talks to them via httpx. When the user has ``HTTP_PROXY`` / ``HTTPS_PROXY``
in their shell env (e.g. ``http://localhost:8888`` for tinyproxy), httpx
honors it and tries to route loopback traffic through the proxy — which
typically can't see the random ephemeral port, so /health polls fail with
"Unable to connect" and child-server registration stalls.

This module exposes a process-wide flag (default: ignore the system proxy
for loopback calls) and a small helper that callers splat into httpx as
kwargs. Toggled at runtime via the /api/config endpoint.
"""

from __future__ import annotations

_ignore_loopback_proxy: bool = True


def set_ignore_loopback_proxy(value: bool) -> None:
    global _ignore_loopback_proxy
    _ignore_loopback_proxy = bool(value)


def get_ignore_loopback_proxy() -> bool:
    return _ignore_loopback_proxy


def loopback_httpx_kwargs() -> dict:
    """Kwargs to splat into ``httpx.get`` / ``httpx.Client`` / ``httpx.AsyncClient``.

    Returns ``{"trust_env": False}`` when bypass is enabled — that disables
    httpx's reading of ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY`` /
    netrc / SSL env vars for this call. Returns ``{}`` otherwise, so the
    caller keeps full httpx defaults.
    """
    if _ignore_loopback_proxy:
        return {"trust_env": False}
    return {}
