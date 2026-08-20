"""litellm gateway — the cross-pairing serving component.

Lets a harness speak its NATIVE wire format to a model from another vendor:
Claude Code (anthropic-messages via ``ANTHROPIC_BASE_URL``) driving gpt-5.6,
codex (openai-chat via ``model_providers``) driving opus-4.8. The gateway is
an owned ``litellm --config`` subprocess bound to loopback; each Route maps
the alias the harness asks for to the litellm slug that actually serves it,
with the vendor key referenced as ``os.environ/<VAR>`` so key material never
touches the config file on disk.

Native cells never route through here — the gateway engages only for cells
whose extra carries ``("api_gateway", "litellm")``. Lifecycle mirrors the
ollama serving contract: the adapter starts it in ``prepare()``, stops it in
``finalize()``; nothing persists between runs, logs land under
``outputs/api_gateway/`` for the same slice-audit treatment as the ollama
serve log.

The master key is a loopback auth token, not a secret: litellm's proxy
refuses unauthenticated calls, Claude Code needs SOMETHING in
``ANTHROPIC_AUTH_TOKEN``, and codex needs an env_key variable to exist —
one constant satisfies all three without inventing key management.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GATEWAY_DIR = REPO_ROOT / "outputs" / "api_gateway"
PORT_POOL = tuple(range(4100, 4110))
MASTER_KEY = "sk-agentcanvas-gateway"
CODEX_KEY_ENV = "AC_GATEWAY_KEY"  # codex resolves its provider key from env


@dataclass(frozen=True)
class Route:
    alias: str            # model name the harness asks for
    litellm_model: str    # litellm slug the gateway routes to
    key_env: str          # vendor key variable, referenced as os.environ/<VAR>
    api_base: str | None = None


def _free_port() -> int:
    for port in PORT_POOL:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free gateway port in {PORT_POOL[0]}-{PORT_POOL[-1]}")


class LitellmGateway:
    def __init__(self, routes: list[Route], tag: str = "gateway") -> None:
        self.routes = routes
        self.tag = tag
        self.port: int | None = None
        self._proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _config_yaml(self) -> str:
        lines = ["model_list:"]
        for r in self.routes:
            lines += [
                f"  - model_name: {r.alias}",
                "    litellm_params:",
                f"      model: {r.litellm_model}",
                f"      api_key: os.environ/{r.key_env}",
            ]
            if r.api_base:
                lines.append(f"      api_base: {r.api_base}")
        lines += [
            "litellm_settings:",
            "  drop_params: true",
            "general_settings:",
            f"  master_key: {MASTER_KEY}",
        ]
        return "\n".join(lines) + "\n"

    def start(self, timeout: float = 120.0) -> int:
        GATEWAY_DIR.mkdir(parents=True, exist_ok=True)
        self.port = _free_port()
        cfg = GATEWAY_DIR / f"{self.tag}_{self.port}.yaml"
        cfg.write_text(self._config_yaml())
        log = (GATEWAY_DIR / f"{self.tag}_{self.port}.log").open("a")
        os.environ[CODEX_KEY_ENV] = MASTER_KEY
        # litellm has no __main__ — the proxy entry is a console script. The
        # gateway runs from its OWN env (ac-litellm-gw, latest litellm): the
        # agentcanvas env keeps 1.83.4 frozen for the mini direct path, and
        # 1.83.4's /v1/messages adapter mangles openai/responses/* routes
        # anyway. Override with $AC_GATEWAY_LITELLM; fall back to this env.
        gw_env_bin = Path.home() / "miniforge3" / "envs" / "ac-litellm-gw" / "bin" / "litellm"
        litellm_bin = os.environ.get("AC_GATEWAY_LITELLM") or (
            str(gw_env_bin) if gw_env_bin.exists()
            else str(Path(sys.executable).parent / "litellm"))
        self._proc = subprocess.Popen(  # noqa: S603
            [litellm_bin, "--config", str(cfg),
             "--host", "127.0.0.1", "--port", str(self.port)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"gateway died on startup (rc={self._proc.returncode}) — "
                    f"see {GATEWAY_DIR / f'{self.tag}_{self.port}.log'}")
            try:
                with urllib.request.urlopen(
                        f"{self.base_url}/health/liveliness", timeout=2):
                    return self.port
            except Exception:  # noqa: BLE001
                time.sleep(1)
        self.stop()
        raise RuntimeError(f"gateway failed to come up on :{self.port} in {timeout}s")

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    # ── harness-facing projections ──

    def anthropic_env(self) -> dict[str, str]:
        """Env for the SDK harness: Claude Code speaks anthropic-messages to us."""
        return {"ANTHROPIC_BASE_URL": self.base_url,
                "ANTHROPIC_AUTH_TOKEN": MASTER_KEY}

    def codex_args(self, provider_id: str = "acgw") -> list[str]:
        """-c overrides for codex exec: a custom openai-chat provider."""
        return [
            "-c", f'model_providers.{provider_id}.name = "AgentCanvas gateway"',
            "-c", f'model_providers.{provider_id}.base_url = "{self.base_url}/v1"',
            "-c", f'model_providers.{provider_id}.env_key = "{CODEX_KEY_ENV}"',
            "-c", f'model_providers.{provider_id}.wire_api = "chat"',
            "-c", f'model_provider = "{provider_id}"',
        ]

    def describe(self) -> dict:
        """Recorded into harness_inherent — the run must say it was gatewayed."""
        return {"backend": "litellm-gateway", "port": self.port,
                "routes": {r.alias: r.litellm_model for r in self.routes}}


def gateway_for_spec(spec, tag: str):
    """Start the wire-appropriate gateway when the cell asks for one.

    A cross-API cell carries ``("api_gateway", "litellm")`` plus
    ``("gateway_route", <litellm slug>)`` — the mini column's slug for the
    same model key. The alias the harness asks for is the cell's own
    model_id, so the harness-side model knob needs no special-casing.

    Wire selection: the sdk harness (anthropic-messages) gets the
    AnthropicShim — litellm proxy's /v1/messages path mangles
    openai/responses/* routes (see anthropic_shim.py) — while codex
    (openai-chat) gets the litellm proxy, whose chat path routes correctly.
    """
    extra = dict(spec.extra)
    if extra.get("api_gateway") != "litellm":
        return None
    from .providers import vendor_of
    route = str(extra.get("gateway_route") or "")
    vendor = vendor_of(route)
    if not route or vendor is None or not vendor.key_env:
        raise RuntimeError(
            f"api_gateway cell without a resolvable gateway_route: {route!r}")
    if tag == "sdk":
        from .anthropic_shim import AnthropicShim
        gw = AnthropicShim({spec.model_id: route}, tag=tag)
    else:
        gw = LitellmGateway([Route(spec.model_id, route, vendor.key_env)], tag=tag)
    gw.start()
    return gw
