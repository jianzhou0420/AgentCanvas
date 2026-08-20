"""anthropic-wire shim — /v1/messages for cross-API seats, without litellm proxy.

Why not the litellm proxy for this wire: its /v1/messages path (and the
underlying ``litellm.anthropic_messages``) resolves the provider EARLY and
hits the chat-completions handler with the literal remainder of the slug —
an ``openai/responses/gpt-5.6`` route reaches OpenAI as the nonexistent
model ``responses/gpt-5.6`` (reproduced on 1.83.4 and 1.97.0). The plain
``litellm.acompletion`` dispatch handles the same slug correctly, so this
shim does what the proxy should: translate the anthropic-format request with
litellm's own adapter, call ``acompletion`` directly, translate back.

Streaming: Claude Code always streams. Rather than re-implementing litellm's
per-chunk translation (content-block-index bookkeeping), the shim completes
the call NON-streaming and emits one protocol-valid anthropic SSE sequence
from the finished response — message_start, one start/delta/stop triple per
content block, message_delta with stop_reason and usage, message_stop. For
eval turns this only trades time-to-first-token, not correctness.

Runs as its own subprocess (see ``AnthropicShim`` at the bottom — the
lifecycle mirrors LitellmGateway: prepare() starts, finalize() stops).
Config rides two env vars: AC_SHIM_ROUTES (JSON alias→litellm slug) and
AC_SHIM_KEY (bearer token Claude Code presents via ANTHROPIC_AUTH_TOKEN).
"""

# NB: no `from __future__ import annotations` here — FastAPI resolves the
# handler annotations at runtime, and with postponed annotations the locally
# imported Request type stops resolving (the parameter degrades to a query
# field and every call 422s). Python 3.10 handles the `X | None` forms natively.
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIM_DIR = REPO_ROOT / "outputs" / "api_gateway"
PORT_POOL = tuple(range(4110, 4120))
MASTER_KEY = "sk-agentcanvas-gateway"


# ── server side (runs under `python -m api.anthropic_shim`) ──

def _build_app():
    import litellm
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    routes: dict[str, str] = json.loads(os.environ["AC_SHIM_ROUTES"])
    key = os.environ.get("AC_SHIM_KEY", MASTER_KEY)
    adapter = LiteLLMAnthropicMessagesAdapter()
    app = FastAPI()

    def _authed(request: Request) -> bool:
        auth = request.headers.get("authorization", "")
        return key in (request.headers.get("x-api-key"), auth.removeprefix("Bearer ").strip())

    def _sse(events: list[tuple[str, dict]]):
        for name, data in events:
            yield f"event: {name}\ndata: {json.dumps(data)}\n\n"

    def _synthesize_sse(anth: dict) -> list[tuple[str, dict]]:
        """One valid SSE sequence from a COMPLETE anthropic response."""
        usage = anth.get("usage") or {}
        events: list[tuple[str, dict]] = [("message_start", {
            "type": "message_start",
            "message": {**anth, "content": [], "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": usage.get("input_tokens", 0),
                                  "output_tokens": 0}}})]
        for i, block in enumerate(anth.get("content") or []):
            btype = block.get("type")
            if btype == "tool_use":
                start = {"type": "tool_use", "id": block["id"],
                         "name": block["name"], "input": {}}
                delta = {"type": "input_json_delta",
                         "partial_json": json.dumps(block.get("input") or {})}
            elif btype == "thinking":
                start = {"type": "thinking", "thinking": ""}
                delta = {"type": "thinking_delta",
                         "thinking": block.get("thinking", "")}
            else:
                start = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": block.get("text", "")}
            events += [
                ("content_block_start", {"type": "content_block_start",
                                         "index": i, "content_block": start}),
                ("content_block_delta", {"type": "content_block_delta",
                                         "index": i, "delta": delta}),
                ("content_block_stop", {"type": "content_block_stop", "index": i}),
            ]
        events += [
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": anth.get("stop_reason"),
                                         "stop_sequence": anth.get("stop_sequence")},
                               "usage": {"output_tokens": usage.get("output_tokens", 0)}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        return events

    @app.get("/health/liveliness")
    async def liveliness():
        return {"status": "ok", "routes": routes}

    @app.post("/v1/messages")
    async def messages(request: Request):
        if not _authed(request):
            raise HTTPException(401, "bad shim key")
        body = await request.json()
        alias = body.get("model", "")
        if alias not in routes:
            raise HTTPException(404, f"unknown alias {alias!r}; routes: {list(routes)}")
        wants_stream = bool(body.pop("stream", False))
        openai_req, tool_name_mapping = adapter.translate_anthropic_to_openai(body)
        openai_req["model"] = routes[alias]
        openai_req.pop("stream", None)
        resp = await litellm.acompletion(**openai_req, drop_params=True)
        anth = adapter.translate_openai_response_to_anthropic(resp, tool_name_mapping)
        anth = anth if isinstance(anth, dict) else anth.model_dump(exclude_none=False)
        anth["model"] = alias  # the caller asked for the alias, echo it back
        if not wants_stream:
            return JSONResponse(anth)
        return StreamingResponse(_sse(_synthesize_sse(anth)),
                                 media_type="text/event-stream")

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        if not _authed(request):
            raise HTTPException(401, "bad shim key")
        body = await request.json()
        alias = body.get("model", "")
        try:
            n = litellm.token_counter(model=routes.get(alias, alias),
                                      messages=body.get("messages") or [])
        except Exception:  # noqa: BLE001 — an estimate, never a blocker
            n = sum(len(json.dumps(m)) // 4 for m in body.get("messages") or [])
        return JSONResponse({"input_tokens": int(n)})

    return app


def main() -> None:
    import uvicorn
    port = int(sys.argv[sys.argv.index("--port") + 1])
    uvicorn.run(_build_app(), host="127.0.0.1", port=port, log_level="warning")


# ── client side (imported by the harness adapters) ──

class AnthropicShim:
    """Same lifecycle surface as LitellmGateway, for the anthropic wire."""

    def __init__(self, routes: dict[str, str], tag: str = "shim") -> None:
        self.routes = routes
        self.tag = tag
        self.port: int | None = None
        self._proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 60.0) -> int:
        SHIM_DIR.mkdir(parents=True, exist_ok=True)
        for port in PORT_POOL:
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", port))
                except OSError:
                    continue
                self.port = port
                break
        else:
            raise RuntimeError(f"no free shim port in {PORT_POOL[0]}-{PORT_POOL[-1]}")
        log = (SHIM_DIR / f"{self.tag}_{self.port}.log").open("a")
        self._proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "core.api.anthropic_shim", "--port", str(self.port)],
            cwd=str(REPO_ROOT / "coding-agent"),
            env={**os.environ, "AC_SHIM_ROUTES": json.dumps(self.routes),
                 "AC_SHIM_KEY": MASTER_KEY},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"shim died on startup (rc={self._proc.returncode}) — "
                    f"see {SHIM_DIR / f'{self.tag}_{self.port}.log'}")
            try:
                with urllib.request.urlopen(
                        f"{self.base_url}/health/liveliness", timeout=2):
                    return self.port
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"shim failed to come up on :{self.port} in {timeout}s")

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def anthropic_env(self) -> dict[str, str]:
        return {"ANTHROPIC_BASE_URL": self.base_url,
                "ANTHROPIC_AUTH_TOKEN": MASTER_KEY}

    def describe(self) -> dict:
        return {"backend": "anthropic-shim (in-process litellm acompletion)",
                "port": self.port, "routes": dict(self.routes),
                "streaming": "synthesized SSE from a completed response"}


if __name__ == "__main__":
    main()
