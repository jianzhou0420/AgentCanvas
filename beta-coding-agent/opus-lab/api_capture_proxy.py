"""Logging forward-proxy for the Anthropic API — capture what the CLI actually
sends the model. Point a session at it with ANTHROPIC_BASE_URL=http://127.0.0.1:PORT;
every request body (system prompt, message history, tool schemas) is appended to
a capture file, then forwarded VERBATIM to the real API with streaming preserved.

This is how we see the *final bytes the API receives*, after any CLI-side
assembly — the one thing session_inputs (driver-side) can't show.

Auth passes through untouched (the subscription OAuth bearer the CLI sends), so
this works with subscription auth; only the Host is rewritten to the upstream.

Usage:  python api_capture_proxy.py --port 8900 --out /path/to/api_capture.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

import aiohttp
from aiohttp import web

UPSTREAM = "https://api.anthropic.com"
# hop-by-hop / length headers we must not blindly forward in either direction;
# accept-encoding is dropped on the way UP so the upstream replies uncompressed
# (simpler, readable passthrough).
DROP_REQ = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}
DROP_RESP = {"content-length", "connection", "transfer-encoding", "content-encoding"}


def make_app(out_path: str) -> web.Application:
    out = open(out_path, "a")

    async def handle(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        try:
            parsed = json.loads(body) if body else None
        except ValueError:
            parsed = None
        entry = {
            "t": round(time.time(), 2),
            "method": request.method,
            "path": str(request.rel_url),
            "headers": {
                k: v for k, v in request.headers.items()
                if k.lower() not in ("authorization", "x-api-key")
            },
            "body": parsed if parsed is not None else body.decode("utf-8", "replace"),
        }
        out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        out.flush()

        fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP_REQ}
        timeout = aiohttp.ClientTimeout(total=900)
        async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as sess:
            async with sess.request(
                request.method,
                UPSTREAM + str(request.rel_url),
                headers=fwd_headers,
                data=body,
            ) as up:
                resp = web.StreamResponse(
                    status=up.status,
                    headers={k: v for k, v in up.headers.items() if k.lower() not in DROP_RESP},
                )
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp

    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--out", default="api_capture.jsonl")
    args = ap.parse_args()
    print(
        f"[proxy] capturing -> {args.out} | forwarding -> {UPSTREAM} | listening :{args.port}",
        flush=True,
    )
    web.run_app(make_app(args.out), host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
