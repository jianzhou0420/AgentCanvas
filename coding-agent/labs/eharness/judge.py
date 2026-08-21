"""V1 — the clean-context judge. One model call, zero env steps.

The mechanism is the *context*, not the model: the judge never walked the 40
steps, so it will not defend them (delegation-verification §5.2). Same model
as planner and sub-agent (2026-08-04 ruling) — tiers differ by context and
permissions, not weights.

Two rules from the spec carried here:
- the agent has the right to present evidence (its chosen frames ride along);
- a judge veto must never directly decide the outcome — the caller escalates
  (forces a look) instead of flipping the claim. ``judge()`` therefore only
  *reports*; policy stays in the wrapper/planner.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

import litellm
import requests


def _ollama_native(model_name: str,
                   model_kwargs: dict[str, Any] | None) -> tuple[str, str] | None:
    """(base_url, bare_model) when the judge can use ollama's native chat API.

    The native API accepts ``think: false`` — qwen3.5 then answers a
    perceptual yes/no in ~0.7s / ~35 tokens instead of free-running a
    3-8k-token reasoning chain (measured 2026-08-06: 17.6s → 0.66s on the
    same claim; the /no_think soft switch no longer exists in qwen3.5).
    litellm has no lever for this, so ollama judges bypass it.
    """
    prefix, _, bare = model_name.partition("/")
    if prefix in ("ollama", "ollama_chat") and bare:
        base = (model_kwargs or {}).get("api_base") or "http://127.0.0.1:11434"
        return base.rstrip("/"), bare
    return None


def _ollama_verdict(base: str, model: str, system: str, user_text: str,
                    images: list[bytes], *, think: bool = True) -> str:
    msg: dict[str, Any] = {"role": "user", "content": user_text}
    if images:
        msg["images"] = [base64.b64encode(p).decode() for p in images]
    resp = requests.post(f"{base}/api/chat", json={
        "model": model, "stream": False, "think": think,
        "messages": [{"role": "system", "content": system}, msg],
    }, timeout=600)
    resp.raise_for_status()
    return (resp.json().get("message") or {}).get("content") or ""


def _claude_subscription(model_name: str) -> str | None:
    """The bare model id when the judge should ride the Claude SDK
    subscription channel. User ruling 2026-08-14: ONE model per run — a
    claude nav model judges through its own subscription auth, never a
    local side-model and never an API key. Accepts "claude-…" and
    "anthropic/claude-…"."""
    prefix, _, bare = (model_name or "").partition("/")
    name = bare or prefix
    return name if name.startswith("claude") else None


def _sdk_verdict(model: str, system: str, user_text: str,
                 images: list[bytes], *, timeout_s: float = 90.0,
                 think: bool = False) -> str:
    """One tool-less subscription one-shot; images ride as raw base64
    content blocks (the SDK passes them through the CLI verbatim — the
    exact shape the adapter's bootstrap message uses). The bridge calls
    judges from inside its MCP event loop, so when a loop is already
    running on this thread the shot gets a dedicated thread with a loop
    of its own. stage3 P0-2: thinking is explicitly DISABLED unless the
    caller asks — a perceptual yes/no needs no reasoning chain, and the
    subscription judges used to free-run the SDK's default."""
    import asyncio

    async def _one() -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            TextBlock,
        )
        opts = ClaudeAgentOptions(
            system_prompt=system, tools=[], setting_sources=[],
            max_turns=1, model=model, permission_mode="bypassPermissions",
            # stage3 §9: a judge one-shot is not an experiment record — the
            # harness logs the verdict; the CLI must not grow .claude/projects
            extra_args={"no-session-persistence": None},
            **({} if think else {"thinking": {"type": "disabled"}}))
        blocks: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        blocks += [{"type": "image",
                    "source": {"type": "base64", "media_type": "image/png",
                               "data": base64.b64encode(p).decode()}}
                   for p in images]

        async def _msg():
            yield {"type": "user",
                   "message": {"role": "user", "content": blocks},
                   "parent_tool_use_id": None}

        texts: list[str] = []
        async with ClaudeSDKClient(options=opts) as client:
            await client.query(_msg())
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for b in message.content:
                        if isinstance(b, TextBlock):
                            texts.append(b.text)
        return "\n".join(texts)

    def _run() -> str:
        return asyncio.run(asyncio.wait_for(_one(), timeout=timeout_s))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()

JUDGE_SYSTEM = (
    "You are a strict verifier for a home robot. You are given a CLAIM, the "
    "task context, and one or more camera images (newest last). Judge ONLY "
    "from the IMAGES. The text context may contain the robot's own progress "
    "claims — those are the thing under test, NOT evidence; ignore any "
    "statement that the task is done. "
    "For nearness claims: supported=true ONLY if the target appears VERY "
    "close (within ~3 meters — almost arrived, filling much of the view). "
    "A target that is visible but at medium or far distance = false. "
    'Answer with a single JSON object: {"supported": true/false, '
    '"reason": "<one short sentence>"}. '
    "If the images are insufficient to tell, answer supported=false with "
    "reason starting with 'insufficient'."
)


def _image_part(png: bytes) -> dict[str, Any]:
    b64 = base64.b64encode(png).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


ROUTE_SYSTEM = (
    "You compare a navigation instruction against the route a robot reports "
    "having taken. The report is the robot's own account — you are checking "
    "its CONSISTENCY with the instruction, not its truth. Answer with one "
    'JSON object: {"supported": true/false, "reason": "<one short '
    'sentence>"}. supported=false if the reported route skips, reverses, or '
    "contradicts the instructed route; true if it plausibly follows it. If "
    "the report is too thin to compare, answer true (do not punish silence)."
)


def judge_route(instruction: str, route_report: str, *,
                model_name: str, model_kwargs: dict[str, Any] | None = None,
                ) -> tuple[bool | None, str]:
    """Text-only consistency check: instructed route vs the robot's own
    story of how it went. A human's recovery test — if even your own account
    of the walk doesn't match the instruction, you are lost regardless of
    what the current view shows."""
    user_text = (f"INSTRUCTION:\n{instruction}\n\n"
                 f"ROBOT'S ROUTE REPORT:\n{route_report}")
    kw = dict(model_kwargs or {})
    # judge_think — stage3 P0-2: OFF is the default everywhere (verdicts
    # measured think-equivalent, 2026-08-06); "1" re-enables for ablation
    think = bool(kw.pop("judge_think", False))
    model_kwargs = kw
    try:
        sub = _claude_subscription(model_name)
        native = _ollama_native(model_name, model_kwargs)
        if sub:
            text = _sdk_verdict(sub, ROUTE_SYSTEM, user_text, [],
                                think=think)
        elif native:
            text = _ollama_verdict(*native, ROUTE_SYSTEM, user_text, [],
                                   think=think)
        else:
            resp = litellm.completion(
                model=model_name,
                messages=[{"role": "system", "content": ROUTE_SYSTEM},
                          {"role": "user", "content": user_text}],
                **(model_kwargs or {}),
            )
            text = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        return None, f"route judge unavailable: {exc!r}"
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            return bool(data.get("supported")), str(data.get("reason", ""))[:200]
        except (ValueError, TypeError):
            pass
    return None, f"unparseable route verdict: {text.strip()[:160]}"


STOP_SYSTEM = (
    "You are a strict verifier for a home robot that has just asked to STOP "
    "and declare its navigation goal reached. You are given the goal and "
    "one or more camera images (newest last). Judge ONLY from the IMAGES. "
    "supported=true ONLY if the goal appears VERY close (within ~3 meters "
    "— almost arrived, filling much of the view); visible at medium or far "
    "distance = false. "
    'Answer with a single JSON object: {"supported": true/false, "reason": '
    '"<one short sentence: what the images show about where the robot '
    'actually is>", "advice": "<one short sentence: if not there, what the '
    "robot should DO next, judged from the images — e.g. 'the bar corner "
    "is visible ahead on your left; walk a few steps toward it until it "
    "fills your view, then stop'>\"}. "
    "If the images are insufficient, answer supported=false, reason "
    "starting with 'insufficient', advice 'look around to re-establish "
    "where the goal is'."
)


def judge_stop(claim: str, context: str, images: list[bytes], *,
               model_name: str, model_kwargs: dict[str, Any] | None = None,
               ) -> tuple[bool | None, str, str]:
    """The pre-stop verdict WITH advice (用户 2026-08-06: 导航器叫停有它的
    理由，你否决也有你的理由——那就给个建议，具体该怎么办，否则它怎么会
    知道). Returns (supported, reason, advice); supported=None = judge
    failed — never a pass or a fail. The advice may compel WORK (look, walk
    closer); the conclusion stays the robot's."""
    user_text = (f"CLAIM: {claim}\n\nGOAL:\n{context}\n\n"
                 f"Images follow (newest last).")
    kw = dict(model_kwargs or {})
    think = bool(kw.pop("judge_think", False))
    try:
        sub = _claude_subscription(model_name)
        native = _ollama_native(model_name, kw)
        if sub:
            text = _sdk_verdict(sub, STOP_SYSTEM, user_text, images[-3:],
                                think=think)
        elif native:
            text = _ollama_verdict(*native, STOP_SYSTEM, user_text,
                                   images[-3:], think=think)
        else:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            content += [_image_part(png) for png in images[-3:]]
            resp = litellm.completion(
                model=model_name,
                messages=[{"role": "system", "content": STOP_SYSTEM},
                          {"role": "user", "content": content}],
                **kw,
            )
            text = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        return None, f"judge unavailable: {exc!r}", ""
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            return (bool(data.get("supported")),
                    str(data.get("reason", ""))[:200],
                    str(data.get("advice", ""))[:200])
        except (ValueError, TypeError):
            pass
    return None, f"unparseable verdict: {text.strip()[:160]}", ""


MILESTONE_SYSTEM = (
    "You are a strict route verifier for a home robot. You are given ONE "
    "sub-instruction from a navigation route and a sequence of camera views "
    "from the robot's walk so far, in chronological order (oldest first). "
    "Decide whether the walk EVIDENCES that the robot actually PERFORMED "
    "this sub-instruction. Be strict about what 'performed' means: "
    "a target merely VISIBLE (even nearby) was not walked; being NEAR or "
    "ADJACENT to the described scene is NOT performing it. 'Walk between X "
    "and Y' is evidenced only if successive views show X on one side and Y "
    "on the OTHER side while the robot advances through the gap. 'Go past "
    "X' is evidenced only if X moves from ahead to beside/behind across "
    "the views. If the sequence is equally compatible with the robot "
    "merely approaching or standing near the scene, answer false. "
    "When the FULL ROUTE is given, the segments happen IN ORDER: the "
    "queried segment's scene comes AFTER the earlier segments' scenes — "
    "a look-alike structure at the route's beginning cannot satisfy a "
    "later segment. "
    'Answer with a single JSON object: {"evidenced": true/false, "reason": '
    '"<one short sentence>"}. If the views are too few or unclear to tell, '
    "answer evidenced=false — for a route claim, absence of evidence "
    "blocks."
)


def judge_milestone(sub_instruction: str, images: list[bytes], *,
                    model_name: str,
                    model_kwargs: dict[str, Any] | None = None,
                    route_context: str = "",
                    ) -> tuple[bool | None, str]:
    """Did this segment of the route actually happen? (用户 2026-08-06: 费那么
    大劲把 instruction 分成几段，每段至少得检查一下。) Judged from the
    harness's own frame records; the model's narrative plays no part. The
    silence-is-innocent rule of judge_route is INVERTED here: no evidence =
    not evidenced (v2.9 EP0 stopped with done=[] having skipped 'walk
    between the bar and chairs' — thin narrative bought a free pass)."""
    user_text = ((f"FULL ROUTE (context):\n{route_context}\n\n"
                  if route_context else "")
                 + f"VERIFY THIS SEGMENT: {sub_instruction}\n\n"
                 f"{len(images[-8:])} views follow, chronological "
                 "(oldest first).")
    kw = dict(model_kwargs or {})
    think = bool(kw.pop("judge_think", False))
    try:
        sub = _claude_subscription(model_name)
        native = _ollama_native(model_name, kw)
        if sub:
            text = _sdk_verdict(sub, MILESTONE_SYSTEM, user_text,
                                images[-8:], think=think)
        elif native:
            text = _ollama_verdict(*native, MILESTONE_SYSTEM, user_text,
                                   images[-8:], think=think)
        else:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            content += [_image_part(png) for png in images[-8:]]
            resp = litellm.completion(
                model=model_name,
                messages=[{"role": "system", "content": MILESTONE_SYSTEM},
                          {"role": "user", "content": content}],
                **kw,
            )
            text = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        return None, f"milestone judge unavailable: {exc!r}"
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            return (bool(data.get("evidenced")),
                    str(data.get("reason", ""))[:200])
        except (ValueError, TypeError):
            pass
    return None, f"unparseable milestone verdict: {text.strip()[:160]}"


def judge(claim: str, context: str, images: list[bytes], *,
          model_name: str, model_kwargs: dict[str, Any] | None = None,
          ) -> tuple[bool | None, str]:
    """Returns (supported, reason). supported=None = judge itself failed
    (network, parse) — callers must treat None as 'no verdict', never as a
    pass or a fail."""
    user_text = (f"CLAIM: {claim}\n\nCONTEXT:\n{context}\n\n"
                 f"Images follow (newest last).")
    kw = dict(model_kwargs or {})
    think = bool(kw.pop("judge_think", False))
    model_kwargs = kw
    try:
        sub = _claude_subscription(model_name)
        native = _ollama_native(model_name, model_kwargs)
        if sub:
            text = _sdk_verdict(sub, JUDGE_SYSTEM, user_text, images[-3:],
                                think=think)
        elif native:
            text = _ollama_verdict(*native, JUDGE_SYSTEM, user_text,
                                   images[-3:], think=think)
        else:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            content += [_image_part(png) for png in images[-3:]]
            resp = litellm.completion(
                model=model_name,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": content}],
                **(model_kwargs or {}),
            )
            text = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 — a broken judge must not kill a run
        return None, f"judge unavailable: {exc!r}"
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            return bool(data.get("supported")), str(data.get("reason", ""))[:200]
        except (ValueError, TypeError):
            pass
    # lenient fallback for small models that answer in prose
    low = text.lower()
    if "supported" in low and "true" in low:
        return True, text.strip()[:200]
    if any(w in low for w in ("not supported", "false", "insufficient", "no")):
        return False, text.strip()[:200]
    return None, f"unparseable verdict: {text.strip()[:160]}"
