"""SoloModel — the single-loop model layer: ephemeral state, event stubs, L2.

The solo paradigm's one new organ (solo-harness §5). Three moves, all inside
``_prepare_messages_for_api`` so nothing is ever persisted:

1. **Ephemeral state message** — the state block render is appended as the
   LAST message of every request (recency helps small models; rebuilt each
   call, never enters the stored history — CC's prependUserContext, mirrored).
2. **Event stubs (L0)** — elided frames become index entries, not holes.
   Mapping is position-independent: the wrapper tags every recorded frame
   with a ``[frame#N]`` text part right after the image, so the stub can name
   the frame it replaced and hand the model a working ``recall(N)`` handle.
   Images without a marker (recall re-attachments) get a generic stub.
3. **L2 compaction** — past a token estimate threshold, everything between
   the head (system + task) and a recent tail is dropped from the request;
   a boundary message re-attaches the top keyframes with captions. The state
   block IS the summary (maintained incrementally, monotone) — no model call,
   no summary-of-summary degradation. Recomputed per call from the full
   stored history: stateless, no drift, disk keeps everything.
"""

from __future__ import annotations

import json
import re
import time as _time
from typing import Any, Callable

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks

from model import NavToolsModel, NavToolsModelConfig  # harnesses/mini on sys.path

IMG_TOKENS = 700          # rough per-image cost in the estimate (qwen-vl scale)
STUB_TOKENS = 30
_MARKER = re.compile(r"^\[frame#(\d+)( auto)?\]$")


class SoloModelConfig(NavToolsModelConfig):
    history_frames: int = 12
    """§3.3: uniform history VISUAL CYCLES (CURRENT and the map are separate;
    the hard total is 14 images, enforced by the compiler)."""
    recent_turns: int = 6
    """(legacy_l2 only) complete cycles kept verbatim by the old cut."""
    legacy_l2: bool = False
    """Opt-in ONLY (ablation): the pre-§12 estimate-triggered L2 cut with
    [COMPACTED] boundary + keyframe re-attach + image window. The default
    path is the typed assembly in eharness.context — deterministic slots,
    not a token estimate."""
    manifest: bool = True
    """§12.9: append one context_manifest.jsonl line per request."""
    compact_at: int = 12000
    """(legacy_l2 only) L2 trigger: estimated request tokens above this → cut."""
    tail_msgs: int = 24
    """Verbatim tail kept across an L2 cut (extended back to a turn boundary)."""
    tail_budget: int = 7000
    """…and a SIZE bound on that tail, in estimated tokens.

    A count alone is not a budget. Adding one sentence per tool result pushed a
    run from 3 compactions to 21 and from a 13.1k-token peak to 16.3k — the cut
    kept dropping messages and could no longer reach the threshold, because the
    24 it is required to retain had simply become fatter. At that size the 9B
    started emitting malformed tool-call XML and the episode died, three
    identical retries in a row. Whichever bound binds first now wins."""
    reattach_k: int = 3
    """Keyframes re-attached at the boundary (CC's top-5 files, embodied)."""


class SoloModel(NavToolsModel):
    """NavToolsModel with the solo context discipline. ``harness`` is the
    HarnessedToolset (frames + state); ``emit`` mirrors compact events into
    the EventSink."""

    def __init__(self, *, harness: Any,
                 emit: Callable[[str, dict], None] | None = None,
                 **kwargs: Any) -> None:
        # NavToolsModel.__init__ hardcodes its own config_class, so replicate
        # its two-line body against SoloModelConfig instead of calling it.
        LitellmModel.__init__(self, config_class=SoloModelConfig, **kwargs)
        self._tool_names = [t["name"] for t in self.config.tools]
        self._litellm_tools = [
            {"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}}
            for t in self.config.tools
        ]
        self._h = harness
        self._emit_cb = emit or (lambda kind, payload: None)

    # ── the whole organ lives in this override ──

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        msgs = [{k: v for k, v in m.items() if k != "extra"} for m in messages]
        if self.config.legacy_l2:
            msgs = self._scrub_stale(msgs)             # 持续思考, not a log
            msgs = self._apply_image_window(msgs)      # window + event stubs
            msgs = self._l2_cut(msgs)                  # boundary cut if over
        else:
            # The cycle compiler (multiturn revision §3): whole cycles kept
            # in place with their ORIGINAL reasoning; unselected cycles leave
            # the payload entirely; 12+1+1 visual slots, 14 enforced. Stale
            # budget counters still go (actively misleading); reasoning stays.
            msgs = self._scrub_stale(msgs, keep_prose_n=None,
                                     keep_reasoning=True)
            before = self._h.state.snapshot()
            from eharness.context import ContextPolicy, assemble
            msgs, manifest = assemble(
                msgs, self._h.frames,
                ContextPolicy(history_cycles=self.config.history_frames),
                live_dir=getattr(self._h, "live_dir", None))
            self._h.state.assert_monotone(before, self._h.state.snapshot())
            self._last_manifest = manifest
        msgs = _reorder_anthropic_thinking_blocks(msgs)
        if self.config.set_cache_control == "default_end":
            msgs = self._set_cache_control_multipart(msgs)
        state_msg = self._state_message()
        if not self.config.legacy_l2 and self._last_manifest is not None:
            self._last_manifest["state_chars"] = len(
                state_msg["content"][0]["text"])
        return msgs + [state_msg]                      # ephemeral, always last

    # ── §12.9 · the manifest: what the provider was actually shown ──

    _last_manifest: dict | None = None
    _n_queries: int = 0

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        t0 = _time.time()
        try:
            out = super().query(messages, **kwargs)
        except Exception as e:
            # A FormatError (bad tool call) is raised AFTER the provider call
            # succeeded and was billed — a request that cost tokens must not
            # be invisible in the audit trail (§12.9).
            self._write_manifest(t0, usage=None, error=type(e).__name__)
            raise
        usage = None
        try:
            usage = ((out.get("extra") or {}).get("response") or {}
                     ).get("usage") or None
        except AttributeError:
            usage = None
        self._write_manifest(t0, usage=usage)
        return out

    def _write_manifest(self, t0: float, *, usage: dict | None,
                        error: str | None = None) -> None:
        self._n_queries += 1
        if not self.config.manifest or self.config.legacy_l2:
            return
        m = dict(self._last_manifest or {})
        self._last_manifest = None
        m.update({
            "turn": self._n_queries,
            # real numbers or null — never an estimate dressed as actual
            "provider_input_tokens": (usage or {}).get("prompt_tokens"),
            "provider_output_tokens": (usage or {}).get("completion_tokens"),
            "request_latency_ms": round((_time.time() - t0) * 1000, 1),
            **({"error": error} if error else {}),
        })
        live = getattr(self._h, "live_dir", None)
        if live is not None:
            try:
                live.mkdir(parents=True, exist_ok=True)
                with (live / "context_manifest.jsonl").open("a") as fh:
                    fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            except OSError:
                pass

    # ── 0 · stale-turn hygiene (用户 2026-08-07) ──

    _BUDGET_KEYS = ("executed", "requested", "steps_taken_total",
                    "steps_remaining_approx", "tool_calls_used",
                    "tool_calls_remaining", "BUDGET_WARNING")

    @classmethod
    def _scrub_stale(cls, messages: list[dict],
                     keep_prose_n: int | None = 2,
                     keep_reasoning: bool = False) -> list[dict]:
        """Two context-hygiene rules, recomputed per call (stored history
        untouched):
        (1) step/budget counters are only truthful on the NEWEST tool
            result — every stale copy is stripped (只有最后一步能看得清，
            其他都可以清理掉);
        (2) a person navigating does not re-read their past thinking — the
            running thought lives in the STATE block, so reasoning_content
            is never sent back at all and older turns' narration is
            blanked; the tool calls stay (they are the action record), and
            the newest `keep_prose_n` turns keep their prose (the live
            train of thought). ``keep_prose_n=None`` skips the blanking
            entirely — the §12 path keeps its six turns INTACT and drops
            everything older whole, so there is nothing left to blank."""
        out = [dict(m) for m in messages]
        last_tool = max((i for i, m in enumerate(out)
                         if m.get("role") == "tool"), default=None)
        assist = [i for i, m in enumerate(out) if m.get("role") == "assistant"]
        keep_prose = (set(assist) if keep_prose_n is None
                      else set(assist[-keep_prose_n:]))
        for i, m in enumerate(out):
            if m.get("role") == "assistant":
                if not keep_reasoning:
                    # §3.5: the new path RETAINS original reasoning on kept
                    # cycles (unselected cycles vanish whole downstream);
                    # only the legacy cut still scrubs it unconditionally
                    m.pop("reasoning_content", None)
                if (i not in keep_prose and m.get("tool_calls")
                        and isinstance(m.get("content"), str)
                        and m["content"].strip()):
                    m["content"] = ""
            elif m.get("role") == "tool" and i != last_tool:
                m["content"] = cls._strip_budget(m.get("content"))
        return out

    @classmethod
    def _strip_budget(cls, content: Any) -> Any:
        def clean(text: str) -> str:
            if not text.lstrip().startswith("{"):
                return text
            try:
                d = json.loads(text)
            except ValueError:
                return text
            if not isinstance(d, dict):
                return text
            for k in cls._BUDGET_KEYS:
                d.pop(k, None)
            return json.dumps(d)
        if isinstance(content, str):
            return clean(content)
        if isinstance(content, list):
            return [{**p, "text": clean(str(p.get("text", "")))}
                    if isinstance(p, dict) and p.get("type") == "text" else p
                    for p in content]
        return content

    # ── 1 · ephemeral state ──

    def _state_message(self) -> dict[str, Any]:
        return {"role": "user", "content": [{
            "type": "text",
            "text": ("[CURRENT STATE — maintained by the harness, rebuilt "
                     "every turn; update it via the state fields on your "
                     "action calls]\n" + self._h.state.render())}]}

    # ── 2 · window with event stubs (overrides the constant-stub version) ──

    def _apply_image_window(self, messages: list[dict]) -> list[dict]:
        k = self.config.image_window
        if k <= 0:
            return messages
        # first pass: every image's (marker idx, auto flag), in order — auto
        # frames are the transient after-move views: shown the turn they
        # arrive, replaced by a stub on every later turn (看完扔掉), and they
        # never consume the K-frame budget
        flags: list[tuple[int | None, bool]] = []
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for i, p in enumerate(content):
                if isinstance(p, dict) and p.get("type") == "image_url":
                    marker, auto = None, False
                    if i + 1 < len(content) and isinstance(content[i + 1], dict):
                        mm = _MARKER.match(str(content[i + 1].get("text", "")))
                        if mm:
                            marker, auto = int(mm.group(1)), bool(mm.group(2))
                    flags.append((marker, auto))
        total = len(flags)
        if total == 0:
            return messages
        def _is_crisis(mk):
            fr_list = self._h.frames.frames
            return (mk is not None and 0 <= mk < len(fr_list)
                    and getattr(fr_list[mk], "crisis", False))
        non_auto = [j for j, (mk, a) in enumerate(flags)
                    if not a and not _is_crisis(mk)]
        keep_set = set(non_auto[-k:])
        keep_set.add(total - 1)          # the globally newest image always shows
        if total <= k and not any(a for _mk, a in flags):
            return messages
        out: list[dict] = []
        seen = 0
        for msg in messages:
            content = msg.get("content")
            if not (isinstance(content, list) and any(
                    isinstance(p, dict) and p.get("type") == "image_url"
                    for p in content)):
                out.append(msg)
                continue
            new_parts: list[Any] = []
            i = 0
            while i < len(content):
                part = content[i]
                if isinstance(part, dict) and part.get("type") == "image_url":
                    j = seen
                    seen += 1
                    marker, _auto = flags[j]
                    has_marker = False
                    if i + 1 < len(content) and isinstance(content[i + 1], dict):
                        has_marker = bool(_MARKER.match(
                            str(content[i + 1].get("text", ""))))
                    if j in keep_set:
                        new_parts.append(part)
                        if has_marker:
                            new_parts.append(content[i + 1])
                    elif _is_crisis(marker):
                        pass  # crisis interior: no image, no stub — the
                              # lesson line in the state carries the memory
                    else:
                        stub = (self._h.frames.stub_text(marker)
                                if marker is not None
                                else "[recalled frame elided]")
                        new_parts.append({"type": "text", "text": stub})
                    i += 2 if has_marker else 1
                    continue
                new_parts.append(part)
                i += 1
            out.append({**msg, "content": new_parts})
        return out

    # ── 3 · L2 cut ──

    @staticmethod
    def _estimate(messages: list[dict]) -> int:
        toks = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                toks += len(content) // 3
            elif isinstance(content, list):
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") == "image_url":
                        toks += IMG_TOKENS
                    else:
                        toks += len(str(p.get("text", ""))) // 3
            for tc in (m.get("tool_calls") or []):
                toks += len(str(tc)) // 3
        return toks

    def _l2_cut(self, messages: list[dict]) -> list[dict]:
        before = self._h.state.snapshot()
        est = self._estimate(messages)
        if est <= self.config.compact_at or len(messages) < 6:
            return messages
        # head = system + the task message; tail = recent turns, extended back
        # to an assistant boundary so tool_use/tool_result pairs stay whole
        head_end = 1
        while head_end < len(messages) and messages[head_end].get("role") in (
                "system", "user"):
            head_end += 1
            break   # system + exactly one task message
        start = max(head_end, len(messages) - self.config.tail_msgs)
        while start > head_end and messages[start].get("role") not in (
                "assistant", "user"):
            start -= 1
        # The count has been honoured; now honour the SIZE. Walk forward one
        # turn boundary at a time while the tail is still too heavy — forward,
        # and boundary to boundary, so the alignment above is never undone and
        # tool_use/tool_result pairs are never split.
        while (start < len(messages) - 2
               and self._estimate(messages[start:]) > self.config.tail_budget):
            start += 1
            while (start < len(messages) - 2
                   and messages[start].get("role") not in ("assistant", "user")):
                start += 1
        if start <= head_end:
            return messages     # nothing to cut
        dropped = messages[head_end:start]
        boundary = self._boundary_message(len(dropped))
        kept = messages[:head_end] + [boundary] + messages[start:]
        self._emit_cb("compact", {
            "est_tokens": est, "dropped_msgs": len(dropped),
            "kept_msgs": len(kept), "tail_tokens": self._estimate(messages[start:]),
            "reattached": min(self.config.reattach_k,
                              len(self._h.frames.keyframes()))})
        # pre_compact tripwire: the cut must not have touched the state block.
        # It used to compare a snapshot with a second snapshot taken on the very
        # next line — itself against itself, which can never fail. The `before`
        # has to be taken BEFORE the cut for this to be a tripwire at all.
        self._h.state.assert_monotone(before, self._h.state.snapshot())
        return kept

    def _boundary_message(self, n_dropped: int) -> dict[str, Any]:
        parts: list[dict[str, Any]] = [{
            "type": "text",
            "text": (f"[COMPACTED: {n_dropped} earlier messages removed — "
                     "your conclusions live in the state header; key views "
                     "re-attached below; recall(<frame#>) retrieves anything "
                     "else]")}]
        digest = self._h.frames.span_digest()
        if digest:
            parts.append({"type": "text",
                          "text": "what happened between the key views:\n"
                                  + "\n".join(digest)})
        for fr in self._h.frames.keyframes()[-self.config.reattach_k:]:
            png = self._h.frames.png_bytes(fr.idx)
            if png is None:
                continue
            parts.append({"type": "text",
                          "text": f"keyframe frame#{fr.idx}: "
                                  + (" · ".join(fr.events[-2:]) or "view")})
            import base64
            b64 = base64.b64encode(png).decode()
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/png;base64,{b64}"}})
        return {"role": "user", "content": parts}
