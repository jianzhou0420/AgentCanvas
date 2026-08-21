#!/usr/bin/env python
"""ImagineVLN 二期 runner — Claude SDK 多轮 + 按需想象（imagine 臂专用）。

会话结构（v2，对齐 AgentCanvas 的自动观察语义）：

* 观察不是工具：episode 开场把首个全景放进第一条 user message；此后每次
  goto() 的 **tool result 直接携带落地后的新全景 + 新候选点**（wp_bridge
  HABITAT_AUTO_OBSERVE / imagine_toolset._tool_goto 的做法）。模型在同一个
  会话里连续走多轮 —— 真正的多轮对话。
* 上下文契约：**当前轮之前的 imagine 预演图不进上下文**。会话是 append-only
  的，所以剔除靠会话重建：只要本会话里发生过 imagine，其后的 goto 一落地
  （预演图即变旧），runner 就 interrupt 掉会话，用「journey + 全景史 +
  当前全景」重开 —— 旧 sheet 不在新会话里。没 imagine 过的连续移动段
  完全不重建，一条会话走到底。
* 工具面只有 imagine / goto / stop（in-process SDK MCP，同一个
  ImagineToolset 实例跨会话存活）。
* 开跑前先过 EHarness 的 nav-instruction-refiner skill（同模型、无工具
  单轮，45 s 超时回退原文）。
* 产物 = AgentCanvas 契约（summary.json / episode_{i}.jsonl / live_{i}/），
  5173 的 ImagineVLN 板块直接显示；开场观察记成不带 tool_use 的裸
  tool_result（前端把 frames 渲染在该行，不伪造工具调用）。

跑法（agentcanvas env，subscription 认证，仓根任意位置）：
    python coding-agent/imaginevln/run_imagine_sdk.py --arm imagine --episodes 0 --split rand100 --model claude-opus-5
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent            # coding-agent/imaginevln
AC = Path(os.environ.get("AGENTCANVAS", str(HERE.parent.parent)))   # 仓根，可覆盖
for p in (str(HERE), str(AC / "coding-agent")):
    if p not in sys.path:
        sys.path.insert(0, p)

from imagine_tools import ImagineToolset  # noqa: E402

ENV_URL = os.environ.get("ENV_URL", "http://127.0.0.1:9200")
WP_URL = os.environ.get("WP_URL", "http://127.0.0.1:9210")
MW_URL = os.environ.get("MW_URL", "http://127.0.0.1:9270")
OUT_ROOT = Path(os.environ.get(
    "IMAGINEVLN_OUT", str(AC / "outputs" / "beta-imaginevln")))

MAX_MOVES = 30
SESSION_MAX_TURNS = 60       # 一个会话可能连走多轮；move 预算才是硬上限
QUERY_TIMEOUT_S = 3600
REFINE_TIMEOUT_S = 45
MAX_NUDGES = 2


IMAGINE_HOWITWORKS = """
- Occasionally the conversation reopens with a recap (your journey and the \
panoramas seen so far) — that is the same episode continuing; earlier \
rounds' predicted-future images are dropped to keep context lean, and only \
the CURRENT round's predictions are ever shown to you."""

IMAGINE_TOOL_PARA = """
- imagine(waypoints=[...]): OPTIONAL. A learned world model predicts, frame \
by frame, what you would see walking to each waypoint you name. Use it when \
the choice is genuinely uncertain — at junctions, when the instruction and \
the view do not obviously line up, or when two candidates both look \
plausible. Only ask for the candidates worth checking; skip the ones that \
clearly make no sense, and skip the call entirely when the right move is \
already obvious. Each requested waypoint costs a few seconds of compute."""

IMAGINE_READING = """

READING PREDICTIONS
Each imagine() image is a numbered filmstrip: frame 0 is the labeled \
panorama view you already saw (the prediction starts from that view, \
already facing it), the last frame is the predicted arrival. These are \
PREDICTIONS — trust layout, geometry, and whether the path stays open more \
than fine detail or exact colour. If a predicted walk-through runs into a \
wall or drifts away from what the instruction describes, do not go there; \
if it arrives at the described place, that is strong evidence for choosing \
it."""


def build_briefing(instruction: str, arm: str = "imagine") -> str:
    im = arm == "imagine"
    return f"""You are a wheeled robot navigating a real indoor environment. \
Your task is to follow this navigation instruction to its endpoint:

"{instruction}"

HOW IT WORKS
- The system has ALREADY given you your first observation below: a panoramic \
image of four views labeled Left / Front / Right / Back, with numbered green \
circles marking the waypoints you can move to, plus a JSON listing each \
waypoint's direction, angle (degrees left of your heading; negative = right) \
and distance in meters.
- Every goto() returns the NEW panorama and waypoints automatically — you \
never need to ask to look. Keep deciding and moving, turn after turn, until \
you stop().{IMAGINE_HOWITWORKS if im else ""}

TOOLS{IMAGINE_TOOL_PARA if im else ""}
- goto(waypoint, reason): move to one numbered waypoint from the LATEST \
panorama. Give a one-line reason — it becomes your journey log. The result \
carries your new panorama and waypoints.
- stop(reason): permanently END the episode, declaring you have reached the \
instruction's endpoint. Irreversible.{IMAGINE_READING if im else ""}

NAVIGATION
Before moving, name the part of the instruction you are currently \
executing, then say which numbered waypoint best matches it and why. \
Prefer the waypoint that carries you toward the landmark the instruction \
names. If you have clearly drifted off the described route, pick the \
waypoint that best gets you back on it.

STOP RULE
stop() succeeds only if you are within 3 meters of the instruction's \
endpoint. Stopping is permanent, and ending without stop() scores zero — \
when you believe you have arrived, stop; do not wander."""


def build_opening(ts: ImagineToolset, nudge: int, arm: str = "imagine") -> tuple[list[dict], dict]:
    """会话开场 user message：journey + 全景史（不含任何预演图）+ 当前观察。"""
    obs = ts.last_obs
    content: list[dict] = []

    def txt(s: str) -> None:
        content.append({"type": "text", "text": s})

    def img(png: bytes) -> None:
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.b64encode(png).decode()}})

    if nudge:
        txt("You ended your turn without acting. You MUST now either "
            "goto(...) one waypoint or stop(). The observation below is "
            "unchanged.")
    elif ts.moves:
        txt("(episode continues — recap below; earlier rounds' predicted-"
            "future images are dropped, everything else is your real history)")
    txt("Journey so far:\n" + ts.journey_text())

    for cyc, png in ts.pano_history[:-1]:
        txt(f"Panorama from round {cyc} (old — its waypoint numbers no "
            "longer apply):")
        img(png)

    txt("CURRENT panorama (the numbered waypoints you can act on now):")
    img(obs["pano_png"])
    txt(ts.obs_text()
        + ("\nOptionally imagine(...) the candidates you are unsure about, "
           "then goto(...) one waypoint — or stop() if you have reached the "
           "instruction's endpoint." if arm == "imagine" else
           "\ngoto(...) one waypoint — or stop() if you have reached the "
           "instruction's endpoint."))

    manifest = {"round": obs["cycle"], "nudge": nudge,
                "payload_images": len(ts.pano_history),
                "past_panos": len(ts.pano_history) - 1,
                "stale_imagine_sheets": 0}   # 按构造为 0：开场只装全景
    return content, manifest


def make_mcp_config(ts: ImagineToolset, arm: str = "imagine"):
    """三个工具包一层 in-process MCP。阻塞的 HTTP/扩散调用丢进线程，
    不能卡住 asyncio 循环（stdio 消息泵还要跑）。"""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool("imagine",
          "Preview walking to chosen waypoints: a world model predicts the "
          "frames you would see on the way. Name only the waypoints worth "
          "checking, e.g. [1,3,4]. Returns one predicted filmstrip per "
          "waypoint. Optional — call it only when it would change your "
          "decision.",
          {"type": "object",
           "properties": {"waypoints": {"type": "array",
                                        "items": {"type": "integer"},
                                        "description": "waypoint numbers from the LATEST panorama"}},
           "required": ["waypoints"]})
    async def t_imagine(args):
        parts = await asyncio.to_thread(ts.tool_imagine, args.get("waypoints"))
        return {"content": parts}

    @tool("goto",
          "Move to one numbered waypoint from the LATEST panorama. The robot "
          "turns toward it and walks there; the result carries your NEW "
          "panorama and waypoints automatically.",
          {"type": "object",
           "properties": {"waypoint": {"type": "integer"},
                          "reason": {"type": "string",
                                     "description": "one line: which part of the instruction this serves"}},
           "required": ["waypoint"]})
    async def t_goto(args):
        parts = await asyncio.to_thread(
            ts.tool_goto, args.get("waypoint"), args.get("reason", ""))
        return {"content": parts}

    @tool("stop",
          "Permanently END the episode, declaring you are within 3 meters of "
          "the instruction's endpoint. Irreversible.",
          {"type": "object",
           "properties": {"reason": {"type": "string"}}})
    async def t_stop(args):
        parts = await asyncio.to_thread(ts.tool_stop, args.get("reason", ""))
        return {"content": parts}

    tools = ([t_imagine] if arm == "imagine" else []) + [t_goto, t_stop]
    return create_sdk_mcp_server(name="env", tools=tools)


async def refine_instruction(model: str, instruction: str, log) -> dict:
    """EHarness §17：同模型、无工具单轮清洗指令；任何失败回退原文。"""
    try:
        from eharness import refiner
        system, user = refiner.build_prompt(refiner.load_skill_text(), instruction)
    except Exception as exc:  # noqa: BLE001
        log("user_text", text=f"[refiner] skill unavailable ({exc!r}) — using "
                              "the original instruction")
        return {"instruction": instruction, "fallback_reason": repr(exc)}

    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions,
                                  ClaudeSDKClient, TextBlock)

    async def _one() -> str:
        opts = ClaudeAgentOptions(
            system_prompt=system, tools=[], setting_sources=[], max_turns=1,
            model=model or None, permission_mode="bypassPermissions")
        texts: list[str] = []
        async with ClaudeSDKClient(options=opts) as client:
            await client.query(user)
            async for m in client.receive_response():
                if isinstance(m, AssistantMessage):
                    texts += [b.text for b in m.content if isinstance(b, TextBlock)]
        return "\n".join(texts)

    t0 = time.time()
    try:
        from eharness import refiner
        text = await asyncio.wait_for(_one(), timeout=REFINE_TIMEOUT_S)
        res = refiner.parse(text, instruction)
        rec = res.as_dict()
    except Exception as exc:  # noqa: BLE001 — 预处理绝不赔掉一集
        rec = {"instruction": instruction, "fallback_reason": repr(exc)}
    rec["duration_ms"] = int((time.time() - t0) * 1000)
    log("user_text", text=(
        "[refiner] " + ("kept the instruction as-is"
                        if rec.get("fallback_reason") or rec["instruction"] == instruction
                        else f"refined -> \"{rec['instruction']}\"")
        + (f" (fallback: {rec['fallback_reason']})" if rec.get("fallback_reason") else "")))
    return rec


async def drain(client, ts: ImagineToolset, log, raw) -> dict:
    """吸干一次 query 的消息流。goto 落地且会话里有旧 imagine 图时
    interrupt —— 这就是上下文剔除的触发点。"""
    from claude_agent_sdk import (AssistantMessage, ResultMessage,
                                  SystemMessage, TextBlock, ThinkingBlock,
                                  ToolResultBlock, ToolUseBlock, UserMessage)

    goto_ids: set[str] = set()
    meta: dict = {"usd": None, "turns": None, "model": None,
                  "interrupted": False}
    async for m in client.receive_response():
        raw(m)
        if isinstance(m, AssistantMessage):
            for b in m.content:
                if isinstance(b, TextBlock):
                    log("assistant_text", text=b.text, cycle=ts.cycle)
                elif isinstance(b, ThinkingBlock):
                    log("thinking", text=b.thinking, chars=len(b.thinking),
                        cycle=ts.cycle)
                elif isinstance(b, ToolUseBlock) and b.name.endswith("__goto"):
                    goto_ids.add(b.id)
        elif isinstance(m, UserMessage):
            blocks = m.content if isinstance(m.content, list) else []
            for b in blocks:
                if (isinstance(b, ToolResultBlock)
                        and b.tool_use_id in goto_ids
                        and ts.session_imagine_calls > 0
                        and not ts.episode_over
                        and not meta["interrupted"]):
                    # 本会话 imagine 过、且已带着结论移动 —— 那些 sheet
                    # 从这一刻起是旧图，不允许再出现在后续请求里
                    meta["interrupted"] = True
                    await client.interrupt()
        elif isinstance(m, SystemMessage):
            if getattr(m, "subtype", None) == "init":
                meta["model"] = (getattr(m, "data", {}) or {}).get("model")
        elif isinstance(m, ResultMessage):
            meta["usd"] = getattr(m, "total_cost_usd", None)
            meta["turns"] = getattr(m, "num_turns", None)
    return meta


async def run_episode(index: int, args, out_dir: Path, on_progress) -> dict:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    live = out_dir / f"live_{index}"
    live.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ev = (out_dir / f"episode_{index}.jsonl").open("w")
    raw_f = (raw_dir / f"episode_{index}.jsonl").open("w")
    t_ep = time.time()

    def log(kind, **kw):
        ev.write(json.dumps({"t": round(time.time() - t_ep, 2), "kind": kind,
                             **kw}, default=str) + "\n")
        ev.flush()

    def raw(message):
        raw_f.write(json.dumps({"type": type(message).__name__,
                                "msg": str(message)[:4000]}) + "\n")
        raw_f.flush()

    ts = ImagineToolset(ENV_URL, WP_URL, MW_URL, live,
                        max_moves=args.max_moves, log_event=log)
    ep = ts.reset_episode(args.split, index)
    instruction = str(ep.get("instruction") or "").strip()
    print(f"\n=== ep {index} [imagine·sdk] {ep.get('episode_id')} :: {instruction}",
          flush=True)
    log("user_text",
        text=f"[ep {index} · {args.arm} · {args.model}] {instruction}",
        index=index, split=args.split, arm=args.arm, model=args.model,
        episode_id=ep.get("episode_id"), scene_id=ep.get("scene_id"),
        instruction=instruction)

    refined = {"instruction": instruction}
    if args.refine:
        refined = await refine_instruction(args.model, instruction, log)
    briefing = build_briefing(refined["instruction"], args.arm)

    obs = ts.look()          # 首个观察；此后由 goto 内部自动观察
    # 开场观察 = 系统送达，不是工具调用：裸 tool_result（无 tool_use），
    # 前端把 frames 渲染在该行；文字用一行摘要，别倒模型侧的 JSON
    log("tool_result",
        texts=["first observation (auto) — " + ts.obs_summary()],
        frames=[obs["pano_name"]], cycle=obs["cycle"])
    if not obs["n_cands"]:
        end_reason = "no_candidates"
        metrics = ts.evaluate()
        rec = {"index": index, "arm": args.arm, "error": "no_candidates",
               "metrics": metrics}
        log("episode_metrics", **rec)
        ev.close(); raw_f.close()
        return rec

    usd, sdk_turns, sessions, queries, nudges_total = 0.0, 0, 0, 0, 0
    end_reason = "move_budget_exhausted"
    manifests: list[dict] = []

    while not ts.episode_over:
        content, manifest = build_opening(ts, nudge=0, arm=args.arm)
        log("context", **manifest, sessions=sessions + 1)
        manifests.append(manifest)
        sessions += 1
        ts.session_imagine_calls = 0

        opts = ClaudeAgentOptions(
            system_prompt=briefing,
            mcp_servers={"env": make_mcp_config(ts, args.arm)},
            allowed_tools=(["mcp__env__imagine"] if args.arm == "imagine"
                           else []) + ["mcp__env__goto", "mcp__env__stop"],
            tools=[],
            setting_sources=[],
            strict_mcp_config=True,
            thinking={"type": "adaptive", "display": "summarized"},
            permission_mode="bypassPermissions",
            max_buffer_size=64 * 1024 * 1024,
            max_turns=SESSION_MAX_TURNS,
            model=args.model or None,
            cwd=str(out_dir),
        )

        async def _stream(_content=content):
            yield {"type": "user",
                   "message": {"role": "user", "content": _content},
                   "parent_tool_use_id": None}

        nudges = 0
        async with ClaudeSDKClient(options=opts) as client:
            await client.query(_stream())
            while True:
                snap_moves, snap_over = ts.moves, ts.episode_over
                t0 = time.time()
                try:
                    meta = await asyncio.wait_for(
                        drain(client, ts, log, raw), timeout=QUERY_TIMEOUT_S)
                except asyncio.TimeoutError:
                    log("assistant_text", text="[query timeout]", cycle=ts.cycle)
                    meta = {"usd": None, "turns": None, "interrupted": False}
                queries += 1
                usd += meta.get("usd") or 0.0
                sdk_turns += meta.get("turns") or 0
                print(f"  session {sessions} q{queries}: moves {ts.moves}, "
                      f"imagine {ts.imagine_calls}, {int(time.time() - t0)}s, "
                      f"turns {meta.get('turns')}, ${meta.get('usd') or 0:.2f}"
                      + (f", model {meta.get('model')}"
                         if sessions == 1 and queries == 1 else ""), flush=True)
                on_progress({"index": index, "arm": "imagine",
                             "model": args.model,
                             "episode_id": ep.get("episode_id"),
                             "instruction": instruction, "in_progress": True,
                             "moves": ts.moves, "llm_calls": queries,
                             "usd": round(usd, 4),
                             "imagine_ms": ts.imagine_ms,
                             "agent": {"env_steps": ts.steps_taken,
                                       "called_stop": ts.called_stop}})

                if ts.episode_over:
                    break
                if meta.get("interrupted"):
                    break            # 重建会话，剔除旧 sheet
                acted = ts.moves > snap_moves or ts.episode_over != snap_over
                if acted:
                    # 干净移动后模型自己收了轮 —— 同会话续推
                    await client.query("Continue navigating.")
                    continue
                nudges += 1
                nudges_total += 1
                if nudges > MAX_NUDGES:
                    ts.episode_over = True
                    end_reason = "no_action"
                    break
                await client.query(
                    "You must act: goto(...) one waypoint from the latest "
                    "panorama, or stop() if you are at the endpoint.")

        if ts.end_reason:
            end_reason = ts.end_reason

    metrics = ts.evaluate()
    rec = {"index": index, "arm": args.arm, "model": args.model,
           "episode_id": ep.get("episode_id"), "scene_id": ep.get("scene_id"),
           "instruction": instruction,
           "refined_instruction": refined.get("instruction"),
           "refine": {k: v for k, v in refined.items() if k != "instruction"},
           "metrics": metrics, "called_stop": ts.called_stop,
           "end_reason": end_reason, "moves": ts.moves,
           "llm_calls": queries, "sessions": sessions, "nudges": nudges_total,
           "sdk_turns": sdk_turns, "usd": round(usd, 4),
           "imagine": {"calls": ts.imagine_calls,
                       "waypoints": ts.imagine_waypoints,
                       "cands_offered": ts.cands_offered,
                       "ms": ts.imagine_ms},
           "context": {"max_payload_images": max((m["payload_images"]
                                                  for m in manifests), default=0),
                       "stale_imagine_sheets_ever": sum(m["stale_imagine_sheets"]
                                                        for m in manifests)},
           "wall_s": round(time.time() - t_ep, 1),
           "agent": {"env_steps": int(metrics.get("steps_taken")
                                      or ts.steps_taken),
                     "called_stop": ts.called_stop}}
    log("episode_metrics", **rec)
    ev.close()
    raw_f.close()
    print(f"  -> {end_reason}: SR={metrics.get('success')} "
          f"SPL={metrics.get('spl')} NE={metrics.get('distance_to_goal')} | "
          f"{rec['wall_s']}s, ${rec['usd']}, {sessions} sessions, "
          f"imagine {ts.imagine_calls} calls / {ts.imagine_waypoints} wps / "
          f"{ts.cands_offered} offered", flush=True)
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="0", help="0 | 0-4 | 0,3,7")
    ap.add_argument("--split", default="rand100")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--arm", choices=["imagine", "base"], default="imagine")
    ap.add_argument("--worker-tag", default=None,
                    help="并行分片：summary 写成 summary_<tag>.json，由合并器汇总")
    ap.add_argument("--max-moves", type=int, default=MAX_MOVES)
    ap.add_argument("--no-refine", dest="refine", action="store_false")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 订阅认证：环境里飘着 ANTHROPIC_API_KEY 会悄悄改成 API 计费
    if os.environ.get("CODING_AGENT_ALLOW_API_KEY") != "1":
        if os.environ.pop("ANTHROPIC_API_KEY", None):
            print("[sdk] ANTHROPIC_API_KEY removed — subscription auth")

    idxs: list[int] = []
    for part in args.episodes.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            idxs += list(range(int(lo), int(hi) + 1))
        else:
            idxs.append(int(part))

    run_name = args.out or f"sdk_{args.model}_{args.arm}"
    out_dir = OUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[imaginevln] sdk imagine arm, model={args.model} "
          f"split={args.split} episodes={idxs}\n[imaginevln] out={out_dir}",
          flush=True)

    eps: list[dict] = []
    live: dict | None = None
    config = {"split": args.split, "model": args.model, "arm": args.arm,
              "skill": f"{args.arm}-sdk",
              "refine": args.refine, "max_moves": args.max_moves,
              "episodes": args.episodes,
              "context": "resident session; goto carries auto-observe; "
                         "session rebuilt after imagine-then-move so stale "
                         "sheets never re-enter; full pano history"}

    def flush():
        rows = eps + ([live] if live else [])
        scored = [e for e in rows
                  if (e.get("metrics") or {}).get("success") is not None]
        agg: dict = {"episode_count": len(scored)}
        for k in ("success", "spl", "ndtw", "oracle_success", "distance_to_goal"):
            vals = [float(e["metrics"][k]) for e in scored
                    if isinstance((e.get("metrics") or {}).get(k), (int, float))]
            if vals:
                agg[k] = round(sum(vals) / len(vals), 4)
        name = (f"summary_{args.worker_tag}.json" if args.worker_tag
                else "summary.json")
        (out_dir / name).write_text(json.dumps(
            {"run": run_name, "arm": args.arm, "model": args.model,
             "split": args.split, "config": config, "aggregate": agg,
             "episodes": rows}, indent=2, default=str))

    def on_progress(partial):
        nonlocal live
        live = partial
        flush()

    for i in idxs:
        for attempt in (1, 2):
            try:
                eps.append(await run_episode(i, args, out_dir, on_progress))
                break
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                if attempt == 2:
                    eps.append({"index": i, "arm": args.arm, "error": repr(e)})
                else:
                    print(f"[imaginevln] ep {i} attempt 1 failed ({e!r}); "
                          "retrying in 60s", flush=True)
                    await asyncio.sleep(60)
        live = None
        flush()
    print("\n[imaginevln] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
