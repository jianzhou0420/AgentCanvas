"""Post-run statistics report — every run gets charts + tables after metrics.

Reads a completed run directory (summary.json + episode_*.jsonl + raw/*.jsonl)
and writes two artifacts next to them:

- ``stats.html`` — self-contained report (matplotlib charts inlined as base64
  PNG; opens anywhere, no server). Layout: run header -> aggregate metrics ->
  every further statistic AFTER the metric block (user decision 2026-07-23),
  then ONE SECTION PER EPISODE (facts table + auto-generated reading notes +
  a per-episode chart triptych; user decision 2026-07-23: run-level numbers
  alone don't explain individual episodes).
- ``stats.json`` — the same numbers, structured.

Per-turn USD is exact, not price-table guesswork: each API call's usage
(reported by the API server, deduped by message_id) is turned into a weighted
share and normalized against the episode's CLI-settled ``total_cost_usd``, so
long-context premiums and cache discounts are absorbed automatically. Runs
whose raw/ stream is missing (older layouts, foreign harnesses) degrade
gracefully: token/cost-curve sections are skipped, tables stay.

Called automatically at the end of driver.run_cell (best-effort — a failed
report never loses a run); backfill old runs via the CLI:

    python coding-agent/ac_support/run_stats.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import base64
import html
import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Relative billing weights (input=1) — used only to APPORTION an episode's
# CLI-settled total across its calls, so absolute prices cancel out.
_TOKEN_WEIGHTS = {"input_tokens": 1.0, "cache_read_input_tokens": 0.1,
                  "output_tokens": 5.0}
_CACHE_WRITE_WEIGHTS = {"ephemeral_1h_input_tokens": 2.0,
                        "ephemeral_5m_input_tokens": 1.25}
_ACTION_NAMES = {0: "STOP", 1: "FWD", 2: "LEFT", 3: "RIGHT",
                 4: "LOOK_UP", 5: "LOOK_DOWN"}


# ── extraction ──


def _episode_events(path: Path) -> dict[str, Any]:
    """Digest one episode_N.jsonl: tool calls, result, metrics, wall time."""
    out: dict[str, Any] = {"tool_names": Counter(), "step_batches": [],
                           "actions": Counter(), "result": {}, "metrics": {},
                           "meta": {}, "wall_sec": 0.0}
    try:
        with path.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = rec.get("kind")
                out["wall_sec"] = rec.get("t", out["wall_sec"])
                if kind == "episode_meta":
                    out["meta"] = {k: rec.get(k) for k in
                                   ("episode_id", "scene_id", "instruction")}
                elif kind == "tool_use":
                    name = str(rec.get("name", "?")).replace("mcp__env__", "")
                    out["tool_names"][name] += 1
                    actions = (rec.get("input") or {}).get("actions")
                    if name == "step" and isinstance(actions, list):
                        out["step_batches"].append(len(actions))
                        for a in actions:
                            if isinstance(a, int):
                                out["actions"][a] += 1
                elif kind == "result":
                    out["result"] = rec.get("result") or {}
                elif kind == "episode_metrics":
                    out["metrics"] = rec.get("metrics") or {}
    except FileNotFoundError:
        pass
    return out


def _raw_calls(path: Path) -> list[dict[str, Any]]:
    """Per-API-call records from the raw stream: usage (server-reported,
    deduped by message_id), issuing tool, images in the following result."""
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with path.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "AssistantMessage":
                    msg = rec.get("msg") or {}
                    mid = msg.get("message_id")
                    usage = msg.get("usage") or {}
                    tool = None
                    for block in (msg.get("content") or []):
                        if isinstance(block, dict) and block.get("_type") == "ToolUseBlock":
                            tool = str(block.get("name", "?")).replace("mcp__env__", "")
                    if mid and mid not in seen and usage:
                        seen.add(mid)
                        cw = usage.get("cache_creation") or {}
                        weight = sum(usage.get(k, 0) * w
                                     for k, w in _TOKEN_WEIGHTS.items())
                        weight += sum(cw.get(k, 0) * w
                                      for k, w in _CACHE_WRITE_WEIGHTS.items())
                        calls.append({
                            "tool": tool,
                            "weight": weight,
                            "input": usage.get("input_tokens", 0),
                            "cache_read": usage.get("cache_read_input_tokens", 0),
                            "cache_write": sum(cw.get(k, 0) for k in _CACHE_WRITE_WEIGHTS),
                            "output": usage.get("output_tokens", 0),
                        })
                    elif mid in seen and tool and calls and calls[-1]["tool"] is None:
                        calls[-1]["tool"] = tool  # tool block arrived in a later batch
                elif rec.get("type") == "UserMessage":
                    for block in ((rec.get("msg") or {}).get("content") or []):
                        if (isinstance(block, dict)
                                and block.get("_type") == "ToolResultBlock" and calls):
                            content = block.get("content")
                            n_img = (sum(1 for x in content
                                         if isinstance(x, dict) and x.get("type") == "image")
                                     if isinstance(content, list) else 0)
                            calls[-1]["images_back"] = calls[-1].get("images_back", 0) + n_img
    except FileNotFoundError:
        pass
    return calls


def collect(run_dir: Path) -> dict[str, Any]:
    """All statistics for one run, structured (the stats.json payload)."""
    summary = json.loads((run_dir / "summary.json").read_text())
    episodes: list[dict[str, Any]] = []
    for ep in sorted(summary.get("episodes", []), key=lambda e: e.get("index", 0)):
        idx = ep.get("index")
        agent = ep.get("agent") or {}
        metrics = ep.get("metrics") or {}
        ev = _episode_events(run_dir / f"episode_{idx}.jsonl")
        calls = _raw_calls(run_dir / "raw" / f"episode_{idx}.jsonl")
        cost = agent.get("total_cost_usd") or 0.0
        total_w = sum(c["weight"] for c in calls)
        per_call_usd = [round(cost * c["weight"] / total_w, 6) if total_w else None
                        for c in calls]
        turns = ev["result"].get("turns") or agent.get("num_turns")
        episodes.append({
            "index": idx,
            "goal": ep.get("instruction"),
            "scene": Path(str((ev["meta"] or {}).get("scene_id") or "?")).stem,
            "error": ep.get("error"),
            "subtype": agent.get("result_subtype"),
            "turns": turns,
            "api_calls": len(calls) or None,
            "tool_calls": dict(ev["tool_names"]),
            "env_steps": agent.get("env_steps"),
            "called_stop": agent.get("called_stop"),
            "cost_usd": round(cost, 4),
            "usd_per_turn": round(cost / turns, 4) if turns else None,
            "steps_per_turn": (round(agent.get("env_steps", 0) / turns, 2)
                               if turns else None),
            "images_received": sum(c.get("images_back", 0) for c in calls) or None,
            "tokens": {
                "input": sum(c["input"] for c in calls),
                "cache_read": sum(c["cache_read"] for c in calls),
                "cache_write": sum(c["cache_write"] for c in calls),
                "output": sum(c["output"] for c in calls),
            } if calls else None,
            "step_batches": ev["step_batches"],
            "actions": {_ACTION_NAMES.get(k, str(k)): v
                        for k, v in sorted(ev["actions"].items())},
            "metrics": metrics,
            "wall_min": round(ev["wall_sec"] / 60, 1),
            "per_call_usd": per_call_usd,
            "per_call_tool": [c["tool"] for c in calls],
        })

    scored = [e for e in episodes if e["metrics"] and e["error"] is None]
    costs = [e["cost_usd"] for e in episodes if e["cost_usd"]]
    cost_by_tool: Counter = Counter()
    for e in episodes:
        for usd, tool in zip(e["per_call_usd"] or [], e["per_call_tool"] or []):
            if usd is not None:
                cost_by_tool[tool or "final_text"] += usd
    action_totals: Counter = Counter()
    batch_sizes: list[int] = []
    for e in episodes:
        action_totals.update(e["actions"])
        batch_sizes.extend(e["step_batches"])

    def _stats(vals: list[float]) -> dict[str, float] | None:
        if not vals:
            return None
        vs = sorted(vals)
        return {"total": round(sum(vs), 4), "mean": round(sum(vs) / len(vs), 4),
                "median": round(vs[len(vs) // 2], 4),
                "min": round(vs[0], 4), "max": round(vs[-1], 4)}

    return {
        "run_name": summary.get("run_name"),
        "cell": summary.get("cell"),
        "harness": summary.get("harness"),
        "config": summary.get("config"),
        "aggregate": summary.get("aggregate"),
        "outcomes": dict(Counter(
            e["error"] or e["subtype"] or "?" for e in episodes)),
        "scored_episodes": len(scored),
        "excluded_episodes": len(episodes) - len(scored),
        "cost": {**(_stats(costs) or {}),
                 "by_issuing_tool": {k: round(v, 2)
                                     for k, v in cost_by_tool.most_common()}},
        "turns": _stats([e["turns"] for e in episodes if e["turns"]]),
        "env_steps": _stats([e["env_steps"] for e in episodes
                             if e["env_steps"] is not None]),
        "steps_per_turn": _stats([e["steps_per_turn"] for e in episodes
                                  if e["steps_per_turn"]]),
        "usd_per_turn": _stats([e["usd_per_turn"] for e in episodes
                                if e["usd_per_turn"]]),
        "actions_total": dict(action_totals),
        "step_batch_size": _stats([float(b) for b in batch_sizes]),
        "wall_min": _stats([e["wall_min"] for e in episodes if e["wall_min"]]),
        "episodes": episodes,
    }


# ── charts ──


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _charts(stats: dict[str, Any]) -> list[tuple[str, str]]:
    """(title, base64 png) pairs; every chart is optional on missing data."""
    eps = stats["episodes"]
    out: list[tuple[str, str]] = []

    with_calls = [e for e in eps if e["per_call_usd"] and any(
        u is not None for u in e["per_call_usd"])]
    if with_calls:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for e in with_calls:
            cum, acc = [], 0.0
            for u in e["per_call_usd"]:
                acc += u or 0.0
                cum.append(acc)
            ax.plot(range(1, len(cum) + 1), cum, alpha=0.7,
                    label=f"ep{e['index']}")
        ax.set_xlabel("turn"); ax.set_ylabel("cumulative USD")
        ax.legend(fontsize=6, ncols=2)
        out.append(("Cumulative cost per turn (exact, CLI-normalized)",
                    _fig_to_b64(fig)))

        fig, ax = plt.subplots(figsize=(7, 3.2))
        for e in with_calls:
            ax.plot(range(1, len(e["per_call_usd"]) + 1),
                    [u or 0 for u in e["per_call_usd"]], alpha=0.6)
        ax.set_xlabel("turn"); ax.set_ylabel("USD per call")
        out.append(("Per-call cost (the resend toll curve)", _fig_to_b64(fig)))

        fig, ax = plt.subplots(figsize=(7, 3.2))
        n = max(len(e["per_call_usd"]) for e in with_calls)
        keys = ("input", "cache_read", "cache_write", "output")
        # token composition per call index, averaged over the episodes that
        # are still alive at that index
        comp = {k: [] for k in keys}
        for i in range(n):
            alive = [e for e in with_calls if i < len(e["per_call_usd"])]
            raws = [
                _raw_calls_cache.get((stats["run_name"], e["index"]))
                for e in alive
            ]
            raws = [r[i] for r in raws if r and i < len(r)]
            for k in keys:
                comp[k].append(sum(r[k] for r in raws) / max(1, len(raws)))
        bottom = [0.0] * n
        for k in keys:
            ax.bar(range(1, n + 1), comp[k], bottom=bottom, width=1.0, label=k)
            bottom = [b + v for b, v in zip(bottom, comp[k])]
        ax.set_xlabel("turn"); ax.set_ylabel("tokens (mean over alive eps)")
        ax.legend(fontsize=7)
        out.append(("Token composition per call (server-reported usage)",
                    _fig_to_b64(fig)))

    if any(e["cost_usd"] for e in eps):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        axes[0].bar([str(e["index"]) for e in eps],
                    [e["cost_usd"] for e in eps])
        axes[0].set_xlabel("episode"); axes[0].set_ylabel("USD")
        axes[0].set_title("cost", fontsize=9)
        axes[1].bar([str(e["index"]) for e in eps],
                    [e["turns"] or 0 for e in eps])
        axes[1].set_xlabel("episode"); axes[1].set_ylabel("turns")
        axes[1].set_title("turns", fontsize=9)
        out.append(("Cost and turns per episode", _fig_to_b64(fig)))

    if stats["actions_total"]:
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
        names = list(stats["actions_total"])
        axes[0].bar(names, [stats["actions_total"][k] for k in names])
        axes[0].set_title("action mix", fontsize=9)
        axes[0].tick_params(axis="x", labelsize=7)
        batches = [b for e in eps for b in e["step_batches"]]
        if batches:
            axes[1].hist(batches, bins=range(0, max(batches) + 2))
            axes[1].set_title("step() batch size", fontsize=9)
        out.append(("Action distribution", _fig_to_b64(fig)))

    dtg = [(e["index"], (e["metrics"] or {}).get("distance_to_goal"),
            (e["metrics"] or {}).get("success"), e["called_stop"])
           for e in eps if (e["metrics"] or {}).get("distance_to_goal") is not None]
    if dtg:
        fig, ax = plt.subplots(figsize=(7, 2.8))
        colors = ["#2a9d2a" if s else ("#e0a000" if st else "#c0392b")
                  for _, _, s, st in dtg]
        ax.bar([str(i) for i, *_ in dtg], [d for _, d, *_ in dtg], color=colors)
        ax.axhline(1.0, ls="--", lw=0.8, color="gray")
        ax.set_xlabel("episode"); ax.set_ylabel("distance to goal (m)")
        out.append(("Final distance to goal "
                    "(green=success, yellow=stopped, red=no stop; dashes: 1 m)",
                    _fig_to_b64(fig)))
    return out


# raw calls are re-read for the token-composition + per-episode charts;
# tiny cache avoids parsing every raw file twice
_raw_calls_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}


# ── per-episode section ──


def _episode_figure(ep: dict[str, Any], raw: list[dict[str, Any]]) -> str | None:
    """One triptych per episode: cost per turn, token mix per call, step
    batches — the single-episode story the run-level charts average away."""
    usd = [u or 0.0 for u in (ep["per_call_usd"] or [])]
    if not usd and not raw and not ep["step_batches"]:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5))

    ax = axes[0]
    cum, acc = [], 0.0
    for u in usd:
        acc += u
        cum.append(acc)
    ax.plot(range(1, len(cum) + 1), cum, color="#1f77b4", label="cumulative")
    ax.set_ylabel("cumulative USD")
    ax2 = ax.twinx()
    ax2.bar(range(1, len(usd) + 1), usd, alpha=0.35, color="#ff7f0e",
            width=1.0, label="per call")
    ax2.set_ylabel("USD per call")
    ax.set_title("cost per turn (CLI-normalized)", fontsize=9)

    ax = axes[1]
    if raw:
        keys = ("input", "cache_read", "cache_write", "output")
        bottom = [0.0] * len(raw)
        for k in keys:
            vals = [c[k] for c in raw]
            ax.bar(range(1, len(raw) + 1), vals, bottom=bottom, width=1.0, label=k)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.legend(fontsize=6, ncols=4)
    ax.set_ylabel("tokens per call")
    ax.set_title("token composition (server-reported usage)", fontsize=9)

    ax = axes[2]
    batches = ep["step_batches"]
    if batches:
        ax.bar(range(1, len(batches) + 1), batches, width=1.0, color="#2a9d2a")
    ax.set_xlabel("step() call #")
    ax.set_ylabel("actions per call")
    ax.set_title("step batch sizes", fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _episode_notes(ep: dict[str, Any], cfg: dict[str, Any],
                   run_median_cache_write: float) -> list[str]:
    """Auto-generated reading notes — what ended the episode and what stands
    out, so a single episode is interpretable without the run context."""
    notes: list[str] = []
    m = ep["metrics"] or {}
    subtype = ep["subtype"]
    if ep["error"]:
        notes.append(f"NOT SCORED — session error `{ep['error']}` "
                     "(excluded from the aggregate as a non-run).")
    elif m.get("success"):
        notes.append(f"SUCCESS — stopped {m.get('distance_to_goal', 0):.2f} m "
                     f"from the goal viewpoint in {ep['turns']} turns.")
    elif subtype == "error_max_turns":
        notes.append(f"Ended by the TURN CAP ({cfg.get('max_turns')}) without "
                     "STOP — scores 0 wherever the robot stood.")
    elif subtype == "error_max_budget_usd":
        notes.append(f"Ended by the ${cfg.get('max_budget_usd')} USD FUSE at "
                     f"turn {ep['turns']} — scored at final position, "
                     "not retried.")
    elif ep["called_stop"]:
        notes.append(f"Agent STOPPED voluntarily but "
                     f"{m.get('distance_to_goal', float('nan')):.2f} m from the "
                     "goal viewpoint — a placement failure, not a stop failure.")
    step_budget = cfg.get("step_budget")
    if step_budget and (ep["env_steps"] or 0) >= step_budget:
        notes.append(f"Exhausted the {step_budget}-step ENV BUDGET — the "
                     "simulator refused further motion.")
    if (not m.get("success") and m.get("distance_to_goal") is not None
            and m["distance_to_goal"] <= 2.0):
        notes.append(f"NEAR MISS — final distance {m['distance_to_goal']:.2f} m.")
    t = ep["tokens"] or {}
    if (t.get("cache_write") and run_median_cache_write
            and t["cache_write"] > 3 * run_median_cache_write
            and t["cache_write"] > 500_000):
        notes.append(f"CACHE-WRITE ANOMALY — {t['cache_write']:,} write tokens "
                     f"(run median {int(run_median_cache_write):,}); prompt-cache "
                     "thrash multiplied this episode's bill.")
    if ep["usd_per_turn"]:
        notes.append(f"Cost rate: ${ep['usd_per_turn']:.3f}/turn, "
                     f"{ep['steps_per_turn'] or 0:.1f} env-steps/turn.")
    return notes


# ── report ──


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{h}</th>" for _, h in cols)
    body = ""
    for r in rows:
        cells = ""
        for key, _ in cols:
            v = r.get(key)
            if isinstance(v, float):
                v = f"{v:.2f}"
            if isinstance(v, str) and v.startswith("<a "):
                cells += f"<td>{v}</td>"  # pre-rendered anchor (episode links)
            else:
                cells += f"<td>{html.escape('' if v is None else str(v))}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def generate(run_dir: Path | str) -> Path:
    """Write stats.html + stats.json into run_dir; returns the html path."""
    run_dir = Path(run_dir)
    stats = collect(run_dir)
    for e in stats["episodes"]:
        _raw_calls_cache[(stats["run_name"], e["index"])] = _raw_calls(
            run_dir / "raw" / f"episode_{e['index']}.jsonl")
    charts = _charts(stats)

    slim = {k: v for k, v in stats.items() if k != "episodes"}
    slim["episodes"] = [{k: v for k, v in e.items()
                         if k not in ("per_call_usd", "per_call_tool")}
                        for e in stats["episodes"]]
    (run_dir / "stats.json").write_text(json.dumps(slim, indent=1))

    agg = stats["aggregate"] or {}
    cfg = stats["config"] or {}
    ep_cols = [("index", "ep"), ("goal", "goal"), ("scene", "scene"),
               ("subtype", "end"), ("error", "err"), ("turns", "turns"),
               ("env_steps", "steps"), ("steps_per_turn", "st/turn"),
               ("cost_usd", "USD"), ("usd_per_turn", "$/turn"),
               ("called_stop", "stop"), ("wall_min", "wall(min)")]
    metric_keys = sorted({k for e in stats["episodes"]
                          for k in (e["metrics"] or {})})
    rows = []
    for e in stats["episodes"]:
        rows.append({**e, **{k: (e["metrics"] or {}).get(k) for k in metric_keys},
                     "index": f"<a href='#ep{e['index']}'>{e['index']}</a>"})
    ep_cols += [(k, k) for k in metric_keys]

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>stats — {html.escape(str(stats['run_name']))}</title>",
        "<style>body{font:14px/1.5 system-ui;margin:2rem auto;max-width:74rem;"
        "padding:0 1rem}table{border-collapse:collapse;font-size:12.5px;"
        "margin:.6rem 0}td,th{border:1px solid #bbb;padding:2px 7px;"
        "text-align:right}th{background:#eee}td:first-child,th:first-child,"
        "td:nth-child(2),td:nth-child(3){text-align:left}img{max-width:100%}"
        "h2{margin-top:1.6em}code{background:#eee;padding:0 3px}</style>",
        f"<h1>Run stats — <code>{html.escape(str(stats['run_name']))}</code></h1>",
        f"<p>cell <code>{html.escape(str(stats['cell']))}</code> · harness "
        f"<code>{html.escape(str(stats['harness']))}</code> · model "
        f"<code>{html.escape(str(cfg.get('model')))}</code> · "
        f"max_turns {cfg.get('max_turns')} · fuse "
        f"${cfg.get('max_budget_usd', '—')} · split "
        f"<code>{html.escape(str(cfg.get('split')))}</code></p>",
        "<h2>Aggregate metrics</h2>",
        _table([agg], [(k, k) for k in agg]),
        f"<p>scored {stats['scored_episodes']} · excluded "
        f"{stats['excluded_episodes']} · outcomes "
        f"{html.escape(json.dumps(stats['outcomes']))}</p>",
        "<h2>Per-episode</h2>",
        _table(rows, ep_cols),
        "<h2>Run-level statistics</h2>",
        _table(
            [{"what": k, **(stats[k] or {})}
             for k in ("cost", "turns", "env_steps", "steps_per_turn",
                       "usd_per_turn", "step_batch_size", "wall_min")
             if isinstance(stats.get(k), dict)],
            [("what", ""), ("total", "total"), ("mean", "mean"),
             ("median", "median"), ("min", "min"), ("max", "max")]),
        f"<p>cost by issuing tool: "
        f"{html.escape(json.dumps(stats['cost'].get('by_issuing_tool', {})))}"
        f"</p><p>actions total: "
        f"{html.escape(json.dumps(stats['actions_total']))}</p>",
    ]
    for title, b64 in charts:
        parts.append(f"<h2>{html.escape(title)}</h2>"
                     f"<img src='data:image/png;base64,{b64}'>")

    # ── one section per episode (facts + reading notes + chart triptych) ──
    cache_writes = sorted((e["tokens"] or {}).get("cache_write", 0)
                          for e in stats["episodes"] if e["tokens"])
    median_cw = cache_writes[len(cache_writes) // 2] if cache_writes else 0.0
    parts.append("<h2 id='episodes'>Episode-level reports</h2><p>"
                 + " · ".join(f"<a href='#ep{e['index']}'>ep{e['index']}</a>"
                              for e in stats["episodes"]) + "</p>")
    fact_cols = [("subtype", "end"), ("error", "err"), ("turns", "turns"),
                 ("api_calls", "calls"), ("env_steps", "steps"),
                 ("steps_per_turn", "st/turn"), ("cost_usd", "USD"),
                 ("usd_per_turn", "$/turn"), ("images_received", "imgs"),
                 ("called_stop", "stop"), ("wall_min", "wall(min)")]
    for e in stats["episodes"]:
        raw = _raw_calls_cache.get((stats["run_name"], e["index"]), [])
        m = e["metrics"] or {}
        t = e["tokens"] or {}
        parts.append(
            f"<h3 id='ep{e['index']}'>Episode {e['index']} — "
            f"“{html.escape(str(e['goal']))}” "
            f"<small>({html.escape(str(e['scene']))})</small></h3>")
        notes = _episode_notes(e, cfg, median_cw)
        if notes:
            parts.append("<ul>" + "".join(
                f"<li>{html.escape(n)}</li>" for n in notes) + "</ul>")
        parts.append(_table(
            [{**e, **{k: m.get(k) for k in metric_keys}}],
            fact_cols + [(k, k) for k in metric_keys]))
        if t:
            parts.append(
                "<p>tokens — input "
                f"{t.get('input', 0):,} · cache-read {t.get('cache_read', 0):,}"
                f" · cache-write {t.get('cache_write', 0):,} · output "
                f"{t.get('output', 0):,}; tool calls "
                f"{html.escape(json.dumps(e['tool_calls']))}; actions "
                f"{html.escape(json.dumps(e['actions']))}</p>")
        b64 = _episode_figure(e, raw)
        if b64:
            parts.append(f"<img src='data:image/png;base64,{b64}'>")

    _raw_calls_cache.clear()
    out = run_dir / "stats.html"
    out.write_text("\n".join(parts))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: run_stats.py <run_dir> [<run_dir> ...]")
    for arg in sys.argv[1:]:
        try:
            print(generate(arg))
        except Exception as exc:  # noqa: BLE001 — keep batch backfills going
            print(f"{arg}: FAILED {exc!r}")
