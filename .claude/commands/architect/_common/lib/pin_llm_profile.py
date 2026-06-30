"""Architect llmCall profile pin — force every llmCall node onto one model.

Shared by every variant's implementer. Architecture-search runs let the
meta-LLM add ``llmCall`` nodes freely; left unchecked it picks arbitrary
(often expensive) profiles. iter_1 of ``explore_eqa_hmeqa`` pinned
``gpt-4o`` for a vision adjudicator and — combined with a per-step
re-fire — drove the gpt-4o vision-call count up ~29x for one 100-episode
eval. The proposer cheat-sheet no longer names a profile at all — this
module is the only place an llmCall's profile is decided.

This module is the deterministic enforcement. After the implementer's
editing sub-agent runs, every ``llmCall`` node in the iter's overlay
graphs is rewritten to the single pinned profile + temperature,
regardless of what the proposer asked for.

Policy (the pin):
  * ``config.profile``     -> :data:`PINNED_PROFILE`
  * ``config.temperature`` -> :data:`PINNED_TEMPERATURE`
  * ``config.model``       -> dropped (dead field — the runtime never
                              reads it; only ``profile`` is, see
                              ``proposer.md`` step 3.5)
Every other ``llmCall`` config key (``max_tokens``, ``system_prompt``,
``template``, ``mode``, ``n``, ``stop``, ...) is left untouched.

:data:`PINNED_PROFILE` is ``gpt-5-mini``: the gpt-5 family is multimodal
(a vision adjudicator keeps working) and REQUIRES ``temperature=1`` —
which is why :data:`PINNED_TEMPERATURE` is ``1``. The two constants are
coupled; do not change one without the other.

Non-blocking: the pin never fails an iter — it only rewrites config. A
graph that does not ``json.load`` is already an ``edit_error`` caught by
the implementer's post-edit validation before this runs.
"""

from __future__ import annotations

import glob
import json
import os
import sys

# ── the pin — coupled constants, see module docstring ────────────────
PINNED_PROFILE = "gpt-5-mini"
PINNED_TEMPERATURE = 1


def pin_graph(graph_path: str) -> list[dict]:
    """Rewrite every ``llmCall`` node in one graph JSON to the pinned
    profile + temperature. Writes the file back only if something
    changed. Returns one change record per node actually modified:
    ``{node_id, profile_before, temperature_before, dropped_model}``.
    """
    with open(graph_path) as f:
        g = json.load(f)

    changes: list[dict] = []
    for n in g.get("nodes", []):
        if (n.get("type") or n.get("node_type")) != "llmCall":
            continue
        cfg = n.setdefault("config", {})
        before_profile = cfg.get("profile")
        before_temp = cfg.get("temperature")
        had_model = "model" in cfg

        if before_profile == PINNED_PROFILE and before_temp == PINNED_TEMPERATURE and not had_model:
            continue  # already pinned — no-op

        cfg["profile"] = PINNED_PROFILE
        cfg["temperature"] = PINNED_TEMPERATURE
        cfg.pop("model", None)
        changes.append(
            {
                "node_id": n.get("id", "?"),
                "profile_before": before_profile,
                "temperature_before": before_temp,
                "dropped_model": had_model,
            }
        )

    if changes:
        with open(graph_path, "w") as f:
            json.dump(g, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return changes


def pin_active_workspace(active_ws: str) -> int:
    """Pin every ``graphs/*.json`` under an iter's ``active_workspace/``.
    Prints one line per modified node. Returns 0 (the pin is
    non-blocking) unless a graph file cannot be read."""
    graphs_dir = os.path.join(active_ws, "graphs")
    graph_files = sorted(glob.glob(os.path.join(graphs_dir, "**", "*.json"), recursive=True))
    if not graph_files:
        print(f"[pin] no overlay graphs under {graphs_dir} — nothing to pin")
        return 0

    total = 0
    for gp in graph_files:
        try:
            changes = pin_graph(gp)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[pin] ERROR reading {gp}: {e}", file=sys.stderr)
            return 1
        for c in changes:
            model_note = ", dropped dead `model` field" if c["dropped_model"] else ""
            print(
                f"[pin] {os.path.basename(gp)}:{c['node_id']} — "
                f"profile {c['profile_before']!r}->{PINNED_PROFILE!r} "
                f"temperature {c['temperature_before']!r}->{PINNED_TEMPERATURE}"
                f"{model_note}"
            )
        total += len(changes)
    print(
        f"[pin] {total} llmCall node(s) pinned to "
        f"profile={PINNED_PROFILE} temperature={PINNED_TEMPERATURE} "
        f"across {len(graph_files)} graph(s)"
    )
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Pin every llmCall node in an iter's overlay graphs to a single "
            f"profile ({PINNED_PROFILE}, temperature {PINNED_TEMPERATURE})."
        )
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pin", help="pin all graphs/*.json under an active_workspace/")
    p.add_argument("--active-ws", required=True, help="path to iter's active_workspace/")
    args = ap.parse_args()
    if args.cmd == "pin":
        sys.exit(pin_active_workspace(args.active_ws))
