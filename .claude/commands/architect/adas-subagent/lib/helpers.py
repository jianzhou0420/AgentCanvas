"""adas-subagent loop helpers — bootstrap CI, graph_summary renderer, atomic writer."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

# ══════════════════════════════════════════════════════════════════════
# Backend LLM cheat sheet renderer (proposer step 3.5)
# ══════════════════════════════════════════════════════════════════════


def render_backend_llm_cheat_sheet() -> str:
    """Build the llmCall config-schema block injected into the proposer's
    base_prompt (see proposer.md step 3.5).

    Purpose: tell the meta-LLM — inherited from ADAS upstream — exactly
    which `llmCall.config` keys the runtime reads. The classic failure is
    a `model` field (never read; only `profile` is): the iter_1 focus_llm
    bug.

    The block is static — it neither enumerates backend profiles nor
    advises a model. Every llmCall node's profile/temperature is pinned
    deterministically post-edit by `_common/lib/pin_llm_profile.py`
    (implementer step 3a-pin), so the proposer needs no profile guidance.
    Output is ~15 lines, injected verbatim into base_prompt between the
    upstream "Your task" paragraph and "# Recent behavior summary".
    """
    return "\n".join(
        [
            "## llmCall config schema — ONLY these keys are read by the runtime",
            "",
            "| Key | Type | Meaning |",
            "|---|---|---|",
            "| `profile` | str | Backend LLM profile to route to |",
            "| `temperature` | float | Sampling temperature |",
            "| `max_tokens` | int | Output token cap |",
            "| `system_prompt` | str | System message |",
            "| `template` | str | Prompt body with `{port}` placeholders |",
            '| `mode` | "single_turn" \\| "conversation" | Single-shot or multi-turn |',
            "| `n` | int | Number of samples (use n>1 for self-consistency patterns) |",
            "| `stop` | str \\| list[str] | Stop sequences |",
            "",
            "**The `model` field is NOT read by the runtime.** Writing",
            '`"model": "gpt-4o-mini"` does nothing — the call still routes to',
            "the active profile. To route to a specific profile use",
            '`"profile": "gpt-4o-mini"` instead.',
            "",
        ]
    )


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """Bootstrap CI over the mean of `data` (upstream ADAS utils.py:31).

    Returns a (fitness_str, median) pair:
      - fitness_str: human-readable CI string for archive display.
      - median: bare numeric median of the bootstrap means, a fraction
        in [0, 1]. aflow's select_round softmax consumes this directly
        as the per-iter `score`; do NOT rescale it.
    """
    data = np.array(data, dtype=float)
    bootstrap_means = []
    for _ in range(num_bootstrap_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    bootstrap_means = np.array(bootstrap_means)
    lower_p = (1.0 - confidence_level) / 2.0
    upper_p = 1.0 - lower_p
    ci_lo = np.percentile(bootstrap_means, lower_p * 100)
    ci_hi = np.percentile(bootstrap_means, upper_p * 100)
    median = np.median(bootstrap_means)
    fitness_str = (
        f"95% Bootstrap Confidence Interval: "
        f"({ci_lo * 100:.1f}%, {ci_hi * 100:.1f}%), Median: {median * 100:.1f}%"
    )
    return fitness_str, float(median)


def render_graph_summary(graph_path: str) -> dict:
    """Render a compact structural snapshot of a graph.json.

    Output schema (per adas-subagent README "Archive contract"):
      { "nodes": [{"id": ..., "node_type": ...}, ...],
        "wires": [{"from": "src_id.port", "to": "dst_id.port"}, ...],
        "loop":  {"iter_in": <id>, "iter_out": <id>, "step_budget": int} }
    """
    with open(graph_path) as f:
        g = json.load(f)
    nodes = []
    for n in g.get("nodes", []):
        node_type = n.get("node_type") or n.get("type")
        entry = {"id": n["id"], "node_type": node_type}
        if n.get("config"):
            keep = {}
            for k in ("model", "system_prompt", "max_steps", "step_budget", "temperature"):
                if k in n["config"]:
                    val = n["config"][k]
                    if isinstance(val, str) and len(val) > 200:
                        val = val[:200] + "..."
                    keep[k] = val
            if keep:
                entry["config"] = keep
        nodes.append(entry)
    wires = []
    for e in g.get("edges", []):
        src_node = e.get("source") or e.get("from_node") or e.get("source_node") or "?"
        src_port = e.get("sourceHandle") or e.get("from_port") or e.get("source_port") or "?"
        dst_node = e.get("target") or e.get("to_node") or e.get("target_node") or "?"
        dst_port = e.get("targetHandle") or e.get("to_port") or e.get("target_port") or "?"
        wires.append({"from": f"{src_node}.{src_port}", "to": f"{dst_node}.{dst_port}"})
    loop = {}
    for n in g.get("nodes", []):
        nt = n.get("node_type") or n.get("type")
        if nt == "iterIn":
            loop["iter_in"] = n["id"]
        elif nt == "iterOut":
            loop["iter_out"] = n["id"]
    if "step_budget" in g:
        loop["step_budget"] = g["step_budget"]
    return {"nodes": nodes, "wires": wires, "loop": loop}


def build_acc_list_from_export(export_path: str, primary_metric: str) -> list:
    with open(export_path) as f:
        exp = json.load(f)
    eps = exp.get("episodes", [])
    acc = []
    for ep in eps:
        m = ep.get("metrics", {})
        if primary_metric in m:
            acc.append(m[primary_metric])
    return acc


def write_evaluator_staging(
    staging_dir: str,
    run_id: str,
    acc_list: list,
    primary_metric: str,
    secondary_metrics: dict,
    fitness_str: str,
    export_path: str | None = None,
    summary_csv_path: str | None = None,
):
    Path(staging_dir).mkdir(parents=True, exist_ok=True)
    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0
    metrics = {
        "fitness_str": fitness_str,
        "primary_metric": primary_metric,
        "primary_metric_value": mean_acc,
        "secondary_metrics": secondary_metrics,
        "acc_list_len": len(acc_list),
        "run_id": run_id,
        "episode_count": len(acc_list),
        "primary_metric_definition": f"{primary_metric} (per {primary_metric} field in exp.yaml)",
    }
    with open(os.path.join(staging_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(staging_dir, "eval_run_id.txt"), "w") as f:
        f.write(run_id + "\n")
    if export_path and os.path.exists(export_path):
        shutil.copy(export_path, os.path.join(staging_dir, "export.json"))
    if summary_csv_path and os.path.exists(summary_csv_path):
        shutil.copy(summary_csv_path, os.path.join(staging_dir, "summary.csv"))
    return metrics


def atomic_commit(run_dir: str, iter_n: int, archive_entry: dict, graph_name: str) -> str:
    """Move .staging/iter_n/ to iteration/iter_n/, append archive line, update trace+lineage.

    Target path follows files-contract (2026-05-15): per-iter dirs live under
    `v{N}/iteration/iter_M/`, not flat at `v{N}/iter_M/`.
    """
    staging = os.path.join(run_dir, ".staging", f"iter_{iter_n}")
    target_parent = os.path.join(run_dir, "iteration")
    target = os.path.join(target_parent, f"iter_{iter_n}")
    if os.path.exists(target):
        raise RuntimeError(f"Atomic Writer: target {target} already exists")
    if not os.path.exists(staging):
        raise RuntimeError(f"Atomic Writer: staging {staging} missing")
    os.makedirs(target_parent, exist_ok=True)
    shutil.move(staging, target)
    # Strip scaffolding fields per upstream search.py:232-235
    entry = {k: v for k, v in archive_entry.items() if k not in ("reflection", "debug_thought")}
    archive_path = os.path.join(run_dir, "archive.jsonl")
    with open(archive_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return target


def append_reference_seed(run_dir: str, entry: dict):
    """Append a reference-pattern entry (fitness:null, iter_id:null)."""
    archive_path = os.path.join(run_dir, "archive.jsonl")
    with open(archive_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def update_trace_md(
    run_dir: str,
    iter_n: int,
    name: str,
    fitness_str: str,
    primary_metric: str,
    primary_value: float,
    secondary_metrics: dict,
):
    path = os.path.join(run_dir, "trace.md")
    header_needed = not os.path.exists(path) or os.path.getsize(path) == 0
    cols = ["iter", "name", primary_metric, *secondary_metrics.keys(), "fitness_str"]
    with open(path, "a") as f:
        if header_needed:
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        row = [f"iter_{iter_n}", name, f"{primary_value:.3f}"]
        for k in secondary_metrics:
            v = secondary_metrics[k]
            row.append(f"{v:.3f}" if isinstance(v, (int, float)) else str(v))
        row.append(fitness_str.replace("|", "\\|"))
        f.write("| " + " | ".join(row) + " |\n")


def append_lineage_md(run_dir: str, iter_n: int, section_md: str):
    path = os.path.join(run_dir, "lineage.md")
    with open(path, "a") as f:
        f.write(f"\n## iter_{iter_n}\n\n{section_md}\n")
