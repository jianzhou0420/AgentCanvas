"""myloop loop helpers — bootstrap CI, graph_summary renderer, atomic writer."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

# ══════════════════════════════════════════════════════════════════════
# Backend LLM cheat sheet renderer (proposer step 3.5)
# ══════════════════════════════════════════════════════════════════════


_GPT5_FAMILY_PREFIXES = ("gpt-5",)

# Provider → env var the backend's get_llm_config() requires.
# Mirrors agentcanvas/backend/app/llm/providers.py. Profiles whose env var
# is unset return None from get_llm_config() and are de-facto unusable,
# so the cheat sheet filters them out to avoid the proposer choosing
# something that will fail at runtime.
_PROVIDER_ENV: dict = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "togetherai": "TOGETHERAI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "ollama": "",  # ollama needs no key
}


def _temp_constraint(profile_name: str) -> str:
    if any(profile_name.startswith(p) for p in _GPT5_FAMILY_PREFIXES):
        return "gpt-5 family — temperature MUST be 1"
    if profile_name.startswith("gpt-4") or profile_name.startswith("gpt-3"):
        return "accepts temperature 0..2"
    if profile_name.startswith("claude"):
        return "accepts temperature 0..1"
    return "accepts temperature 0..1 (default range)"


def _profile_available(profile: dict) -> bool:
    provider = (profile.get("provider") or "").lower()
    if provider not in _PROVIDER_ENV:
        return False
    env_var = _PROVIDER_ENV[provider]
    if env_var == "":  # ollama — always available
        return True
    return bool(os.environ.get(env_var, "").strip())


def render_backend_llm_cheat_sheet(
    profiles_json_path: str = "agentcanvas/backend/profiles.json",
) -> str:
    """Build the cheat-sheet block to inject into the proposer's base_prompt
    (see proposer.md step 3.5). Reads the live profiles.json and renders
    a Markdown section tagging the active profile + per-family temperature
    constraints.

    Profiles whose provider env var is not set are filtered out — this
    mirrors backend's get_llm_config() gating: a profile with no key
    returns None at runtime, so listing it would mislead the proposer.

    Output is ~30 lines and goes verbatim into base_prompt between the
    upstream "Your task" paragraph and "# Recent behavior summary".
    """
    with open(profiles_json_path) as f:
        p = json.load(f)
    active = p["active"]
    all_profiles = p["profiles"]
    # Filter: only profiles whose provider env var is set
    available = {n: prof for n, prof in all_profiles.items() if _profile_available(prof)}
    names = list(available.keys())
    if not names:
        raise RuntimeError(
            "render_backend_llm_cheat_sheet: no usable profiles (no provider API key set in env)"
        )
    if active not in available:
        # Active profile's key missing — backend will fall back too; keep
        # active label so the proposer knows what's selected, but flag it.
        active_label = f"{active} (UNAVAILABLE — env var missing)"
    else:
        active_label = active
    skipped = [n for n in all_profiles if n not in available]

    lines = [
        "# Backend LLM environment",
        "",
        f"Active default profile: **{active_label}** (used when an llmCall node has",
        'no `profile` field — the runtime calls `get_llm_config("")` which',
        "falls back to the active profile).",
        "",
        f"Available profiles (from `{profiles_json_path}` — filtered to those",
        "whose provider API key is actually set in the backend's env):",
    ]
    width = max(len(n) for n in names)
    for name in names:
        mark = "  ← DEFAULT" if name == active else ""
        lines.append(f"  * {name:<{width}} ({_temp_constraint(name)}){mark}")
    if skipped:
        lines.append("")
        lines.append(
            "Profiles registered but NOT available (provider API key missing — "
            "DO NOT use): " + ", ".join(sorted(skipped))
        )
    lines += [
        "",
        "## llmCall config schema — ONLY these keys are read by the runtime",
        "",
        "| Key | Type | Meaning |",
        "|---|---|---|",
        "| `profile` | str | Profile name from list above. Omit → uses the active default. |",
        "| `temperature` | float | gpt-5 family REQUIRES 1.0; gpt-4o/Claude families accept 0..2 |",
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
        "## Default-use guidance",
        "",
        "- **Prefer leaving `profile` unset** for new llmCall nodes. The active",
        f"  default ({active}) matches what the main planner / existing llmCall",
        "  nodes use — keeps the agent's reasoning on a single consistent model.",
        f'- If you specify explicitly, write `profile: "{active}"` (not `model:`',
        "  anything). Prefer this same active profile unless there is a clear",
        "  reason to deviate.",
        "- Only swap profile when there's a concrete cost / latency / capability",
        "  justification — and state the reason in your `thought`. Examples of",
        "  valid deviation:",
        "  - Cheap per-step text-only sub-task (one-sentence extraction) →",
        "    gpt-4o-mini at temperature 0 (~10x cheaper for tiny outputs)",
        "  - You need temperature=0 deterministic decoding → MUST swap off the",
        "    gpt-5 family",
        "- **Wrong-family + wrong-temperature combinations fail at runtime**",
        "  (`litellm.UnsupportedParamsError`). With strict-mode backend",
        "  (`AGENTCANVAS_STRICT_ERRORS=1`) this kills the run on the first",
        "  failed call instead of silently returning an empty string.",
        "",
    ]
    return "\n".join(lines)


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """Verbatim upstream ADAS utils.py:31."""
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
    return f"95% Bootstrap Confidence Interval: ({ci_lo * 100:.1f}%, {ci_hi * 100:.1f}%), Median: {median * 100:.1f}%"


def render_graph_summary(graph_path: str) -> dict:
    """Render a compact structural snapshot of a graph.json.

    Output schema (per myloop README "Archive contract"):
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
        src = f"{e.get('from_node', e.get('source_node', '?'))}.{e.get('from_port', e.get('source_port', '?'))}"
        dst = f"{e.get('to_node', e.get('target_node', '?'))}.{e.get('to_port', e.get('target_port', '?'))}"
        wires.append({"from": src, "to": dst})
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
    """Move .staging/iter_n/ to iter_n/, append archive line, update trace+lineage."""
    staging = os.path.join(run_dir, ".staging", f"iter_{iter_n}")
    target = os.path.join(run_dir, f"iter_{iter_n}")
    if os.path.exists(target):
        raise RuntimeError(f"Atomic Writer: target {target} already exists")
    if not os.path.exists(staging):
        raise RuntimeError(f"Atomic Writer: staging {staging} missing")
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
