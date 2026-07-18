#!/usr/bin/env bash
# Clean experiment runner for beta-react-harness (mini-swe-agent VLN).
#
# Design rules (from the 2026-07-13 defaults audit):
#   1. ZERO reliance on CLI defaults — every knob is explicit here, encoded in
#      the run name, and written to provenance.json inside the run dir.
#   2. The script OWNS the serving stack: it (re)starts ollama with the pinned
#      context length and keeps a per-run serve.log, so llama.cpp's exact
#      per-request token counts (task.n_tokens) are auditable per run.
#   3. Serving-side state that the harness cannot see (sampling defaults,
#      model digest, context length) is captured into provenance.json —
#      the layer that was invisible in all runs before this script.
#   4. GPU memory is sampled for the whole run; the peak lands in audit.json.
#
# Usage (all knobs env-overridable):
#   EPISODES=0 bash beta-react-harness/run_experiment.sh
#   EPISODES=0-49 SKILL=ledger-nav bash beta-react-harness/run_experiment.sh
#   EPISODES=0-9 TEMPLATE=bare bash beta-react-harness/run_experiment.sh
set -euo pipefail

# ---------------- THE STANDARD PROTOCOL (v1, frozen 2026-07-14) -------------
# Two canonical harness conditions, one standard perception/context config.
# Any deviation from these values gets flagged in the run name (see below), so
# a plain `std_<model>_<mode>` name always means exactly this protocol.
STD_RGB=224                          # render px
STD_IMAGE_WINDOW=4                   # newest K frames kept in the API payload
STD_CTX=131072                       # 128k serve context
STD_MAX_TURNS=100
STD_STEP_BUDGET=500

# MODE = the harness condition:
#   bare : bare prompt, no clearance readout, no turn-budget broadcast,
#          no STOP gate, no look_around, no skill      (the vanilla baseline)
#   nav  : full mechanisms + ledger-nav skill          (the tuned condition)
MODE="${MODE:-nav}"

EPISODES="${EPISODES:?set EPISODES, e.g. 0 or 0-49}"
MODEL="${MODEL:-ollama_chat/qwen3.5:4b-q8_0}"
RGB="${RGB:-$STD_RGB}"
IMAGE_WINDOW="${IMAGE_WINDOW:-$STD_IMAGE_WINDOW}"
CTX="${CTX:-$STD_CTX}"
MAX_TURNS="${MAX_TURNS:-$STD_MAX_TURNS}"
STEP_BUDGET="${STEP_BUDGET:-$STD_STEP_BUDGET}"
EPISODE_TIMEOUT="${EPISODE_TIMEOUT:-2400}"
SPLIT="${SPLIT:-rand100}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:9200}"

case "$MODE" in
  bare) TEMPLATE="bare"; SKILL="" ;;
  nav)  TEMPLATE="mech"; SKILL="${SKILL:-ledger-nav}" ;;
  *)    echo "ERROR: MODE must be 'bare' or 'nav' (got '$MODE')"; exit 2 ;;
esac

OLLAMA_BIN="${OLLAMA_BIN:-$HOME/ollama/bin/ollama}"
PY="${PY:-$HOME/miniconda3/envs/agentcanvas/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- run identity ----------------------------------------------
# std_<model>_<mode> when the protocol is untouched; any deviation is appended
# so a non-standard run can never masquerade as a standard one.
MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||; s|[:.]|-|g')
DEV=""
[ "$RGB" != "$STD_RGB" ] && DEV="${DEV}_rgb${RGB}"
[ "$IMAGE_WINDOW" != "$STD_IMAGE_WINDOW" ] && DEV="${DEV}_w${IMAGE_WINDOW}"
[ "$CTX" != "$STD_CTX" ] && DEV="${DEV}_ctx$((CTX/1024))k"
[ "$MAX_TURNS" != "$STD_MAX_TURNS" ] && DEV="${DEV}_t${MAX_TURNS}"
[ "$STEP_BUDGET" != "$STD_STEP_BUDGET" ] && DEV="${DEV}_sb${STEP_BUDGET}"
[ "$MODE" = "nav" ] && [ "$SKILL" != "ledger-nav" ] && DEV="${DEV}_${SKILL}"
RUN_NAME="${RUN_NAME:-std_${MODEL_SLUG}_${MODE}${DEV}}"
RUN_DIR="$REPO/outputs/beta-react-harness/$RUN_NAME"
mkdir -p "$RUN_DIR"
echo "[exp] run: $RUN_NAME"

# ---------------- serving: owned by this run --------------------------------
echo "[exp] restarting ollama with OLLAMA_CONTEXT_LENGTH=$CTX ..."
pkill -f "[o]llama serve" 2>/dev/null || true
for _ in $(seq 30); do curl -s --max-time 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1 || break; sleep 1; done
OLLAMA_CONTEXT_LENGTH="$CTX" setsid nohup "$OLLAMA_BIN" serve > "$RUN_DIR/serve.log" 2>&1 &
for _ in $(seq 60); do curl -s --max-time 1 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break; sleep 1; done
curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null || { echo "ERROR: ollama did not come up"; exit 3; }

curl -s --max-time 3 -o /dev/null "$SERVER_URL/" || { echo "ERROR: habitat auto_host not reachable at $SERVER_URL"; exit 4; }

# ---------------- provenance (the previously-invisible layer) ---------------
OLLAMA_TAG="${MODEL#ollama_chat/}"
"$PY" - "$RUN_DIR" <<PYEOF
import json, subprocess, sys, urllib.request
run_dir = sys.argv[1]
def sh(*cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception as e: return f"<{e}>"
show = {}
try:
    req = urllib.request.Request("http://127.0.0.1:11434/api/show",
        data=json.dumps({"model": "$OLLAMA_TAG"}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read())
    show = {"parameters": d.get("parameters"),
            "context_length_native": (d.get("model_info") or {}).get(
                next((k for k in (d.get("model_info") or {}) if k.endswith("context_length")), ""), None),
            "quant": (d.get("details") or {}).get("quantization_level"),
            "family": (d.get("details") or {}).get("family")}
except Exception as e:
    show = {"error": repr(e)}
prov = {
    "knobs": {"episodes": "$EPISODES", "model": "$MODEL", "rgb": $RGB,
              "image_window": $IMAGE_WINDOW, "serve_ctx": $CTX,
              "mode": "$MODE", "template": "$TEMPLATE", "skill": "$SKILL" or None,
              "standard_protocol": ("$RUN_NAME".startswith("std_")
                                    and "$RUN_NAME".count("_") == 2),
              "max_turns": $MAX_TURNS, "step_budget": $STEP_BUDGET,
              "episode_timeout": $EPISODE_TIMEOUT, "split": "$SPLIT",
              "server_url": "$SERVER_URL"},
    "serving": {"ollama_version": sh("$OLLAMA_BIN", "--version"),
                "model_list": sh("$OLLAMA_BIN", "list"),
                "api_show": show,
                "note": "sampling params NOT pinned by harness -> Modelfile defaults in api_show.parameters apply"},
    "host": {"git_rev": sh("git", "-C", "$REPO", "rev-parse", "--short", "HEAD"),
             "gpu": sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")},
}
json.dump(prov, open(f"{run_dir}/provenance.json", "w"), indent=1, ensure_ascii=False)
print("[exp] provenance.json written")
PYEOF

# ---------------- VRAM sampler ----------------------------------------------
( set +e; PEAK=0; while :; do V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1) || break
    if [ "${V:-0}" -gt "$PEAK" ] 2>/dev/null; then PEAK=$V; echo "$PEAK" > "$RUN_DIR/vram_peak_mib.txt"; fi
    sleep 0.5; done ) &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null || true' EXIT

# ---------------- the run ----------------------------------------------------
DRIVER_ARGS=( --episodes "$EPISODES" --split "$SPLIT" --model "$MODEL"
              --image-window "$IMAGE_WINDOW" --rgb-resolution "$RGB"
              --max-turns "$MAX_TURNS" --step-budget "$STEP_BUDGET"
              --episode-timeout "$EPISODE_TIMEOUT" --server-url "$SERVER_URL"
              --run-name "$RUN_NAME" )
[ "$TEMPLATE" = "bare" ] && DRIVER_ARGS+=( --bare )
[ -n "$SKILL" ] && DRIVER_ARGS+=( --skill "$SKILL" )

echo "[exp] driver: run_episodes.py ${DRIVER_ARGS[*]}"
MSWEA_COST_TRACKING=ignore_errors "$PY" "$REPO/beta-react-harness/run_episodes.py" \
    "${DRIVER_ARGS[@]}" 2>&1 | tee "$RUN_DIR/driver.log"

# ---------------- post-run audit ---------------------------------------------
"$PY" - "$RUN_DIR" <<'PYEOF'
import json, re, sys
run_dir = sys.argv[1]
toks = [int(m.group(1)) for ln in open(f"{run_dir}/serve.log", errors="ignore")
        if (m := re.search(r"new prompt, n_ctx_slot = \d+, n_keep = \d+, task\.n_tokens = (\d+)", ln))]
try: vram = int(open(f"{run_dir}/vram_peak_mib.txt").read().strip())
except Exception: vram = None
summary = json.load(open(f"{run_dir}/summary.json"))
agg = summary.get("aggregate") or {}
audit = {"llm_requests": len(toks),
         "prompt_tokens": {"peak": max(toks) if toks else 0,
                            "sum": sum(toks), "first": toks[0] if toks else 0},
         "vram_peak_mib": vram,
         "aggregate": agg}
json.dump(audit, open(f"{run_dir}/audit.json", "w"), indent=1)
print("[exp] audit:", json.dumps(audit, indent=1)[:800])
PYEOF
echo "[exp] done -> $RUN_DIR"
