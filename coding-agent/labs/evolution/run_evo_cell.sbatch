#!/bin/bash
#SBATCH -A od-240014
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 2:00:00
#SBATCH -J evo_cell
#SBATCH -o /scratch3/lin256/xunyi/logs/evo_cell-%j.out
#
# eharness-evo student board on a virga GPU node — SAME code, SAME settings as
# the local 5090 runs. Derived from logs/run_cell.sbatch (keep its gotchas),
# trimmed to what a mini+qwen3.5-9b bare-family cell needs: no judge, no SAM,
# no waypoint predictor. Two cards: last card = env_habitat, the rest = ollama.
#
#   sbatch --export=ALL,CELL=evo9b_allin_blocked01,EPISODES=0-24 run_evo_cell.sbatch
#   sbatch --export=ALL,CELL=evo9b_g2,EPISODES=0-24,EVO_ROOT=/scratch3/lin256/xunyi/AgentCanvas-evo run_evo_cell.sbatch
#
# Parity contract (checked at the top of the log):
#   - git commit of EVO_ROOT must equal the local commit the board is compared to
#   - ollama model qwen3.5:9b-bf16-std, Modelfile sampling seed 42/temp 1/top_k 20/top_p .95
#     (verified identical on both hosts 2026-08-22; the adapter prints the read-back)
#   - MINI_SERVE_CTX=65536 on BOTH hosts (local box is RAM-bound; keep served_context equal)
#   - cell registry decides every other knob (200 turns, 512 px, 500 actions, 2400 s)
set -u
X=/scratch3/lin256/xunyi
source $X/env.sh
EVO_ROOT=${EVO_ROOT:-$AC_ROOT}          # a worktree of dev/coding-agent-xunyi; defaults to AC_ROOT
CELL=${CELL:-}
EPISODES=${EPISODES:-}
export MINI_SERVE_CTX=${MINI_SERVE_CTX:-65536}
export OLLAMA_MODELS=$X/ollama/models
export HF_HOME=$X/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NG=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
AUX_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d, -f$NG)     # last card: habitat
LLM_GPUS=${CUDA_VISIBLE_DEVICES%,$AUX_GPU}
export OLLAMA_GPUS=$LLM_GPUS
export OLLAMA_LLM_LIBRARY=cuda_v13
LOG=$X/logs
J=${SLURM_JOB_ID:-local}

echo "== $(date) node=$(hostname) job=$J cell=$CELL episodes=${EPISODES:-all} gpus=$CUDA_VISIBLE_DEVICES aux=$AUX_GPU ctx=$MINI_SERVE_CTX"
echo "== code: $EVO_ROOT @ $(git -C $EVO_ROOT rev-parse --short HEAD) ($(git -C $EVO_ROOT branch --show-current)) dirty=$(git -C $EVO_ROOT status --short | wc -l)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

PIDS=()
cleanup() { echo "== cleanup $(date)"; for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; pkill -f "ollama serve" 2>/dev/null; sleep 2; }
trap cleanup EXIT

# env_habitat auto_host from the bare arm's nodeset (same module the local 9200 serves)
cd $EVO_ROOT/agentcanvas/backend
CUDA_VISIBLE_DEVICES=$AUX_GPU env PYTHONPATH=$PWD:$PWD/../..:$EVO_ROOT/coding-agent \
  $X/conda-envs/ac-vlnce/bin/python -m app.server.auto_host \
  --module exp_workspace.bare.nodeset --class EnvHabitatNodeSet --port 9200 \
  > $LOG/env_habitat-$J-9200.log 2>&1 &
PIDS+=($!)

# ollama pre-start with the posture the adapter pins (ctx = MINI_SERVE_CTX) so it adopts it
SERVE_LOG=$EVO_ROOT/outputs/beta-react-harness/_ollama_serve.log
mkdir -p "$(dirname $SERVE_LOG)"
export OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=2h OLLAMA_CONTEXT_LENGTH=$MINI_SERVE_CTX
CUDA_VISIBLE_DEVICES=$LLM_GPUS $X/ollama/bin/ollama serve >> $SERVE_LOG 2>&1 &
for i in $(seq 1 30); do curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
t0=$(date +%s)
curl -s -m 900 http://127.0.0.1:11434/api/chat -d '{"model":"qwen3.5:9b-bf16-std","stream":false,"think":false,"keep_alive":"2h","messages":[{"role":"user","content":"hi"}]}' >/dev/null
echo "  [warm] qwen3.5:9b-bf16-std in $(( $(date +%s) - t0 ))s"

for i in $(seq 1 300); do curl -sf -m 3 http://127.0.0.1:9200/health >/dev/null 2>&1 && { echo "  [ok] env_habitat up after ${i}s"; break; }; sleep 1; done
curl -sf -m 3 http://127.0.0.1:9200/health >/dev/null || { echo "== env_habitat not healthy; see $LOG/env_habitat-$J-9200.log"; exit 1; }

[ -n "$CELL" ] || { echo "CELL not set"; exit 2; }
cd $EVO_ROOT
echo "== $(date) launching: stdrun.py run $CELL ${EPISODES:+--episodes $EPISODES}"
$X/conda-envs/agentcanvas/bin/python coding-agent/stdrun.py run "$CELL" --servers http://127.0.0.1:9200 ${EPISODES:+--episodes "$EPISODES"}
rc=$?
echo "== $(date) stdrun EXIT=$rc"
exit $rc
