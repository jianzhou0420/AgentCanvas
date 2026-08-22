#!/bin/bash
# eharness-evo overnight campaign c1: parent allin_blocked01 (to 50 eps) -> candidate g2 (50 eps)
# -> gate -> promote -> engineer proposes gen2 child -> its board ... up to 4 generations.
export MINI_SERVE_CTX=65536
PY=$HOME/miniconda3/envs/agentcanvas/bin/python
cd /home/xunyi/Desktop/Projects/AgentCanvas
R=outputs/beta-react-harness/_campaigns/evo9b_c1
if [ ! -f $R/manifest.json ]; then
  $PY coding-agent/labs/evolution/supervisor.py init $R --parent evo9b_allin_blocked01 --candidate evo9b_g2 --episodes 0-49 --max-generations 4 --engine sdk
fi
echo "=== [$(date '+%F %T')] campaign run start ==="
$PY coding-agent/labs/evolution/supervisor.py run $R --engine sdk --max-hours 14
echo "=== [$(date '+%F %T')] campaign run exit ==="
