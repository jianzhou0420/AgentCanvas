# hybrid — the agent-selected hybrid interface (MIP §4.5)

Folder = method arm: primitive step(0-3) AND waypoint goto in ONE toolface
(hybrid_bridge copy); the look-then-move gate makes the lens choice the
interface choice. Migrated 2026-08-18, byte-parity. Cells keep their
`std_*_hybrid` names (sdk fable/sonnet · mini qwen 4b/9b); r2r profile
only, STD_FROZEN fallback, WP_MAX_TURNS 100. Servers: env auto_host (this
folder's nodeset = bare's env_habitat copy) + --wp-server, same as wp
(predictor tree: `exp_workspace/wp/ac_wp_predictor_shim/`).
Paper: fable ×3 rep = 76.7±0.6 (tab:hybrid); scripts/analyze_hybrid.py
feeds the behavioral stats. Rule: NEVER edit — fork instead.
