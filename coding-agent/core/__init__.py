"""core — everything shared across experiments: the engine (cells registry,
driver episode loop, off-arm prompts, monitor_api contract), the harness
adapters (harnesses/), and the API layer (api/). The variable part —
experiment arms — lives outside in exp_workspace/; entries are
stdrun.py / uirun.py at the package root."""
