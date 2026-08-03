# legacy — frozen pre-unification drivers (provenance; never edited)

Until 2026-07-20 the three harnesses lived as separate repo-root dirs, each
with its own full driver. They were unified into `coding-agent/` (shared core
+ per-harness adapters); the drivers below are kept byte-frozen because:

1. They document exactly how the archived pre-std runs under
   `outputs/beta-*/archive/` were produced.
2. `../mini/check_equivalence.py` imports them as fixtures — the prompt texts
   and tool surface the live path must stay byte-equal to.

`beta-coding-agent/` also keeps `opus-lab/` (the API-capture side lab).

Do not edit anything here, and do not expect it to RUN: in-file path
constants (`beta-coding-agent/mcp_bridge.py`, `skills/`, …) still name the
pre-unification layout. Importing for constants (prompts, templates) is fine —
that is all the equivalence gate does. The READMEs likewise describe the old
layout; the current run recipes live in `../README.md`.
