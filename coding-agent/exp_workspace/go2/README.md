# go2 — bare surface on the real Unitree Go2 (MIP §4.8 / App D)

Folder = method arm: go2_bridge + go2_host copies (the host runs on the
robot's machine — CycloneDDS is layer-2). Cells `go2_sdk_{trio}` unchanged;
STD_FROZEN posture via benchmark r2r; instruction is operator-supplied
(--set instruction=...), driver skips evaluate (human judges success from
the recording). Migrated 2026-08-18, byte-parity. Robot facts (hosts, DDS
quirks, camera geometry): memory nodeset/project_go2_*. Rule: NEVER edit —
fork instead.
