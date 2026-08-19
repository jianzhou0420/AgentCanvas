# libero_tb — LIBERO loaded toolbox (GT rung)

Folder = method arm: atomic views + the simulator's GT get_objects readout
+ move_to/gripper macros + native step() escape hatch. Deliberately
PRIVILEGED — asks "does the loaded surface complete tasks at all"; the
minimal-interface story keeps bare/full. Cells `libero_sdk_{trio}_tb`;
shared "libero" frozen; bridge flags BAKED (TOOLBOX=True, TOOLBOX_GT=True).
Board: sonnet eps0-9 = 9/10. Migrated 2026-08-18, byte-parity. Rule: NEVER
edit — fork instead.
