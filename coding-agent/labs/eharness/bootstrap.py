"""§16.2 — loading the RESIDENT bootstrap artifact, executor-neutral.

The first version of this module fetched its own opening look straight off
the env server — a second, stateless reading of the world whose numbered
candidates lived in the ADAPTER process while goto() answered from the
bridge's toolset. The model could see "place 1" and be told "no places are
current" one call later (§16.2's split-brain).

Now there is exactly ONE first look: the resident toolset (the same
instance that will execute every goto) bootstraps itself — frame-0 fuse,
candidate table, snapshot, archive — and writes ``bootstrap.json`` plus its
PNGs into the episode live dir before the bridge starts serving. This
module only LOADS that artifact and shapes it for a provider message; it
never touches the env and never derives a candidate.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_bootstrap_artifact(live_dir: Path | str | None,
                            ) -> tuple[str, list[bytes]] | None:
    """(text, [png, …]) from the resident session's bootstrap artifact, or
    None when no artifact exists (bridge boot failed / non-dwp line). The
    text is the artifact's own labelled lines — telemetry, candidate list,
    status — exactly what the resident toolset attached to the frame."""
    if live_dir is None:
        return None
    base = Path(live_dir)
    p = base / "bootstrap.json"
    if not p.exists():
        return None
    try:
        art = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    texts = [str(t) for t in (art.get("texts") or []) if str(t).strip()]
    if art.get("candidates"):
        texts.append(
            "candidate_epoch %s · map v%s — these numbers are live: goto(n) "
            "walks to place n." % (art.get("candidate_epoch", "?"),
                                   art.get("map_version", "?")))
    pngs: list[bytes] = []
    # order mirrors the artifact's part order — the joined text's IMAGE 1..4
    # labels only stay honest if the pngs ride in the same sequence
    for key in ("current", "map", "left", "right"):
        fn = (art.get("images") or {}).get(key)
        if not fn:
            continue
        fp = base / str(fn)
        if fp.exists():
            try:
                pngs.append(fp.read_bytes())
            except Exception:  # noqa: BLE001
                continue
    return ("\n".join(texts), pngs)
