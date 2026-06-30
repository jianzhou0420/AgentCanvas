#!/usr/bin/env python3
"""Sync the run() listing in design-docs/graph/graph-executor.html from source.

The annotations readers see in the doc ARE the source comments of
``GraphExecutor.run()`` — there is no separate annotation layer to drift.
This script extracts run() from graph_executor.py, splits it into bands on
the file's own ``# ───`` banner comments (the shared vocabulary between the
file and Part I of the doc), adds the doc-side § cross-links, and writes
the result between the ``<!-- run-listing:begin/end -->`` markers in the
page.

Two-stage pipeline: stage 1, Pygments tokenizes the source into static
HTML spans (GitHub-palette CSS lives in docs/assets/style.css under .hl); stage 2, this script
decorates — band split, line-number gutter, doc-side section links.

Run from the repo root after any edit to graph_executor.py's run():
    python docs/_lib/_sync_run_listing.py
(any python with pygments works; the agentcanvas env has 2.20)
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hl import highlight_lines

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "agentcanvas/backend/app/agent_loop/graph_executor.py"
PAGE = REPO / "docs/pages/developer-guide/design-docs/graph/graph-executor.html"
BEGIN = "<!-- run-listing:begin -->"
END = "<!-- run-listing:end -->"

# Doc-side cross-links: banner-title prefix → (label, anchor).
# The banner text itself lives in the source file; only the §-mapping is
# doc-side, so renumbering the doc never touches the code.
SECTION_MAP = [
    ("pre-loop · BUILD", "§3.1", "#31-build"),
    ("pre-loop · ENTRY DISCOVERY", "§3.2", "#32-seed"),
    ("in-loop · the firing loop", "§4.1\u2013§4.2", "#41-one-iteration"),
    ("boundary phase 1/4", "§4.3 ①", "#43-iterout-boundary"),
    ("boundary phase 2/4", "§4.3 ②", "#43-iterout-boundary"),
    ("boundary phase 3/4", "§4.3 ③", "#43-iterout-boundary"),
    ("boundary phase 4/4", "§4.3 ④", "#43-iterout-boundary"),
    ("iterOut boundary", "§4.3", "#43-iterout-boundary"),
    ("ordinary path", "§4.4", "#44-propagation"),
    ("after-loop", "§5.1", "#51-post-loop-pass"),
    ("finalise — clean exit", "§5.2", "#52-finalise"),
    ("finalise — error path", "§5.3", "#53-error-path"),
]


def extract_run() -> tuple[int, list[str]]:
    """Return (1-based start line, lines) of run() up to the next method."""
    lines = SRC.read_text().splitlines()
    start = end = None
    for i, ln in enumerate(lines):
        if start is None and ln.startswith("    async def run("):
            start = i
        elif start is not None and re.match(r"    (async )?def \w+", ln) and i > start:
            end = i
            break
    assert start is not None and end is not None, "run() bounds not found"
    while end > start and not lines[end - 1].strip():
        end -= 1
    return start + 1, lines[start:end]


def band_header(title: str, lo: int, hi: int) -> str:
    link = ""
    for prefix, label, anchor in SECTION_MAP:
        if title.startswith(prefix):
            link = f' · ↔ <a href="{anchor}">{label}</a>'
            break
    rng = f"graph_executor.py:{lo}\u2013{hi}"
    return f'<span class="h">{html.escape(title)} <span class="hr">── {rng}{link}</span></span>'


def build() -> str:
    start, lines = extract_run()
    # band split on the file's banner comments (``# ──`` / ``# ───`` lines);
    # a banner whose first line ends with ":" continues on the next line.
    bands: list[tuple[str, int, list[tuple[int, str]]]] = []
    title, body = "run() — signature & docstring", []
    t_line = start
    hl = highlight_lines("\n".join(lines), lang="python")
    in_doc = False
    for off, ln in enumerate(lines):
        n = start + off
        stripped = ln.strip()
        if '"""' in ln and ln.count('"""') % 2 == 1:
            in_doc = not in_doc
        if not in_doc and stripped.startswith("# ─"):
            bands.append((title, t_line, body))
            title = stripped.lstrip("# ").strip("─ ")
            body, t_line = [], n
            continue
        if not in_doc and not body and stripped.startswith("#") and title.endswith(":"):
            # two-line banner: fold the continuation into the title
            title = title[:-1] + " — " + stripped.lstrip("# ").strip("─ ")
            continue
        body.append((n, hl[off]))
    bands.append((title, t_line, body))

    short = "?"
    try:
        short = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet", "--", str(SRC.relative_to(REPO))],
                cwd=REPO,
                capture_output=True,
            ).returncode
            != 0
        )
        if dirty:
            short += "+dirty"
    except Exception:
        pass

    out = [
        f'<p class="codelegend">Synced from <code>graph_executor.py:{start}\u2013{start + len(lines) - 1}</code>'
        f" at <code>{short}</code> by <code>docs/_lib/_sync_run_listing.py</code>"
        " — the band headers are the file's own banner comments; re-run the script after editing the source.</p>",
        '<pre class="rxcode hl">',
    ]
    for title, t_line, body in bands:
        if not body:
            continue
        hi = body[-1][0]
        out.append(band_header(title, t_line, hi))
        for n, rendered in body:
            out.append(f"{n:<5}{rendered}".rstrip())
    out.append("</pre>")
    return "\n".join(out)


def main() -> None:
    page = PAGE.read_text()
    assert BEGIN in page and END in page, "run-listing markers missing in page"
    pre, rest = page.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    new = pre + BEGIN + "\n" + build() + "\n" + END + post
    if new != page:
        PAGE.write_text(new)
        print("run-listing synced")
    else:
        print("run-listing unchanged")


if __name__ == "__main__":
    main()
