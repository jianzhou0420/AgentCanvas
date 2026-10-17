#!/usr/bin/env python3
"""Sync the run() + seams listings in design-docs/graph/graph-executor.html.

The annotations readers see in the doc ARE the source comments of
``GraphExecutor.run()`` and of the seams it calls (``_fire`` /
``_report_node_error`` / ``_deliver`` / ``_after_fire`` / ``_iter_boundary``
/ ``_settle_drain``) — there is no separate annotation layer to drift.
This script extracts both regions from graph_executor.py, splits them into
bands on the file's own ``# ───`` banner comments and (for the seams) on
each method ``def`` (the shared vocabulary between the file and Part I of
the doc), adds the doc-side § cross-links, and writes the result between
the ``<!-- run-listing:begin/end -->`` markers in the page.

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
    ("in-loop · the scheduling loop", "§4.1\u2013§4.2", "#41-one-iteration"),
    ("iterOut boundary", "§4.3", "#43-iterout-boundary"),
    ("ordinary path", "§4.4", "#44-propagation"),
    ("after-loop", "§5.1", "#51-post-loop-pass"),
    ("conviction", "§5.2", "#52-finalise"),
    ("finalise — clean exit", "§5.2", "#52-finalise"),
    ("finalise — error path", "§5.3", "#53-error-path"),
    # the seams
    ("async def _fire(", "§4.2", "#42-the-gauntlet"),
    ("def _report_node_error(", "§4.2", "#42-the-gauntlet"),
    ("def _deliver(", "§4.4", "#44-propagation"),
    ("def _after_fire(", "§4.4", "#44-propagation"),
    ("async def _iter_boundary(", "§4.3", "#43-iterout-boundary"),
    ("phase 1/4", "§4.3 ①", "#43-iterout-boundary"),
    ("phase 2/4", "§4.3 ②", "#43-iterout-boundary"),
    ("phase 3/4", "§4.3 ③", "#43-iterout-boundary"),
    ("phase 4/4", "§4.3 ④", "#43-iterout-boundary"),
    ("async def _settle_drain(", "§4.3 ②", "#43-iterout-boundary"),
]


def extract_region(start_prefix: str, end_prefix: str) -> tuple[int, list[str]]:
    """Return (1-based start line, lines) from the line starting with
    ``start_prefix`` up to (not including) the line starting with
    ``end_prefix``, trailing blank lines dropped."""
    lines = SRC.read_text().splitlines()
    start = end = None
    for i, ln in enumerate(lines):
        if start is None and ln.startswith(start_prefix):
            start = i
        elif start is not None and ln.startswith(end_prefix) and i > start:
            end = i
            break
    assert start is not None and end is not None, f"region {start_prefix!r} bounds not found"
    while end > start and not lines[end - 1].strip():
        end -= 1
    return start + 1, lines[start:end]


def extract_run() -> tuple[int, list[str]]:
    """``run()`` up to the seams banner that follows it."""
    return extract_region("    async def run(", "    # ── The seams")


def extract_seams() -> tuple[int, list[str]]:
    """The seams banner through ``_settle_drain`` (``_enqueue`` ends it)."""
    return extract_region("    # ── The seams", "    def _enqueue(")


def band_header(title: str, lo: int, hi: int) -> str:
    link = ""
    for prefix, label, anchor in SECTION_MAP:
        if title.startswith(prefix):
            link = f' · ↔ <a href="{anchor}">{label}</a>'
            break
    rng = f"graph_executor.py:{lo}\u2013{hi}"
    return f'<span class="h">{html.escape(title)} <span class="hr">── {rng}{link}</span></span>'


def bands_of(
    start: int, lines: list[str], first_title: str, *, split_defs: bool = False
) -> list[tuple[str, int, list[tuple[int, str]]]]:
    """Split a region into (title, first line, [(line no, rendered)]) bands
    on the file's banner comments (``# ──`` / ``# ───`` lines) and, when
    ``split_defs`` is set, on each method ``def`` (title = its signature
    head). A banner whose first line ends with ":" continues on the next
    line."""
    bands: list[tuple[str, int, list[tuple[int, str]]]] = []
    title, body = first_title, []
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
        if split_defs and not in_doc and re.match(r"    (async )?def \w+\(", ln):
            bands.append((title, t_line, body))
            title = stripped.split("(", 1)[0] + "(…)"
            body, t_line = [], n
            body.append((n, hl[off]))
            continue
        if not in_doc and not body and stripped.startswith("#") and title.endswith(":"):
            # two-line banner: fold the continuation into the title
            title = title[:-1] + " — " + stripped.lstrip("# ").strip("─ ")
            continue
        body.append((n, hl[off]))
    bands.append((title, t_line, body))
    return bands


def render(bands: list[tuple[str, int, list[tuple[int, str]]]]) -> list[str]:
    out = ['<pre class="rxcode hl">']
    for title, t_line, body in bands:
        if not any(r.strip() for _, r in body):
            continue  # a banner with nothing under it (e.g. the seams header)
        hi = body[-1][0]
        out.append(band_header(title, t_line, hi))
        for n, rendered in body:
            out.append(f"{n:<5}{rendered}".rstrip())
    out.append("</pre>")
    return out


def build() -> str:
    start, lines = extract_run()
    s_start, s_lines = extract_seams()

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
    ]
    out += render(bands_of(start, lines, "run() — signature & docstring"))
    out.append(
        '<h3 id="61-the-seams"><span class="num">6.1</span> The seams — <code>_fire</code> · '
        "<code>_deliver</code> · <code>_after_fire</code> · <code>_iter_boundary</code> · "
        "<code>_settle_drain</code></h3>"
    )
    out.append(
        "<p>The methods the scheduling loop calls at fixed points, in source order, with the "
        "same band / § convention. The iterOut boundary's four phases are the "
        "<code># ──</code> banners inside <code>_iter_boundary</code>.</p>"
    )
    out.append(
        f'<p class="codelegend">Synced from <code>graph_executor.py:{s_start}\u2013{s_start + len(s_lines) - 1}</code>'
        f" at <code>{short}</code> by the same script.</p>"
    )
    out += render(bands_of(s_start, s_lines, "The seams", split_defs=True))
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
