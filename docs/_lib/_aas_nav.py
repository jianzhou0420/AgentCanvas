#!/usr/bin/env python3
"""Inject a cross-site nav strip into every page under docs/pages/aas/.

The AAS site (formerly `architect-site/`, moved into `docs/pages/aas/`) keeps
its own internal layout — 1080px max-width body, its own CSS, its own
breadcrumbs. We do NOT graft the docs/ sidebar/breadcrumbs onto it.
What we do graft is a thin sticky strip at the very top of `<body>`
with three tabs back to the rest of the site (Developer Guide /
Research / AAS — AAS marked active).

Idempotent: the strip is wrapped in a `<!-- aas-cross-nav: v1 -->` /
`<!-- /aas-cross-nav -->` block; reruns replace the block in place.

Usage:
    python docs/_lib/_aas_nav.py

Also called from docs/pages/aas/build_mirror.py after mirror/ is regenerated
so the regenerated pages keep the strip.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import _nav

# docs/ root — script lives at docs/_lib/_aas_nav.py
V2 = Path(__file__).resolve().parent.parent
AAS_DIR = V2 / "pages" / "aas"

OPEN_MARK = "<!-- aas-cross-nav: v1 -->"
CLOSE_MARK = "<!-- /aas-cross-nav -->"

# Match a previously injected block (idempotency).
BLOCK_RE = re.compile(
    re.escape(OPEN_MARK) + r".*?" + re.escape(CLOSE_MARK) + r"\s*",
    re.DOTALL,
)
BODY_OPEN_RE = re.compile(r"(<body[^>]*>)", re.IGNORECASE)


def _strip_block(html_path: Path, text: str) -> str:
    """Return text with any previously injected block removed."""
    return BLOCK_RE.sub("", text)


def _build_block(prefix: str) -> str:
    """The injected strip. `prefix` is a relative path from the page
    directory to the docs/ root (with trailing slash, e.g. '../' or '../../').

    Tab list is sourced from _nav.get_tabs() so the strip reflects whatever
    is currently in docs/pages/. Tabs whose dir is absent locally are
    silently omitted; the AAS tab is marked active."""
    tab_links: list[str] = []
    for tab in _nav.get_tabs():
        active = ' class="active"' if tab.get("_dir") == "aas" else ""
        href = f"{prefix}{tab['landing']}"
        tab_links.append(f'    <a{active} href="{href}">{html.escape(tab["tab"])}</a>')
    tabs_block = "\n".join(tab_links)

    return f"""{OPEN_MARK}
<script>
  // Match the site-wide theme choice (shared 'site-theme' key) before first
  // paint, so an AAS page opened from a dark main-site session renders dark.
  (function() {{
    var t;
    try {{ t = localStorage.getItem('site-theme'); }} catch (e) {{}}
    if (!t) {{ t = (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light'; }}
    document.documentElement.setAttribute('data-theme', t);
  }})();
</script>
<nav class="aas-xnav" aria-label="Site sections">
  <a class="aas-xnav-brand" href="{prefix}index.html">AgentCanvas</a>
  <div class="aas-xnav-tabs">
{tabs_block}
  </div>
  <button class="aas-xnav-theme" type="button" aria-label="Toggle light / dark theme"
    onclick="(function(){{var d=document.documentElement;var n=d.getAttribute('data-theme')==='dark'?'light':'dark';d.setAttribute('data-theme',n);try{{localStorage.setItem('site-theme',n);}}catch(e){{}}}})()">&#9680;</button>
</nav>
<style>
.aas-xnav {{
  background: #1a1a1a; color: #fafaf7;
  padding: .55rem 1.5rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
               "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: .88rem;
  display: flex; align-items: center; gap: 1.6rem;
  position: sticky; top: 0; z-index: 1000;
  margin: 0; box-shadow: 0 1px 0 rgba(0,0,0,.15);
}}
.aas-xnav .aas-xnav-brand {{
  color: #fafaf7; text-decoration: none; font-weight: 700;
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: .92rem;
}}
.aas-xnav .aas-xnav-tabs {{ display: flex; gap: .4rem; }}
.aas-xnav .aas-xnav-tabs a {{
  color: #fafaf7; text-decoration: none; opacity: .65;
  padding: .25rem .65rem; border-radius: 3px;
  transition: opacity .12s, background .12s;
}}
.aas-xnav .aas-xnav-tabs a:hover {{ opacity: 1; background: rgba(255,255,255,.06); }}
.aas-xnav .aas-xnav-tabs a.active {{
  opacity: 1; font-weight: 600; background: rgba(255,255,255,.12);
}}
.aas-xnav-theme {{
  margin-left: auto;
  background: transparent; border: 1px solid rgba(255,255,255,.28);
  color: #fafaf7; cursor: pointer; border-radius: 3px;
  padding: .15rem .55rem; font-size: .95rem; line-height: 1;
}}
.aas-xnav-theme:hover {{ background: rgba(255,255,255,.1); }}

/* Dark theme for the AAS subsite. These pages are self-contained (own
   <head>/<style>, no shared style.css) but all share the same CSS var
   names + boilerplate classes, so one override block covers the whole
   subtree. It lives in <body> (injected after <head>), so it wins on
   source order for equal-specificity rules. Per-page saturated --accent
   text is left as-is; only the light --accent-bg tint is neutralised. */
[data-theme="dark"] {{
  --bg: #1a1a1c; --fg: #e8e8e6; --muted: #8a8a86;
  --frame: #3a3a3e; --frame-strong: #cfcfcc;
  --box: #232328; --code-bg: #2a2a2e;
  --accent-bg: #2b2b32; --link: #6ba6dc;
}}
[data-theme="dark"] th {{ background: #26262b; }}
[data-theme="dark"] .frontmatter {{ background: #232328; }}
[data-theme="dark"] details.toc {{ background: #232328; }}
[data-theme="dark"] .group {{ background: #232328; }}
[data-theme="dark"] blockquote {{ color: #c0c0bc; }}
</style>
{CLOSE_MARK}
"""


def _prefix_for(page: Path) -> str:
    """Relative path from `page`'s directory to docs/ root, with trailing slash.
    A page at docs/pages/aas/index.html gets '../../'; deeper pages get
    '../../../' etc."""
    rel_to_v2 = page.relative_to(V2)
    depth = len(rel_to_v2.parts) - 1  # number of directories above the file
    if depth == 0:
        return ""
    return "../" * depth


def inject(page: Path) -> bool:
    """Inject (or refresh) the cross-nav block into `page`. Returns True on success."""
    try:
        text = page.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  ERROR {page.relative_to(V2)}: read: {e}")
        return False

    cleaned = _strip_block(page, text)
    block = _build_block(_prefix_for(page))

    m = BODY_OPEN_RE.search(cleaned)
    if not m:
        # No <body> — skip (probably a fragment / partial). Should not happen
        # for our content but be defensive.
        print(f"  skip (no <body>): {page.relative_to(V2)}")
        return False

    new = cleaned[: m.end()] + "\n" + block + cleaned[m.end() :]

    if new == text:
        return True  # already up-to-date, no write

    try:
        page.write_text(new, encoding="utf-8")
    except Exception as e:
        print(f"  ERROR {page.relative_to(V2)}: write: {e}")
        return False
    return True


def main() -> None:
    if not AAS_DIR.exists():
        print(f"no AAS dir at {AAS_DIR}")
        return
    pages = sorted(AAS_DIR.rglob("*.html"))
    ok = 0
    skipped = 0
    for p in pages:
        if inject(p):
            ok += 1
        else:
            skipped += 1
    print(
        f"aas-cross-nav: {ok} pages updated, {skipped} skipped (root: {AAS_DIR.relative_to(V2.parent)})"
    )


if __name__ == "__main__":
    main()
