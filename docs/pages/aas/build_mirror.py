#!/usr/bin/env python3
"""
build_mirror.py — auto-generate docs/pages/aas/mirror/ from
.claude/commands/architect/**/*.md as a 1:1 HTML mirror.

Usage:
    python docs/pages/aas/build_mirror.py
    # or: cd docs/pages/aas && python build_mirror.py

Behavior:
    - Clears docs/pages/aas/mirror/ and rebuilds it from scratch.
    - One HTML page per source .md, preserving directory structure.
    - YAML frontmatter (name/description/argument-hint/...) rendered
      as a summary card at the top of each page.
    - Intra-mirror .md links auto-rewritten to .html.
    - Generates mirror/index.html with a per-variant tree view.

This script is intentionally dependency-light: only the `markdown`
package on PyPI (CommonMark + GFM tables + fenced code + admonitions).
Install with: `pip install markdown`.
"""

from __future__ import annotations

import re
import shutil
import sys
from html import escape
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.exit(
        "error: the `markdown` package is required.\n       install with: pip install markdown"
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent.parent.parent
SRC = REPO / ".claude" / "commands" / "architect"
DST = REPO / "docs" / "pages" / "aas" / "mirror"


# ---------------------------------------------------------------------------
# Frontmatter parsing (intentionally minimal — no PyYAML)
# ---------------------------------------------------------------------------
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract top-level YAML frontmatter as a flat dict, return (fm, body)."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_text, body = m.group(1), text[m.end() :]
    fm: dict[str, str] = {}
    current_key: str | None = None
    for raw in fm_text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        # only handle top-level "key: value" — nested blocks join to current
        if re.match(r"^[A-Za-z_][\w-]*\s*:", raw):
            k, _, v = raw.partition(":")
            current_key = k.strip()
            fm[current_key] = v.strip()
        elif current_key is not None:
            fm[current_key] += " " + raw.strip()
    return fm, body


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
MD_EXTENSIONS = [
    "extra",  # fenced code, tables, attr_list, def_list, footnotes, abbr
    "sane_lists",
    "toc",
    "admonition",
]

MD_LINK_RE = re.compile(r'(<a[^>]+href="[^"]+?)\.md(#[^"]*)?(")')


def md_to_html(body: str) -> tuple[str, str]:
    """Convert markdown body to HTML; return (body_html, toc_html)."""
    md = markdown.Markdown(extensions=MD_EXTENSIONS, output_format="html5")
    html = md.convert(body)
    return html, getattr(md, "toc", "")


def rewrite_md_links(html: str) -> str:
    """Rewrite intra-doc .md links → .html (only inside <a href> attributes)."""
    return MD_LINK_RE.sub(r"\1.html\2\3", html)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
PAGE_CSS = """\
:root {
  --bg: #fafaf7; --fg: #1a1a1a; --muted: #6b6b6b;
  --accent: #3a4a55; --accent-bg: #eef1f4;
  --frame: #d4d4d0; --code-bg: #f0efe9;
  --link: #1e5a8a;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
               "PingFang SC", "Microsoft YaHei", sans-serif;
  background: var(--bg); color: var(--fg);
  max-width: 1080px; margin: 0 auto; padding: 1.5rem 1.5rem 3rem;
  line-height: 1.6;
}
nav.crumbs {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: .85rem; color: var(--muted);
  padding-bottom: .6rem; border-bottom: 1px solid var(--frame); margin-bottom: 1.5rem;
}
nav.crumbs a { color: var(--accent); text-decoration: none; }
nav.crumbs a:hover { text-decoration: underline; }
nav.crumbs .here { color: var(--fg); font-weight: 600; }
h1 { font-size: 1.75rem; margin: .2rem 0 .8rem; }
h2 {
  font-size: 1.25rem; margin-top: 2.4rem;
  border-bottom: 1px solid var(--frame); padding-bottom: .25rem;
}
h3 { font-size: 1.05rem; margin-top: 1.6rem; color: var(--accent); }
h4 { font-size: .98rem; margin-top: 1.2rem; }
a { color: var(--link); }
code, pre { font-family: "SF Mono", Menlo, Consolas, monospace; }
code { background: var(--code-bg); padding: .05rem .35rem; border-radius: 2px; font-size: .88em; }
pre {
  background: var(--code-bg); padding: .8rem 1rem; border-radius: 3px;
  overflow-x: auto; font-size: .82rem; border-left: 3px solid var(--accent);
  line-height: 1.5;
}
pre code { background: transparent; padding: 0; }
blockquote {
  border-left: 3px solid var(--accent); margin: 1rem 0;
  padding: .2rem 1rem; color: #444; background: var(--accent-bg);
  border-radius: 0 3px 3px 0;
}
table { border-collapse: collapse; margin: .8rem 0; font-size: .9rem; }
th, td { border: 1px solid var(--frame); padding: .45rem .65rem; text-align: left; vertical-align: top; }
th { background: #f0efe9; font-weight: 600; }
ul, ol { padding-left: 1.5rem; }
li { margin-bottom: .25rem; }
.frontmatter {
  background: #fff; border: 1px solid var(--frame);
  border-left: 3px solid var(--accent); border-radius: 3px;
  padding: .8rem 1rem; margin: 0 0 1.5rem; font-size: .9rem;
}
.frontmatter dl { margin: 0; display: grid; grid-template-columns: max-content 1fr; gap: .25rem 1rem; }
.frontmatter dt { font-family: "SF Mono", Menlo, monospace; color: var(--muted); font-size: .82rem; }
.frontmatter dd { margin: 0; }
details.toc {
  background: #fff; border: 1px solid var(--frame); border-radius: 3px;
  padding: .4rem .8rem .6rem; margin: 0 0 1.5rem; font-size: .88rem;
}
details.toc summary { cursor: pointer; color: var(--accent); font-weight: 600; }
details.toc ul { margin: .4rem 0; padding-left: 1.2rem; list-style: none; }
details.toc li { margin: .15rem 0; }
details.toc a { color: var(--link); text-decoration: none; }
details.toc a:hover { text-decoration: underline; }
footer.src {
  margin-top: 3rem; padding-top: .8rem; border-top: 1px solid var(--frame);
  font-size: .78rem; color: var(--muted);
  font-family: "SF Mono", Menlo, monospace;
}
"""

INDEX_CSS = (
    PAGE_CSS
    + """\
.tree { display: grid; grid-template-columns: 1fr; gap: 1.2rem; margin-top: 1.5rem; }
@media (min-width: 720px) { .tree { grid-template-columns: 1fr 1fr; } }
.group {
  background: #fff; border: 1px solid var(--frame);
  border-left: 4px solid var(--accent); border-radius: 3px;
  padding: .9rem 1.1rem;
}
.group h3 { margin: 0 0 .4rem; font-family: "SF Mono", Menlo, monospace; color: var(--accent); }
.group ul { list-style: none; padding-left: 0; margin: 0; }
.group li { margin: .15rem 0; }
.group li a { font-family: "SF Mono", Menlo, monospace; font-size: .88rem; }
.lede {
  background: var(--accent-bg); border-left: 4px solid var(--accent);
  border-radius: 3px; padding: 1rem 1.2rem; margin: 1.5rem 0 0;
}
"""
)


def build_breadcrumbs(rel_html_path: Path) -> str:
    """Path like 'adas/loop.html' → mirror / adas / loop."""
    parts = rel_html_path.parts
    depth = len(parts) - 1  # number of dirs above the file
    root_href = ("../" * depth) + "index.html" if depth else "index.html"
    crumbs = [f'<a href="{root_href}">mirror</a>']
    # intermediate dirs are plain text (no per-dir index page)
    for d in parts[:-1]:
        crumbs.append(f"<span>{escape(d)}</span>")
    crumbs.append(f'<span class="here">{escape(Path(parts[-1]).stem)}</span>')
    return " / ".join(crumbs)


def render_frontmatter(fm: dict[str, str]) -> str:
    if not fm:
        return ""
    rows = []
    for key in ("name", "description", "argument-hint", "allowed-tools", "model"):
        if key in fm:
            rows.append(f"<dt>{escape(key)}</dt><dd>{escape(fm[key])}</dd>")
    # any remaining keys
    for key, val in fm.items():
        if key not in {"name", "description", "argument-hint", "allowed-tools", "model"}:
            rows.append(f"<dt>{escape(key)}</dt><dd>{escape(val)}</dd>")
    if not rows:
        return ""
    return f'<div class="frontmatter"><dl>{"".join(rows)}</dl></div>'


PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<nav class="crumbs">{crumbs}</nav>
{frontmatter}
{toc_block}
<article>
{content}
</article>
<footer class="src">source: {source_path}</footer>
</body>
</html>
"""


def render_toc_block(toc_html: str) -> str:
    """toc_html from markdown lib is already a <div class="toc"><ul>...</ul></div>.
    If empty or trivial (only one <li>), skip."""
    if not toc_html or "<li>" not in toc_html:
        return ""
    # count list items — if less than 3 headings, skip
    if toc_html.count("<li>") < 3:
        return ""
    # strip the wrapping <div class="toc"> the md lib adds, re-wrap into details
    inner = toc_html
    inner = re.sub(r'^\s*<div class="toc">', "", inner)
    inner = re.sub(r"</div>\s*$", "", inner)
    return f'<details class="toc" open><summary>On this page</summary>{inner}</details>'


def render_page(md_path: Path, rel: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    body_html, toc_html = md_to_html(body)
    body_html = rewrite_md_links(body_html)
    rel_source = md_path.relative_to(REPO)
    title_part = fm.get("name") or rel.stem
    return PAGE_TEMPLATE.format(
        title=f"{title_part} — architect mirror",
        css=PAGE_CSS,
        crumbs=build_breadcrumbs(rel),
        frontmatter=render_frontmatter(fm),
        toc_block=render_toc_block(toc_html),
        content=body_html,
        source_path=escape(str(rel_source)),
    )


def render_index(pages: list[tuple[Path, Path, str]]) -> str:
    """pages: list of (rel_html_path, source_md_path, title_from_frontmatter_or_stem)."""
    # group by top-level directory
    groups: dict[str, list[tuple[Path, str]]] = {}
    for rel, _src, title in pages:
        top = rel.parts[0]
        groups.setdefault(top, []).append((rel, title))
    for top in groups:
        groups[top].sort(key=lambda x: x[0].name)

    # render
    group_html_parts = []
    for top in sorted(groups):
        items = groups[top]
        li_parts = []
        for rel, title in items:
            link = f'<a href="{escape(rel.as_posix())}"><code>{escape(rel.stem)}</code></a>'
            # only show a separate title span when it differs from the file stem
            tail = (
                f' <span style="color:#6b6b6b;font-size:.82rem;">— {escape(title)}</span>'
                if title and title != rel.stem
                else ""
            )
            li_parts.append(f"<li>{link}{tail}</li>")
        li_html = "".join(li_parts)
        group_html_parts.append(
            f'<div class="group"><h3>{escape(top)}/</h3><ul>{li_html}</ul></div>'
        )
    groups_block = '<div class="tree">' + "".join(group_html_parts) + "</div>"

    body = f"""\
<header>
  <h1>Architect Markdown Mirror</h1>
  <p style="color:#6b6b6b;margin:0;">Auto-generated 1:1 HTML rendering of
  <code>.claude/commands/architect/</code>. Rebuild with
  <code>python docs/pages/aas/build_mirror.py</code>.</p>
</header>

<div class="lede">
  <strong>This is Part 2 of the AAS site.</strong> For hand-written
  algorithm reference and the files-contract reader's guide, see the
  <a href="../reference/index.html">reference site</a>. Source markdown
  lives in <code>.claude/commands/architect/</code> — that is the source
  of truth; this HTML is derived.
</div>

<h2>Pages by variant</h2>
{groups_block}
"""
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Architect Markdown Mirror</title>
<style>{INDEX_CSS}</style>
</head>
<body>
<nav class="crumbs"><a href="../index.html">aas</a> / <span class="here">mirror</span></nav>
{body}
<footer class="src">generated by docs/pages/aas/build_mirror.py from {escape(str(SRC.relative_to(REPO)))}/</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not SRC.exists():
        sys.exit(f"error: source dir not found: {SRC}")

    # clean rebuild
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    pages: list[tuple[Path, Path, str]] = []
    for md_path in sorted(SRC.rglob("*.md")):
        rel_md = md_path.relative_to(SRC)
        rel_html = rel_md.with_suffix(".html")
        out_path = DST / rel_html
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # title for index: frontmatter name or description; fallback to file stem
        fm, _ = parse_frontmatter(md_path.read_text(encoding="utf-8"))
        title = fm.get("description") or fm.get("name") or rel_html.stem

        out_path.write_text(render_page(md_path, rel_html), encoding="utf-8")
        pages.append((rel_html, md_path, title))

    (DST / "index.html").write_text(render_index(pages), encoding="utf-8")

    print(f"built {len(pages)} pages → {DST.relative_to(REPO)}")
    by_top: dict[str, int] = {}
    for rel, _, _ in pages:
        by_top[rel.parts[0]] = by_top.get(rel.parts[0], 0) + 1
    for top in sorted(by_top):
        print(f"  {top}/  ({by_top[top]} pages)")

    # Re-inject the cross-site nav strip on the freshly regenerated pages.
    # _aas_nav.py walks all of docs/pages/aas/, so it'll also touch hand-written
    # reference/ pages — that's intended (idempotent refresh).
    _AAS_NAV = REPO / "docs" / "_lib" / "_aas_nav.py"
    if _AAS_NAV.exists():
        import subprocess

        subprocess.run([sys.executable, str(_AAS_NAV)], check=False)


if __name__ == "__main__":
    main()
