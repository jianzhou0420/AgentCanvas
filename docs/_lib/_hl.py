#!/usr/bin/env python3
"""Stage-1 highlighter for doc-site code listings (see standard:
.claude/standard/code-highlighting.md).

Tokenizes source offline with Pygments into static HTML spans — one
output line per input line, so stage-2 decorators (line gutters, band
headers, cross-links) can interleave freely. The token classes are
Pygments' short names (``k``/``s2``/``c1``/``nf``/…); their GitHub-Primer
colours live once in ``docs/assets/style.css`` under ``.hl``.

Library use:
    from _hl import highlight_lines
    lines = highlight_lines(text, lang="python")

CLI use (prints a paste-ready ``<pre class="hl">`` block):
    python docs/_lib/_hl.py FILE [--lang python] [--gutter]
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

from pygments.lexers import get_lexer_by_name
from pygments.token import STANDARD_TYPES


def highlight_lines(text: str, lang: str = "python") -> list[str]:
    """Tokenize ``text`` with Pygments; return per-line HTML (spans never
    cross line boundaries)."""
    lexer = get_lexer_by_name(lang)
    lines: list[list[str]] = [[]]
    for tok, val in lexer.get_tokens(text):
        cls = ""
        t = tok
        while t is not None and not cls:
            cls = STANDARD_TYPES.get(t, "")
            t = t.parent
        for k, piece in enumerate(val.split("\n")):
            if k:
                lines.append([])
            if piece:
                esc = html.escape(piece)
                lines[-1].append(f'<span class="{cls}">{esc}</span>' if cls else esc)
    return ["".join(parts) for parts in lines]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", nargs="?", help="source file (default: stdin)")
    ap.add_argument("--lang", default=None, help="pygments lexer name (default: from file suffix)")
    ap.add_argument("--gutter", action="store_true", help="prefix real line numbers")
    args = ap.parse_args()

    if args.file:
        text = Path(args.file).read_text()
        lang = args.lang or {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".js": "javascript",
            ".json": "json",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".html": "html",
        }.get(Path(args.file).suffix, "text")
    else:
        text = sys.stdin.read()
        lang = args.lang or "text"

    out = highlight_lines(text.rstrip("\n"), lang)
    print('<pre class="hl">')
    for i, ln in enumerate(out, 1):
        print(f"{i:<5}{ln}".rstrip() if args.gutter else ln)
    print("</pre>")


if __name__ == "__main__":
    main()
