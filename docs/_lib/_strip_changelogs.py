#!/usr/bin/env python3
"""Publish-time guard: keep per-page Changelog sections off the doc-site.

Per-page Changelog sections are personal dev history. They live OUTSIDE the
tracked tree as gitignored fragments under `docs/_changelogs/<page-rel>`
(parallel tree mirroring `docs/pages/**`); `assets/nav.js` re-attaches each
fragment to its page's tail on the local dev site. Tracked pages therefore
carry no changelog — this script is the guard that keeps the published site
clean if a section is ever committed into a page again by mistake.

The Pages workflow runs it on the CI checkout *before* `_wrap_handwritten.py`,
so the rebuilt right-TOC and search-index.json are derived from the already-
cleaned `<main>`. Never run it against a working tree you intend to keep —
it edits pages in place. CI mutates only its own checkout (the Pages
artifact); locally the sections live in `docs/_changelogs/`, which this
script never touches.

Two cleanups per `docs/pages/**/*.html`:
  1. Remove any `<h2 id="…changelog">` section — the `<hr>` immediately
     preceding it (if any), the heading, and everything up to the next
     `<h2`, `<nav class="page-nav"`, `<footer`, or `</main>` boundary.
  2. Unwrap same-page anchors pointing at a changelog heading (keep their
     link text) — their target only exists after local fragment injection,
     so on the published page they would be dead links.
Idempotent: a second run finds nothing to clean.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# docs/ root — script lives at docs/_lib/_strip_changelogs.py.
V2 = Path(__file__).resolve().parent.parent

CHANGELOG_H2_RE = re.compile(
    r'<h2[^>]*\bid="[^"]*changelog[^"]*"[^>]*>.*?</h2>', re.DOTALL | re.IGNORECASE
)
SECTION_END_RE = re.compile(
    r'<h2\b|<nav\s+class="page-nav"|<footer\b|</main>', re.IGNORECASE
)
LEADING_HR_RE = re.compile(r"<hr\s*/?>\s*$", re.IGNORECASE)
CHANGELOG_ANCHOR_RE = re.compile(
    r'<a\s[^>]*href="#[^"]*changelog[^"]*"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE
)


def clean_page(text: str) -> str | None:
    """Return the cleaned page text, or None if nothing needed cleaning."""
    original = text
    heading = CHANGELOG_H2_RE.search(text)
    if heading:
        end = SECTION_END_RE.search(text, heading.end())
        if end:  # no </main> after the heading would mean a malformed page
            start = heading.start()
            hr = LEADING_HR_RE.search(text, 0, start)
            if hr:
                start = hr.start()
            text = text[:start] + text[end.start():]
    text = CHANGELOG_ANCHOR_RE.sub(r"\1", text)
    return text if text != original else None


def main() -> int:
    cleaned = []
    for path in sorted((V2 / "pages").rglob("*.html")):
        text = path.read_text(encoding="utf-8")
        new_text = clean_page(text)
        if new_text is None:
            continue
        path.write_text(new_text, encoding="utf-8")
        cleaned.append(path.relative_to(V2))
    for rel in cleaned:
        print(f"cleaned: {rel}")
    print(f"{len(cleaned)} page(s) cleaned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
