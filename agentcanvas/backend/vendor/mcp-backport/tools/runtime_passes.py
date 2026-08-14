"""Mechanical py3.8 passes for the mcp backport (run after demote_match.py).

Usage: python runtime_passes.py <path-to-mcp-package-dir>

Stages (idempotent, in order):
  1. add `from __future__ import annotations` where missing
  2. runtime `isinstance/issubclass(x, A | B)` -> tuple form
  3. type unions in VALUE positions (Assign/AnnAssign values, class bases)
     -> `Union[...]`, adding the `Union` import when needed
  4. `TypeAlias` / `Annotated` moved from `typing` to `typing_extensions`
  5. builtin/abc generic subscripts in VALUE positions (`dict[str, Any]`,
     `Iterable[X]`) -> `typing.*` equivalents, adding `import typing`

Hand fixes NOT covered here (see README): match blocks demote_match.py
rejects, parenthesized/tuple multi-`with`, dict `|=` merges, the
GenericAlias shim, the FastMCP import guard, the version fallback.

Run under py>=3.10 (operates on upstream source, which needs a 3.10 parser).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

MOVE_TO_TYPING_EXTENSIONS = {"TypeAlias", "Annotated", "ParamSpec", "TypeGuard", "Concatenate"}
BUILTIN_GENERICS = {"list": "List", "dict": "Dict", "tuple": "Tuple", "set": "Set",
                    "frozenset": "FrozenSet", "type": "Type"}
ABC_GENERICS = {"Iterable", "Callable", "Awaitable", "AsyncIterator", "AsyncIterable",
                "Sequence", "Mapping", "MutableMapping", "Iterator", "Generator",
                "AsyncGenerator", "Coroutine"}


def is_typeish(n: ast.expr) -> bool:
    if isinstance(n, ast.Constant) and n.value is None:
        return True
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
        return is_typeish(n.left) and is_typeish(n.right)
    return isinstance(n, (ast.Name, ast.Attribute, ast.Subscript))


class UnionRewriter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.hits = 0

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if isinstance(node.op, ast.BitOr) and is_typeish(node.left) and is_typeish(node.right):
            parts: list[ast.expr] = []

            def flat(n: ast.expr) -> None:
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
                    flat(n.left)
                    flat(n.right)
                elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "Union":
                    elts = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
                    parts.extend(elts)
                else:
                    parts.append(n)

            flat(node)
            self.hits += 1
            return ast.Subscript(value=ast.Name(id="Union", ctx=ast.Load()),
                                 slice=ast.Tuple(elts=parts, ctx=ast.Load()), ctx=ast.Load())
        return node


class GenericRewriter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.hits = 0

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            if node.value.id in BUILTIN_GENERICS:
                node.value = ast.Attribute(value=ast.Name(id="typing", ctx=ast.Load()),
                                           attr=BUILTIN_GENERICS[node.value.id], ctx=ast.Load())
                self.hits += 1
            elif node.value.id in ABC_GENERICS:
                node.value = ast.Attribute(value=ast.Name(id="typing", ctx=ast.Load()),
                                           attr=node.value.id, ctx=ast.Load())
                self.hits += 1
        return node


def value_targets(tree: ast.Module):
    for stmt in ast.walk(tree):
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
            yield stmt.value
        elif isinstance(stmt, ast.ClassDef):
            yield from stmt.bases
            for k in stmt.keywords:
                yield k.value


def splice(lines: list[str], edits: list[tuple[int, int, int, int, str]]) -> None:
    for (l1, c1, l2, c2, txt) in sorted(edits, key=lambda e: (-e[0], -e[1])):
        pre, post = lines[l1 - 1][:c1], lines[l2 - 1][c2:]
        lines[l1 - 1:l2] = [pre + txt + post]


def rewrite_values(f: Path, transformer_cls) -> int:
    src = f.read_text()
    lines = src.splitlines()
    edits = []
    for expr in value_targets(ast.parse(src)):
        rw = transformer_cls()
        new = rw.visit(expr)
        if rw.hits:
            ast.fix_missing_locations(new)
            edits.append((expr.lineno, expr.col_offset, expr.end_lineno,
                          expr.end_col_offset, ast.unparse(new)))
    if not edits:
        return 0
    splice(lines, edits)
    f.write_text("\n".join(lines) + "\n")
    return len(edits)


def ensure_line(f: Path, needle_re: str, insert_after: str, line: str) -> None:
    src = f.read_text()
    if re.search(needle_re, src, re.M):
        return
    # keep aliased future-imports intact: match to end-of-line, then append
    m = re.search(r"^from __future__ import annotations.*$", src, re.M)
    if m:
        src = src[:m.end()] + "\n" + line + src[m.end():]
        f.write_text(src)


def add_future_import(f: Path) -> bool:
    src = f.read_text()
    if "from __future__ import annotations" in src or f.name == "__main__.py":
        return False
    lines = src.splitlines(keepends=True)
    i = 0
    if lines and re.match(r'\s*("""|\'\'\')', lines[0] or ""):
        q = lines[0].strip()[:3]
        if lines[0].count(q) >= 2 and len(lines[0].strip()) > 3:
            i = 1
        else:
            for j in range(1, len(lines)):
                if q in lines[j]:
                    i = j + 1
                    break
    f.write_text("".join(lines[:i]) + "from __future__ import annotations\n" + "".join(lines[i:]))
    return True


def fix_isinstance_unions(f: Path) -> int:
    src = f.read_text()
    pat = re.compile(r"\b(isinstance|issubclass)\(([^()]+?),\s*([A-Za-z_][\w.]*(?:\s*\|\s*[A-Za-z_][\w.]*)+)\)")
    hits = len(pat.findall(src))
    if hits:
        src = pat.sub(lambda m: f"{m.group(1)}({m.group(2)}, ({', '.join(p.strip() for p in m.group(3).split('|'))}))", src)
        f.write_text(src)
    return hits


def move_typing_names(f: Path) -> list[str]:
    src = f.read_text()
    m = re.search(r"^from typing import ([^\n(]+|\([^)]+\))$", src, re.M)
    if not m:
        return []
    names = [x.strip() for x in m.group(1).strip("()").replace("\n", ",").split(",") if x.strip()]
    moved = [x for x in names if x in MOVE_TO_TYPING_EXTENSIONS]
    if not moved:
        return []
    kept = [x for x in names if x not in MOVE_TO_TYPING_EXTENSIONS]
    repl = ("from typing import " + ", ".join(kept) + "\n" if kept else "")
    repl += "from typing_extensions import " + ", ".join(moved)
    f.write_text(src[:m.start()] + repl + src[m.end():])
    return moved


def main(root: Path) -> None:
    files = sorted(root.rglob("*.py"))
    stats = {"future": 0, "isinstance": 0, "unions": 0, "typing_moves": 0, "generics": 0}
    for f in files:
        stats["future"] += add_future_import(f)
    for f in files:
        stats["isinstance"] += fix_isinstance_unions(f)
    for f in files:
        n = rewrite_values(f, UnionRewriter)
        if n:
            ensure_line(f, r"^from typing import .*\bUnion\b|^import typing$", "", "from typing import Union")
        stats["unions"] += n
    for f in files:
        stats["typing_moves"] += len(move_typing_names(f))
    for f in files:
        n = rewrite_values(f, GenericRewriter)
        if n:
            ensure_line(f, r"^import typing$", "", "import typing")
        stats["generics"] += n
    print(stats)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
