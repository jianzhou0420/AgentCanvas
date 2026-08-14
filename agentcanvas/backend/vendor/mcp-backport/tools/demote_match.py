"""Transpile `match` statements to py3.9-compatible if/elif chains.

Handles the pattern subset used by mcp 1.27.0:
  - MatchClass with keyword captures:  case types.Foo(params=params):
  - bare MatchClass:                   case types.Foo():
  - MatchAs over class/or-patterns:    case str() | bytes() as data:
  - MatchValue (literals):             case "endpoint":
  - wildcard:                          case _:
  - guards:                            case X() if cond:
Anything else -> hard error (fix by hand).

Comments inside match bodies are lost (ast.unparse); acceptable for a
vendored port. Run under py>=3.10 (needs ast Match support).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def cond_and_binds(pat: ast.pattern, subj: str) -> tuple[str, list[str]]:
    """Return (condition-expr, bind-statements) for one case pattern."""
    if isinstance(pat, ast.MatchAs):
        if pat.pattern is None:  # wildcard `_` or capture `case x:`
            binds = [f"{pat.name} = {subj}"] if pat.name else []
            return "True", binds
        cond, binds = cond_and_binds(pat.pattern, subj)
        if pat.name:
            binds = [f"{pat.name} = {subj}"] + binds
        return cond, binds
    if isinstance(pat, ast.MatchOr):
        parts = []
        for p in pat.patterns:
            c, b = cond_and_binds(p, subj)
            if b:
                raise SyntaxError("bind inside MatchOr unsupported")
            parts.append(c)
        return "(" + " or ".join(parts) + ")", []
    if isinstance(pat, ast.MatchClass):
        cls = ast.unparse(pat.cls)
        if pat.patterns:
            raise SyntaxError(f"positional class pattern unsupported: {cls}")
        cond = f"isinstance({subj}, {cls})"
        binds = []
        for attr, kp in zip(pat.kwd_attrs, pat.kwd_patterns):
            if isinstance(kp, ast.MatchAs) and kp.pattern is None and kp.name:
                binds.append(f"{kp.name} = {subj}.{attr}")
            elif isinstance(kp, ast.MatchValue):
                cond += f" and {subj}.{attr} == {ast.unparse(kp.value)}"
            else:
                raise SyntaxError(f"nested pattern unsupported in {cls}.{attr}")
        return cond, binds
    if isinstance(pat, ast.MatchValue):
        return f"{subj} == {ast.unparse(pat.value)}", []
    if isinstance(pat, ast.MatchSingleton):
        return f"{subj} is {pat.value!r}", []
    raise SyntaxError(f"unsupported pattern {type(pat).__name__}")


def transpile_match(node: ast.Match, indent: str, tag: int) -> str:
    subj = f"_match_subj_{tag}"
    out = [f"{indent}{subj} = {ast.unparse(node.subject)}"]
    first = True
    for case in node.cases:
        cond, binds = cond_and_binds(case.pattern, subj)
        if case.guard is not None:
            cond = f"({cond}) and ({ast.unparse(case.guard)})"
        kw = "if" if first else "elif"
        first = False
        if cond == "True":
            out.append(f"{indent}else:" if not first else f"{indent}if True:")
            # `else:` only valid when not the first branch
        else:
            out.append(f"{indent}{kw} {cond}:")
        for b in binds:
            out.append(f"{indent}    {b}")
        body_src = "\n".join(ast.unparse(s) for s in case.body)
        for line in body_src.splitlines():
            out.append(f"{indent}    {line}")
    return "\n".join(out)


def process(path: Path) -> int:
    src = path.read_text()
    tree = ast.parse(src)
    matches: list[ast.Match] = [n for n in ast.walk(tree) if isinstance(n, ast.Match)]
    if not matches:
        return 0
    lines = src.splitlines()
    # replace innermost-last so line numbers stay valid
    for tag, node in enumerate(sorted(matches, key=lambda n: -n.lineno)):
        indent = " " * node.col_offset
        repl = transpile_match(node, indent, tag)
        lines[node.lineno - 1 : node.end_lineno] = repl.splitlines()
    path.write_text("\n".join(lines) + "\n")
    return len(matches)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    total = 0
    for f in sorted(root.rglob("*.py")):
        try:
            n = process(f)
        except SyntaxError as e:
            print(f"MANUAL  {f}: {e}")
            continue
        if n:
            print(f"ok      {f.relative_to(root)}: {n} match block(s)")
            total += n
    print(f"total transpiled: {total}")
