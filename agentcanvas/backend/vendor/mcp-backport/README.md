# mcp-backport — mcp 1.27.0 for Python 3.8/3.9

Vendored port of the [MCP python SDK](https://github.com/modelcontextprotocol/python-sdk)
1.27.0 (MIT) to Python 3.8+, so nodeset envs stuck below the SDK's official
`>=3.10` floor (habitat-sim 0.1.7 / conda-py3.9 habitat stacks) can host the
native `/mcp` projection in-process (`app/server/mcp_projection.py`).

One tree serves both 3.8 and 3.9: every transform is backwards-compatible, and
FastMCP (the only casualty, see below) self-restores on 3.9 where its
`typing_inspection` dependency exists.

```bash
# into any py>=3.8 nodeset env:
pip install /path/to/agentcanvas/backend/vendor/mcp-backport
# pip list shows: mcp 1.27.0+backport38
```

Hub (`agentcanvas`) and other py>=3.10 envs keep upstream `mcp==1.27.0` — this
package is only for py<3.10 hosts.

## What was changed (vs upstream 1.27.0)

Mechanical passes (`tools/`, re-runnable against upstream 1.x updates):

1. `tools/demote_match.py` — `match` -> `if/elif` (8 blocks auto; 3 in
   `server/lowlevel/server.py` hand-rewritten: nested `RequestResponder`
   pattern, str/bytes rebinds).
2. `tools/runtime_passes.py` — everything else scripted:
   - add `from __future__ import annotations` where missing (92 files)
   - runtime `isinstance(x, A | B)` -> tuples (8 sites)
   - value-position / class-base type unions -> `Union[...]` (35 + 12 stmts)
   - `TypeAlias`/`Annotated` imports -> `typing_extensions` (3 + 5 files)
   - value-position builtin/abc generics (`dict[str, Any]`,
     `Iterable[X]`) -> `typing.*` equivalents (24 stmts)

Hand fixes on top:

- 2 parenthesized multi-`with` (py3.8 SyntaxError) unrolled
  (`client/stdio/__init__.py`, `server/__main__.py`).
- 2 **tuple-with semantic bombs** unrolled (`shared/session.py`,
  `shared/memory.py`): `async with (cm1, cm2):` parses on 3.8 as a single
  tuple context manager — compiles clean, dies at runtime with
  `AttributeError: __aexit__`. Grep-proof; found via 3.8-ast scan for
  single-item `With` whose context_expr is a `Tuple`.
- 2 dict merges `d |= {...}` (py3.9 operator) -> `.update()`.
- `types.GenericAlias` (3.9 stdlib) shimmed to `type(typing.List[int])` in
  `fastmcp/utilities/func_metadata.py`.
- `mcp/server/__init__.py`: FastMCP import guarded — **FastMCP is unavailable
  on 3.8** (`typing_inspection` has no 3.8 distribution); `mcp.server.FastMCP`
  is `None` there. The lowlevel Server + streamable HTTP path — all the
  projection needs — is unaffected. On 3.9 FastMCP works normally.
- `fastmcp/__init__.py` version lookup falls back when no dist-info.

Load-bearing runtime dep: `eval_type_backport` — pydantic uses it on <3.10 to
evaluate `str | int` / `list[int]` string annotations, which is what spares
the thousands of model annotations in `types.py` from rewriting.

## Verification (2026-08-14)

- Syntax: 109/109 files compile on 3.8.20 and 3.9.23.
- PoC: real Streamable HTTP server on py3.8.20 and py3.9.23 —
  initialize / tools/list / tools/call round-trip correct.
- Dep resolution on 3.8 lands anyio 4.5.2 (last 3.8 release, exactly meets
  upstream's `>=4.5` floor — no anyio-3 API shimming anywhere).

Client side, auth flows, SSE/stdio transports are ported but not exercised —
this vendor exists for the server-side projection path.

## Maintenance

Pinned to upstream 1.27.0 (matches the framework-wide `mcp 1.x` bound; 2.0
removed the lowlevel Server API `mcp_projection.py` is written against). To
absorb an upstream 1.x patch: fetch the new tree, re-run `tools/`, re-apply
the hand-fix list above, re-run `app/server/test_mcp_projection.py` under 3.8
and 3.9.
