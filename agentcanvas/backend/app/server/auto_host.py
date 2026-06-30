"""Auto-host CLI — launch any BaseNodeSet in server mode.

Usage (single-file nodeset)::

    python -m app.server.auto_host \\
        --file /path/to/workspace/nodesets/sam.py \\
        --class SamNodeSet \\
        --port 9200

Usage (package nodeset, e.g. ``policy_vla/__init__.py``)::

    python -m app.server.auto_host \\
        --module workspace.nodesets.server.policy_vla \\
        --class PolicyVlaNodeSet \\
        --port 9200

The process exposes ``GET /manifest``, ``POST /call/{fn}``, and
``GET /health`` — the standard AgentCanvas manifest protocol.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import sys


def _import_class(
    *,
    file_path: str | None,
    module_name: str | None,
    class_name: str,
) -> type:
    """Import a class either by dotted module name or by file path.

    ``module_name`` (preferred for package nodesets) takes precedence —
    it goes through the normal import system so relative/absolute imports
    inside the package resolve. ``file_path`` is the fallback for
    single-file nodesets where the dotted path is not known.
    """
    if module_name:
        mod = importlib.import_module(module_name)
    elif file_path:
        spec = importlib.util.spec_from_file_location("_auto_nodeset", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from: {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_auto_nodeset"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        raise ImportError("Either --module or --file is required")
    cls = getattr(mod, class_name, None)
    if cls is None:
        target = module_name or file_path
        raise ImportError(f"Class {class_name} not found in {target}")
    return cls


def _arm_pdeathsig() -> None:
    """Linux ``PR_SET_PDEATHSIG``: SIGTERM when our parent dies.

    Belt-and-suspenders pair with ``base_server._preexec_setsid_pdeathsig``:
    that one arms the spawn-side shell wrapper before exec; this one
    re-arms after the python interpreter has settled, covering any
    layer of process plumbing (shell tail-exec, conda wrapper) between
    uvicorn and us. Either layer dying triggers our SIGTERM.
    No-op on non-Linux.
    """
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            1,
            signal.SIGTERM,
            0,
            0,
            0,  # PR_SET_PDEATHSIG = 1
        )
    except OSError:
        pass


def main() -> None:
    _arm_pdeathsig()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Auto-host a BaseNodeSet in server mode",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Absolute path to the nodeset Python file (single-file mode)",
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Dotted module name (package mode, e.g. workspace.nodesets.server.policy_vla)",
    )
    parser.add_argument("--class", dest="cls", required=True, help="BaseNodeSet class name")
    parser.add_argument("--port", type=int, default=9200, help="Port to serve on (default: 9200)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    if not args.file and not args.module:
        parser.error("Either --file or --module is required")

    nodeset_cls = _import_class(file_path=args.file, module_name=args.module, class_name=args.cls)

    # Move 3 (#54): forward this subprocess's WARNING+ logs to the executor so
    # they surface on the canvas. No-op when AGENTCANVAS_EXECUTOR_URL is unset.
    from . import event_push

    event_push.install_log_bridge(nodeset=getattr(nodeset_cls, "name", args.cls))

    from .auto_server_app import AutoServerApp

    app = AutoServerApp(nodeset_cls)
    app.port = args.port
    app.serve(host=args.host)


if __name__ == "__main__":
    main()
