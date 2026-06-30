"""Replay renderer host — launch a renderer class from a file path.

Mirrors :mod:`app.server.auto_host`, but does not require a
:class:`BaseNodeSet`. Loaded classes only need a ``build_app()`` method
returning a FastAPI app. Used by env replay parsers' smooth-mode hooks
to spawn an isolated renderer subprocess (e.g. habitat-sim in the
``hmeqa`` conda env) without coupling to the nodeset / graph machinery.

Usage::

    python -m app.replay.renderer_host \\
        --file /path/to/workspace/nodesets/env/env_hmeqa/hmeqa_renderer.py \\
        --class HMEQARendererServer \\
        --port 9300
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys


def _import_class(file_path: str, class_name: str) -> type:
    spec = importlib.util.spec_from_file_location("_replay_renderer", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_replay_renderer"] = mod
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(f"Class {class_name} not found in {file_path}")
    return cls


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Replay renderer subprocess host")
    parser.add_argument("--file", required=True, help="Absolute path to renderer .py file")
    parser.add_argument("--class", dest="cls", required=True, help="Renderer class name")
    parser.add_argument("--port", type=int, default=9300, help="Port to serve on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    args = parser.parse_args()

    renderer_cls = _import_class(args.file, args.cls)
    renderer = renderer_cls()
    fastapi_app = renderer.build_app()

    import uvicorn

    uvicorn.run(fastapi_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
