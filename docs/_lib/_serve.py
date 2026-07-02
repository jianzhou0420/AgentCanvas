#!/usr/bin/env python3
"""Lightweight live-reload dev server for docs/.

Pure stdlib — no pip / no npm install. Watches the file tree with mtime
polling (~500ms); when anything changes, pushes a reload event to all
connected browser tabs via Server-Sent Events. HTML responses get a tiny
script injected before </body> that subscribes to the event stream.

Usage:
  python3 docs/_serve.py                 # serves on 0.0.0.0:8092
  python3 docs/_serve.py --port 9000

Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue

# docs/ root — script lives at docs/_lib/_serve.py.
V2_DIR = Path(__file__).resolve().parent.parent

# Injected before </body> of every served HTML page.
RELOAD_SNIPPET = b"""
<script>
(function () {
  var es;
  function connect() {
    es = new EventSource('/__reload-stream');
    es.onmessage = function (e) {
      if (e.data === 'reload') { location.reload(); }
    };
    es.onerror = function () {
      es.close();
      // Server gone; back off + try again. If it comes back, refresh once.
      setTimeout(function () {
        fetch('/__reload-stream', { method: 'HEAD' })
          .then(function () { location.reload(); })
          .catch(connect);
      }, 1000);
    };
  }
  // CRITICAL: close on navigation so old SSE doesn't hold a browser connection
  // slot during nav. Browsers limit ~6 connections per origin; without this,
  // rapid nav exhausts the pool and new requests stall.
  function shutdown() { try { es && es.close(); } catch (_) {} }
  window.addEventListener('beforeunload', shutdown);
  window.addEventListener('pagehide', shutdown);
  connect();
})();
</script>
"""

EXCLUDED_DIR_NAMES = {".git", "__pycache__", "node_modules", ".idea", ".vscode"}
EXCLUDED_SUFFIXES = (".swp", ".swo", ".tmp", ".pyc")

# Committed assets with a gitignored ".local" twin holding the full view —
# including gitignored sections (see _wrap_handwritten). The dev server serves
# the twin so local-only pages show up locally; the published site has no
# .local files and serves the committed public view.
LOCAL_ASSET_VARIANTS = {
    "/assets/nav.json": "/assets/nav.local.json",
    "/assets/search-index.json": "/assets/search-index.local.json",
}


def is_excluded(p: Path) -> bool:
    if p.suffix in EXCLUDED_SUFFIXES:
        return True
    if p.name.startswith("."):
        return True
    return any(part in EXCLUDED_DIR_NAMES for part in p.parts)


def snapshot(root: Path) -> tuple:
    """Hashable signature of the tree's mtime+size state."""
    sig = []
    try:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if is_excluded(p.relative_to(root)):
                continue
            try:
                st = p.stat()
                sig.append((str(p.relative_to(root)), st.st_mtime_ns, st.st_size))
            except (FileNotFoundError, OSError):
                continue
    except (FileNotFoundError, OSError):
        pass
    return tuple(sig)


class Watcher(threading.Thread):
    def __init__(self, root: Path, interval: float, on_change, *, label: str = "watch"):
        super().__init__(daemon=True)
        self.root = root
        self.interval = interval
        self.on_change = on_change
        self.label = label
        self.last = snapshot(root)
        self._running = True

    def run(self):
        while self._running:
            time.sleep(self.interval)
            snap = snapshot(self.root)
            if snap != self.last:
                changed = self._diff(self.last, snap)
                self.last = snap
                self.on_change(changed)

    def _diff(self, old: tuple, new: tuple) -> list[str]:
        old_set = {p for p, _, _ in old}
        new_set = {p for p, _, _ in new}
        old_map = {p: (m, s) for p, m, s in old}
        new_map = {p: (m, s) for p, m, s in new}
        changed = []
        for p in new_set - old_set:
            changed.append(p)
        for p in old_set - new_set:
            changed.append(p)
        for p in new_set & old_set:
            if old_map[p] != new_map[p]:
                changed.append(p)
        return changed


# ---------- SSE coordination ----------

_subscribers = []
_subscribers_lock = threading.Lock()


def broadcast_reload(changed_paths: list[str]) -> None:
    print(
        f"  ↻ reload  ({len(changed_paths)} changed: {', '.join(changed_paths[:3])}"
        f"{' ...' if len(changed_paths) > 3 else ''})"
    )
    with _subscribers_lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait("reload")
            except Exception:
                dead.append(q)
        for q in dead:
            with contextlib.suppress(ValueError):
                _subscribers.remove(q)


def subscribe() -> Queue:
    q = Queue()
    with _subscribers_lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: Queue) -> None:
    with _subscribers_lock, contextlib.suppress(ValueError):
        _subscribers.remove(q)


# ---------- HTTP handler ----------


class LiveReloadHandler(SimpleHTTPRequestHandler):
    # Suppress chatty 200-OK logging; keep 4xx/5xx + reload markers.
    def log_message(self, fmt, *args):
        if args and isinstance(args[1], str) and args[1].startswith("2"):
            return
        super().log_message(fmt, *args)

    def end_headers(self):
        # Dev-server: disable browser caching so CSS/JS edits show on next nav.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if self.path == "/__reload-stream":
            self._serve_sse()
            return
        local = LOCAL_ASSET_VARIANTS.get(self.path)
        if local and os.path.exists(self.translate_path(local)):
            self.path = local
        if (self.path.endswith(".html") or self.path.endswith("/")) and self._serve_html():
            return
        super().do_GET()

    def do_HEAD(self):
        if self.path == "/__reload-stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            return
        super().do_HEAD()

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = subscribe()
        try:
            self.wfile.write(b"event: hello\ndata: ok\n\n")
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=2)
                    self.wfile.write(f"data: {msg}\n\n".encode())
                    self.wfile.flush()
                except Empty:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            unsubscribe(q)

    def _serve_html(self) -> bool:
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            for index in ("index.html", "index.htm"):
                cand = os.path.join(path, index)
                if os.path.exists(cand):
                    path = cand
                    break
            else:
                return False  # let parent handle dir listing
        if not os.path.exists(path) or not os.path.isfile(path):
            return False
        try:
            with open(path, "rb") as f:
                content = f.read()
        except OSError:
            return False
        idx = content.lower().rfind(b"</body>")
        if idx >= 0:
            content = content[:idx] + RELOAD_SNIPPET + content[idx:]
        else:
            content = content + RELOAD_SNIPPET
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)
        return True


# ---------- driver ----------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--port", type=int, default=8092, help="HTTP port (default: 8092)")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p.add_argument(
        "--interval", type=float, default=0.5, help="poll interval in seconds (default: 0.5)"
    )
    args = p.parse_args()

    os.chdir(V2_DIR)
    autowrap = os.environ.get("NO_AUTOWRAP", "") == ""
    print("docs/ live-reload server")
    print(f"  root      : {V2_DIR}")
    print(f"  url       : http://{args.host}:{args.port}/")
    print(f"  interval  : {args.interval}s")
    print(f"  auto-wrap : {'on' if autowrap else 'off (NO_AUTOWRAP set)'}")

    # Re-bake the chrome on every change before reloading the browser — so
    # dropping a folder of pages "just works" with no manual wrap step.
    if autowrap:
        import _wrap_handwritten as _wrap

    def on_change(changed):
        if autowrap:
            try:
                n = _wrap.main(quiet=True)
                if n:
                    print(f"  ⤷ wrapped {n} page(s)")
            except Exception as e:
                print(f"  (auto-wrap error: {type(e).__name__}: {e})")
            # Absorb the wrap's own writes so they aren't seen as a new change.
            html_watcher.last = snapshot(V2_DIR)
        broadcast_reload(changed)

    html_watcher = Watcher(V2_DIR, args.interval, on_change, label="html")

    # Bind + serve FIRST so the port is reachable instantly. The initial chrome
    # bake can take a few seconds on a large site (Python 3.8, hundreds of pages
    # + the search index), so it runs in a background thread instead of blocking
    # the bind — otherwise the port refuses connections until the bake finishes.
    # Pages are already baked on disk, so requests served before it completes are
    # correct; when the bake finishes it nudges any open browser to reload.
    server = ThreadingHTTPServer((args.host, args.port), LiveReloadHandler)

    def _initial_bake():
        if not autowrap:
            return
        try:
            _wrap.main(quiet=True)
        except Exception as e:
            print(f"  (initial wrap skipped: {type(e).__name__}: {e})")
            return
        # Absorb the bake's own writes, then reload any browser onto the fresh build.
        html_watcher.last = snapshot(V2_DIR)
        broadcast_reload(["(initial bake)"])

    threading.Thread(target=_initial_bake, daemon=True).start()
    html_watcher.start()

    print("  Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye.")
        server.server_close()


if __name__ == "__main__":
    main()
