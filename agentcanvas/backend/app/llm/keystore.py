"""Persistent API-key store — ``~/.agentcanvas/.keys``.

The one place keys are persisted, deliberately *outside* the repository
so the release mirror can never leak a secret. Format is dotenv-style,
keyed by each provider's standard env-var name::

    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...

Resolution order (see :func:`app.llm.providers.get_provider_api_key`):
file entry first, process env var as fallback — so a key saved from the
Settings UI takes effect immediately, while pure-env setups keep working
untouched. The file is written with ``0600`` permissions.

Override the location with ``AGENTCANVAS_KEYS_FILE`` (absolute path;
used by tests).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("agentcanvas.keystore")

_DEFAULT_KEYS_FILE = Path.home() / ".agentcanvas" / ".keys"


def _resolve_keys_file() -> Path:
    override = os.environ.get("AGENTCANVAS_KEYS_FILE", "").strip()
    return Path(override).expanduser() if override else _DEFAULT_KEYS_FILE


class KeyStore:
    """mtime-cached dotenv file of ``ENV_VAR=key`` lines."""

    def __init__(self, path: Path | None = None):
        self._path = path or _resolve_keys_file()
        self._entries: dict[str, str] | None = None
        self._lock = threading.RLock()
        self._file_mtime: float = 0.0
        self._last_mtime_check: float = 0.0
        self._mtime_check_interval: float = 2.0

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> dict[str, str]:
        with self._lock:
            now = time.monotonic()
            if self._entries is not None:
                # TTL-gated mtime check — stat() at most every 2 seconds
                if now - self._last_mtime_check > self._mtime_check_interval:
                    self._last_mtime_check = now
                    try:
                        current_mtime = self._path.stat().st_mtime
                    except FileNotFoundError:
                        current_mtime = 0.0
                    if current_mtime != self._file_mtime:
                        self._entries = None
                if self._entries is not None:
                    return self._entries
            entries: dict[str, str] = {}
            if self._path.exists():
                try:
                    for line in self._path.read_text().splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        name, _, value = line.partition("=")
                        entries[name.strip()] = value.strip().strip("'\"")
                except Exception:
                    log.exception("Failed to read %s — treating as empty", self._path)
            self._entries = entries
            self._file_mtime = self._path.stat().st_mtime if self._path.exists() else 0.0
            self._last_mtime_check = now
            return entries

    def _write(self, entries: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{name}={value}" for name, value in sorted(entries.items())]
        self._path.write_text("\n".join(lines) + ("\n" if lines else ""))
        os.chmod(self._path, 0o600)
        self._entries = dict(entries)
        self._file_mtime = self._path.stat().st_mtime

    def get(self, env_var: str) -> str:
        if not env_var:
            return ""
        return self._load().get(env_var, "")

    def set(self, env_var: str, key: str) -> None:
        if not env_var:
            raise ValueError("env_var must be non-empty")
        with self._lock:
            entries = dict(self._load())
            entries[env_var] = key
            self._write(entries)

    def delete(self, env_var: str) -> bool:
        with self._lock:
            entries = dict(self._load())
            if env_var not in entries:
                return False
            del entries[env_var]
            self._write(entries)
            return True


_store: KeyStore | None = None


def get_key_store() -> KeyStore:
    global _store
    if _store is None:
        _store = KeyStore()
    return _store
