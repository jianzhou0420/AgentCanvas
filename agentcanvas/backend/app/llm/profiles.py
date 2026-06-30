"""Persistent LLM profile store.

Default location: ``<repo>/agentcanvas/backend/profiles.json`` — checked-in
location alongside ``requirements.txt``. Override via the
``AGENTCANVAS_PROFILES_FILE`` env var (absolute path).

Profiles never carry API keys (those live in per-provider env vars like
``OPENAI_API_KEY``), so this file is safe to commit.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("agentcanvas.profiles")

PROFILES_SCHEMA_VERSION = 1

# Default: alongside requirements.txt at agentcanvas/backend/profiles.json.
# This file is 4 dirs deep:  backend/app/llm/profiles.py
_DEFAULT_PROFILES_FILE = Path(__file__).resolve().parents[2] / "profiles.json"


def _resolve_profiles_file() -> Path:
    override = os.environ.get("AGENTCANVAS_PROFILES_FILE", "").strip()
    return Path(override).expanduser() if override else _DEFAULT_PROFILES_FILE


PROFILES_FILE = _resolve_profiles_file()
PROFILES_DIR = PROFILES_FILE.parent


@dataclass
class LLMProfile:
    """Non-secret per-profile configuration.

    API keys are NEVER stored here — they are read from the provider's
    standard env var at call time (see :func:`app.llm.providers.get_provider_api_key`).
    Legacy profiles.json files containing an ``api_key`` field are
    silently dropped on load and never written back.
    """

    provider: str  # key into PROVIDER_REGISTRY
    model: str
    base_url: str = ""  # "" = use registry default
    api_type: str = ""  # "" = use registry default


@dataclass
class ProfileData:
    active: str = ""
    profiles: dict[str, LLMProfile] = field(default_factory=dict)


class ProfileStore:
    def __init__(self, path: Path = PROFILES_FILE):
        self._path = path
        self._data: ProfileData | None = None
        self._lock = threading.RLock()
        self._file_mtime: float = 0.0
        self._last_mtime_check: float = 0.0
        self._mtime_check_interval: float = 2.0

    def _ensure_dir(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> ProfileData:
        with self._lock:
            now = time.monotonic()
            if self._data is not None:
                # TTL-gated mtime check — stat() at most every 2 seconds
                if now - self._last_mtime_check > self._mtime_check_interval:
                    self._last_mtime_check = now
                    try:
                        current_mtime = self._path.stat().st_mtime
                    except FileNotFoundError:
                        current_mtime = 0.0
                    if current_mtime != self._file_mtime:
                        self._data = None  # invalidate cache
                if self._data is not None:
                    return self._data
            # Read from disk
            if self._path.exists():
                try:
                    raw = json.loads(self._path.read_text())
                    _version = raw.get("schema_version", 1)
                    profiles = {}
                    for name, p in raw.get("profiles", {}).items():
                        # Legacy ``api_key`` field is silently dropped — see
                        # the LLMProfile docstring for migration semantics.
                        profiles[name] = LLMProfile(
                            provider=p.get("provider", "custom"),
                            model=p.get("model", ""),
                            base_url=p.get("base_url", ""),
                            api_type=p.get("api_type", ""),
                        )
                    self._data = ProfileData(
                        active=raw.get("active", ""),
                        profiles=profiles,
                    )
                except Exception:
                    log.exception("Failed to load profiles, starting fresh")
                    self._data = ProfileData()
            else:
                self._data = ProfileData()
            self._file_mtime = self._path.stat().st_mtime if self._path.exists() else 0.0
            self._last_mtime_check = now
            return self._data

    def save(self):
        with self._lock:
            data = self._data if self._data is not None else ProfileData()
            self._ensure_dir()
            out = {
                "schema_version": PROFILES_SCHEMA_VERSION,
                "active": data.active,
                "profiles": {n: asdict(p) for n, p in data.profiles.items()},
            }
            self._path.write_text(json.dumps(out, indent=2))
            # No secrets in this file — keep default umask permissions so it
            # plays nicely with shared checkouts and git.
            self._file_mtime = self._path.stat().st_mtime

    def list_profiles(self) -> dict[str, LLMProfile]:
        return dict(self.load().profiles)

    def get(self, name: str) -> LLMProfile | None:
        return self.load().profiles.get(name)

    def create(self, name: str, profile: LLMProfile) -> None:
        with self._lock:
            data = self.load()
            if name in data.profiles:
                raise ValueError(f"Profile '{name}' already exists")
            data.profiles[name] = profile
            self.save()

    def update(self, name: str, **fields) -> None:
        with self._lock:
            data = self.load()
            p = data.profiles.get(name)
            if p is None:
                raise KeyError(f"Profile '{name}' not found")
            for k, v in fields.items():
                if hasattr(p, k):
                    setattr(p, k, v)
            self.save()

    def delete(self, name: str) -> None:
        with self._lock:
            data = self.load()
            if name not in data.profiles:
                raise KeyError(f"Profile '{name}' not found")
            del data.profiles[name]
            if data.active == name:
                data.active = ""
            self.save()

    def get_active(self) -> str:
        return self.load().active

    def set_active(self, name: str) -> None:
        with self._lock:
            data = self.load()
            if name and name not in data.profiles:
                raise KeyError(f"Profile '{name}' not found")
            data.active = name
            self.save()


_store: ProfileStore | None = None


def get_profile_store() -> ProfileStore:
    global _store
    if _store is None:
        _store = ProfileStore()
    return _store
