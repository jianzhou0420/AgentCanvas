"""Keystore semantics — file persistence, permissions, resolution order."""

from __future__ import annotations

import os
import stat

from . import keystore as keystore_mod
from . import providers as providers_mod
from .keystore import KeyStore


def test_set_get_delete_roundtrip(tmp_path):
    store = KeyStore(tmp_path / ".keys")
    assert store.get("OPENAI_API_KEY") == ""
    store.set("OPENAI_API_KEY", "sk-test-123")
    assert store.get("OPENAI_API_KEY") == "sk-test-123"
    assert store.delete("OPENAI_API_KEY") is True
    assert store.get("OPENAI_API_KEY") == ""
    assert store.delete("OPENAI_API_KEY") is False


def test_file_mode_0600(tmp_path):
    store = KeyStore(tmp_path / ".keys")
    store.set("OPENAI_API_KEY", "sk-test")
    mode = stat.S_IMODE(os.stat(store.path).st_mode)
    assert mode == 0o600


def test_dotenv_format_with_comments(tmp_path):
    path = tmp_path / ".keys"
    path.write_text("# comment\nOPENAI_API_KEY=sk-abc\nMALFORMED LINE\nHF_TOKEN='hf_x'\n")
    store = KeyStore(path)
    assert store.get("OPENAI_API_KEY") == "sk-abc"
    assert store.get("HF_TOKEN") == "hf_x"  # quotes stripped


def test_resolution_file_wins_over_env(tmp_path, monkeypatch):
    store = KeyStore(tmp_path / ".keys")
    monkeypatch.setattr(keystore_mod, "_store", store)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    assert providers_mod.get_provider_api_key("openai") == "sk-from-env"
    assert providers_mod.get_provider_key_source("openai") == "env"

    store.set("OPENAI_API_KEY", "sk-from-file")
    assert providers_mod.get_provider_api_key("openai") == "sk-from-file"
    assert providers_mod.get_provider_key_source("openai") == "file"

    store.delete("OPENAI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")
    assert providers_mod.get_provider_api_key("openai") == ""
    assert providers_mod.get_provider_key_source("openai") == "none"


def test_ollama_never_uses_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(keystore_mod, "_store", KeyStore(tmp_path / ".keys"))
    assert providers_mod.get_provider_api_key("ollama") == ""
    assert providers_mod.get_provider_key_source("ollama") == "none"
