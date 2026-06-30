"""Unit tests for matterport3d._resolve_mp3d_data_path.

Covers U1 in .omc/plans/data-layout-unification.md — env-var precedence and
default resolution after the data-layout migration.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure workspace/ is importable and the `app.*` namespace is reachable.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agentcanvas" / "backend"))


def _reload_matterport3d():
    from workspace.nodesets.env import env_mp3d as m

    importlib.reload(m)
    return m


def test_mp3d_data_path_envvar_native():
    env = {"MP3D_DATA_PATH": "/custom/root"}
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("MATTERPORT_DATA_DIR", None)
        m = _reload_matterport3d()
        assert m._resolve_mp3d_data_path() == "/custom/root"


def test_mp3d_data_path_matterport_var_with_scans_suffix():
    with patch.dict(os.environ, {"MATTERPORT_DATA_DIR": "/x/v1/scans"}, clear=False):
        os.environ.pop("MP3D_DATA_PATH", None)
        m = _reload_matterport3d()
        # matterport3d normalises MATTERPORT_DATA_DIR (points AT v1/scans)
        # by returning its parent (the dataset root expected by MP3DEnvManager).
        assert m._resolve_mp3d_data_path() == "/x/v1"


def test_mp3d_data_path_default_after_migration():
    for v in ["MP3D_DATA_PATH", "MATTERPORT_DATA_DIR"]:
        os.environ.pop(v, None)
    m = _reload_matterport3d()
    resolved = m._resolve_mp3d_data_path()
    # Post-migration default: REPO_ROOT/data/mp3d (parent of v1/scans).
    assert resolved.endswith(os.sep + os.path.join("data", "mp3d")), resolved


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
