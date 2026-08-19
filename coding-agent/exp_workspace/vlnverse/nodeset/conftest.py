"""Shared test plumbing for the env_vlnverse package tests.

Registers a synthetic package ``envvlnverse_under_test`` whose ``__path__``
is this directory, so tests can ``importlib.import_module`` the pure-logic
sibling modules (``_quat`` / ``_kinematics`` / ``_metrics``) — with their
relative imports resolving — WITHOUT going through the real package name.

Why not just import the package? Its ``__init__.py`` pulls ``app.components``
(fastapi/pydantic backend), so running the tests requires
``PYTHONPATH=<repo>/agentcanvas/backend:<repo>`` for pytest's own conftest
collection; the synthetic package keeps the modules under test themselves
importable in the lean ac-vlnverse env with numpy only.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
TEST_PKG = "envvlnverse_under_test"

if TEST_PKG not in sys.modules:
    _pkg = types.ModuleType(TEST_PKG)
    _pkg.__path__ = [str(PKG_DIR)]
    sys.modules[TEST_PKG] = _pkg
