"""Phase 2 (state normalization) unit tests: the #68 ownership guardrail and
the executor-URL resolver.

    python -m pytest app/test_state_normalization.py -v
"""

from __future__ import annotations

import pytest

from app.components.registry import WorkspaceComponentRegistry


def _bare_registry() -> WorkspaceComponentRegistry:
    # Skip __init__ (heavy) — the guardrail only needs _get_parallelism + log.
    return WorkspaceComponentRegistry.__new__(WorkspaceComponentRegistry)


def test_guardrail_shared_stateful_warns(caplog: pytest.LogCaptureFixture) -> None:
    reg = _bare_registry()
    reg._get_parallelism = lambda n: "shared"  # type: ignore[method-assign]
    with caplog.at_level("WARNING"):
        status = reg._check_container_ownership("state_demo", ["state_demo"], worker_count=4)
    assert status == "shared-stateful"
    assert "#68" in caplog.text
    assert "state_demo" in caplog.text


def test_guardrail_replicated_fanout_unroutable(caplog: pytest.LogCaptureFixture) -> None:
    reg = _bare_registry()
    reg._get_parallelism = lambda n: "replicated"  # type: ignore[method-assign]
    with caplog.at_level("WARNING"):
        status = reg._check_container_ownership("explore_eqa_tsdf", ["tsdf"], worker_count=4)
    assert status == "replicated-fanout-unroutable"
    assert "#17" in caplog.text


def test_guardrail_replicated_single_worker_ok() -> None:
    reg = _bare_registry()
    reg._get_parallelism = lambda n: "replicated"  # type: ignore[method-assign]
    assert reg._check_container_ownership("explore_eqa_tsdf", ["tsdf"], worker_count=1) is None


def test_guardrail_no_containers_ok() -> None:
    reg = _bare_registry()
    reg._get_parallelism = lambda n: "shared"  # type: ignore[method-assign]
    assert reg._check_container_ownership("vlm_prismatic", [], worker_count=4) is None


def test_resolve_executor_url_default_and_explicit() -> None:
    from app import config

    config._settings_instance = None
    s = config.get_settings()
    assert config.resolve_executor_url() == f"http://127.0.0.1:{s.port}"
    s.executor_url = "http://example:9999"
    try:
        assert config.resolve_executor_url() == "http://example:9999"
    finally:
        s.executor_url = None


def test_state_demo_is_replicated() -> None:
    # The footgun fixture must now declare replicated so the #68 warn path is
    # clean (it owns a mutable container).
    import importlib.util
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "workspace"
        / "nodesets"
        / "common"
        / "state_demo.py"
    )
    spec = importlib.util.spec_from_file_location("state_demo_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.StateDemoNodeSet.parallelism == "replicated"
