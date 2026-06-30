"""JobScheduler unit + integration tests.

Unit tests stub out subprocess.Popen so admission/cancel logic is
exercised without spawning real Python.

The integration test (``test_real_popen_smoke``) spawns a trivial
shell command instead of ``app.eval_subprocess_main`` to verify the
full Popen/reap/_DONE plumbing without needing a real graph.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

from .job_scheduler import JobScheduler, _RunningJob
from .run_state_io import atomic_write_json, is_done, read_summary, touch_done


def _spec(
    run_id: str | None = None, marginal_vram_mb: int = 0, exclusive_gpu: bool = False
) -> dict:
    return {
        "run_id": run_id,
        "eval": {"graph_name": "g", "episode_count": 1, "worker_count": 1},
        "scheduling": {
            "marginal_vram_mb": marginal_vram_mb,
            "exclusive_gpu": exclusive_gpu,
            "priority": "normal",
        },
        "graph": {"nodes": [], "edges": []},
    }


def test_submit_writes_files(tmp_path: Path) -> None:
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    run_id = sched.submit(_spec(marginal_vram_mb=1000))
    run_dir = tmp_path / run_id
    assert (run_dir / "spec.json").exists()
    assert (run_dir / "shared_urls.json").exists()
    assert (run_dir / "summary.json").exists()
    assert read_summary(run_dir)["status"] == "pending"
    assert sched.list_active()["queued"][0]["run_id"] == run_id


def test_fresh_run_id_is_timestamp(tmp_path: Path) -> None:
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    rid = sched._fresh_run_id()
    # YYYYMMDD_HHMMSS — 15 chars, all digits except the separator.
    assert len(rid) == 15 and rid[8] == "_"
    assert rid.replace("_", "").isdigit()


def test_fresh_run_id_collision_suffix(tmp_path: Path) -> None:
    """Two submissions in the same wall-clock second get _2, _3, … ."""
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    with patch("time.strftime", return_value="20260515_120000"):
        rid1 = sched._fresh_run_id()
        assert rid1 == "20260515_120000"
        (tmp_path / rid1).mkdir()
        rid2 = sched._fresh_run_id()
        assert rid2 == "20260515_120000_2"
        (tmp_path / rid2).mkdir()
        rid3 = sched._fresh_run_id()
        assert rid3 == "20260515_120000_3"


def test_admit_respects_canvas_lock(tmp_path: Path) -> None:
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    sched.set_canvas_lock_callback(lambda: True)  # canvas Play active
    sched.submit(_spec(marginal_vram_mb=0))
    with patch.object(sched, "_spawn") as spawn:
        asyncio.run(sched._admit())
        spawn.assert_not_called()
    assert len(sched._queue) == 1


def test_admit_respects_vram_budget(tmp_path: Path) -> None:
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=10000)
    sched.submit(_spec(run_id="big", marginal_vram_mb=15000))  # too big
    sched.submit(_spec(run_id="ok", marginal_vram_mb=5000))
    spawned: list[str] = []

    def _fake_spawn(q, ephem_tag=None):
        spawned.append(q.run_id)
        sched._running[q.run_id] = _RunningJob(
            run_id=q.run_id,
            proc=type("P", (), {"pid": 1, "poll": lambda self=None: None})(),
            pgid=1,
            marginal_vram_mb=q.marginal_vram_mb,
            exclusive_gpu=q.exclusive_gpu,
            started_at=time.time(),
        )

    with patch.object(sched, "_spawn", side_effect=_fake_spawn):
        asyncio.run(sched._admit())
    assert spawned == ["ok"]  # big stays queued
    assert [q.run_id for q in sched._queue] == ["big"]


def test_exclusive_gpu_blocks_others(tmp_path: Path) -> None:
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=24000)
    sched.submit(_spec(run_id="exc", marginal_vram_mb=1000, exclusive_gpu=True))
    sched.submit(_spec(run_id="small", marginal_vram_mb=500))
    with patch.object(sched, "_spawn") as spawn:

        def fake(q, ephem_tag=None):
            sched._running[q.run_id] = _RunningJob(
                run_id=q.run_id,
                proc=type("P", (), {"pid": 1, "poll": lambda self=None: None})(),
                pgid=1,
                marginal_vram_mb=q.marginal_vram_mb,
                exclusive_gpu=q.exclusive_gpu,
                started_at=time.time(),
            )

        spawn.side_effect = fake
        asyncio.run(sched._admit())
    assert "exc" in sched._running
    assert "small" not in sched._running  # blocked by exclusive
    assert [q.run_id for q in sched._queue] == ["small"]


def test_cancel_queued(tmp_path: Path) -> None:
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    run_id = sched.submit(_spec(marginal_vram_mb=1000))
    assert sched.cancel(run_id) == "cancelled"
    assert sched._queue == []
    summary = read_summary(tmp_path / run_id)
    assert summary["status"] == "cancelled"
    assert is_done(tmp_path / run_id)


def test_cancel_unknown(tmp_path: Path) -> None:
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    assert sched.cancel("does-not-exist") == "unknown"


def test_real_popen_smoke(tmp_path: Path) -> None:
    """End-to-end through admit + Popen + reap, but with a fake
    eval_runner that just writes _DONE + summary.

    Patches _spawn to use a shell command instead of
    `python -m app.eval_subprocess_main` so the test doesn't need the
    full backend to be importable in the subprocess.
    """
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    run_id = sched.submit(_spec(marginal_vram_mb=0))
    run_dir = tmp_path / run_id

    def _fake_spawn(q, ephem_tag=None):
        # Write a minimal succeeded summary + _DONE; mimics what
        # eval_subprocess_main does on completion.
        bash_script = (
            f"cat > {run_dir}/summary.json <<EOF\n"
            '{"run_id": "' + q.run_id + '", "status": "completed"}\n'
            "EOF\n"
            f"echo done > {run_dir}/_DONE"
        )
        proc = subprocess.Popen(
            ["bash", "-c", bash_script],
            start_new_session=True,
        )
        sched._running[q.run_id] = _RunningJob(
            run_id=q.run_id,
            proc=proc,
            pgid=os.getpgid(proc.pid),
            marginal_vram_mb=q.marginal_vram_mb,
            exclusive_gpu=q.exclusive_gpu,
            started_at=time.time(),
        )

    with patch.object(sched, "_spawn", side_effect=_fake_spawn):
        asyncio.run(sched._admit())
    assert run_id in sched._running

    # Wait for the bash command to finish, then reap.
    deadline = time.time() + 5
    while time.time() < deadline:
        if sched._running[run_id].proc.poll() is not None:
            break
        time.sleep(0.05)
    asyncio.run(sched._reap())
    assert run_id not in sched._running
    assert is_done(run_dir)
    assert read_summary(run_dir)["status"] == "completed"


def test_reconcile_aborted_runs(tmp_path: Path) -> None:
    """Backend-restart cleanup: status='running' rows without _DONE flip
    to 'aborted'.
    """
    from .job_scheduler import reconcile_aborted_runs
    from .run_state_io import initial_running_summary

    # Run A: was running, no _DONE → should be aborted.
    a = tmp_path / "a"
    a.mkdir()
    atomic_write_json(a / "summary.json", initial_running_summary("a", {}, "t"))

    # Run B: completed cleanly with _DONE → must NOT be touched.
    b = tmp_path / "b"
    b.mkdir()
    bsum = initial_running_summary("b", {}, "t")
    bsum["status"] = "completed"
    atomic_write_json(b / "summary.json", bsum)
    touch_done(b)

    fixed = reconcile_aborted_runs(tmp_path)
    assert fixed == 1
    assert read_summary(a)["status"] == "aborted"
    assert is_done(a)  # _DONE written so it doesn't re-trigger
    assert read_summary(b)["status"] == "completed"  # untouched


# ── TODO #60: ephemeral spawn at admit time ──


class _FakeRegistry:
    """Stand-in for WorkspaceComponentRegistry, capturing ephemeral spawn calls."""

    def __init__(self) -> None:
        self._discovered_nodesets: dict = {}
        # The frozen workspace/ dir — the real WorkspaceComponentRegistry sets this
        # in __init__ (Path(scan_dir)). _prepare_ephemerals resolves overlay
        # paths relative to it. Tests that exercise ephemeral spawn set it.
        self._frozen_dir = None
        self.spawn_calls: list[tuple[str, str, str]] = []  # (name, source, tag)
        self.unload_calls: list[str] = []
        # Auto-unload paths exercise unload_nodeset (the regular, non-ephemeral
        # entry point); record those separately so refcount tests can assert.
        self.unload_nodeset_calls: list[str] = []
        self.next_port = 50001

    async def load_nodeset_ephemeral(self, name: str, source_path, tag: str) -> str:
        self.spawn_calls.append((name, str(source_path), tag))
        url = f"http://localhost:{self.next_port}"
        self.next_port += 1
        return url

    def unload_nodeset_ephemeral(self, tag: str) -> int:
        self.unload_calls.append(tag)
        return 1

    async def unload_nodeset(self, name: str) -> dict:
        self.unload_nodeset_calls.append(name)
        return {"name": name, "tools_removed": []}


def _fake_ns(source_file: str):
    """Returns a stub object with the attrs registry._discovered_nodesets values
    are duck-typed on."""

    class _NS:
        _source_file = source_file

    return _NS()


def test_admit_spawns_ephemeral_on_overlay_diff(tmp_path: Path) -> None:
    from .run_state_io import read_shared_urls

    # Production layout: the frozen nodeset lives under <repo>/workspace/,
    # and an iter's active_workspace/ mirrors the *contents* of workspace/
    # (rooted at active_workspace/{graphs,nodesets}/ — there is NO
    # "workspace/" path segment). _prepare_ephemerals must resolve the
    # overlay relative to the frozen workspace/ dir (registry._frozen_dir),
    # not the repo root — passing the repo root leaves a stray "workspace/"
    # in the relative path so the overlay candidate never exists.
    workspace_dir = tmp_path / "repo" / "workspace"
    (workspace_dir / "nodesets" / "server").mkdir(parents=True)
    frozen_file = workspace_dir / "nodesets" / "server" / "vlm.py"
    frozen_file.write_text("class VLM: pass  # frozen\n")

    overlay_dir = tmp_path / "aw"
    (overlay_dir / "nodesets" / "server").mkdir(parents=True)
    overlay_file = overlay_dir / "nodesets" / "server" / "vlm.py"
    overlay_file.write_text("class VLM: pass  # overlay\n")

    sched = JobScheduler(eval_runs_dir=tmp_path / "runs", usable_vram_mb=20000)

    reg = _FakeRegistry()
    reg._frozen_dir = workspace_dir
    reg._discovered_nodesets["vlm"] = _fake_ns(str(frozen_file))
    sched.set_workspace_component_registry(reg)

    spec = _spec(marginal_vram_mb=0)
    spec["active_workspace_dir"] = str(overlay_dir)
    spec["_shared_urls"] = {"vlm": "http://localhost:10000"}
    run_id = sched.submit(spec)

    # Sanity: shared_urls.json was written with frozen URL.
    pre = read_shared_urls(tmp_path / "runs" / run_id)
    assert pre == {"vlm": "http://localhost:10000"}

    with patch.object(sched, "_spawn") as spawn:

        def fake(q, ephem_tag=None):
            sched._running[q.run_id] = _RunningJob(
                run_id=q.run_id,
                proc=type("P", (), {"pid": 1, "poll": lambda self=None: None})(),
                pgid=1,
                marginal_vram_mb=q.marginal_vram_mb,
                exclusive_gpu=q.exclusive_gpu,
                started_at=time.time(),
                ephem_tag=ephem_tag,
            )

        spawn.side_effect = fake
        asyncio.run(sched._admit())

    # Ephemeral was spawned for the redefined nodeset.
    assert len(reg.spawn_calls) == 1
    name, source, tag = reg.spawn_calls[0]
    assert name == "vlm"
    assert source == str(overlay_file.resolve())
    assert tag == f"ephem-{run_id}"

    # shared_urls.json was rewritten on disk with the ephemeral URL.
    post = read_shared_urls(tmp_path / "runs" / run_id)
    assert post["vlm"].startswith("http://localhost:5000")  # ephem URL pattern

    # _RunningJob carries the tag for _reap teardown.
    assert sched._running[run_id].ephem_tag == tag


def test_admit_skips_ephemeral_on_byte_identical_overlay(tmp_path: Path) -> None:
    from .run_state_io import read_shared_urls

    workspace_dir = tmp_path / "repo" / "workspace"
    (workspace_dir / "nodesets" / "server").mkdir(parents=True)
    frozen_file = workspace_dir / "nodesets" / "server" / "vlm.py"
    frozen_file.write_text("class VLM: pass  # same\n")

    overlay_dir = tmp_path / "aw"
    (overlay_dir / "nodesets" / "server").mkdir(parents=True)
    overlay_file = overlay_dir / "nodesets" / "server" / "vlm.py"
    overlay_file.write_text("class VLM: pass  # same\n")  # byte-identical

    sched = JobScheduler(eval_runs_dir=tmp_path / "runs", usable_vram_mb=20000)
    reg = _FakeRegistry()
    reg._frozen_dir = workspace_dir
    reg._discovered_nodesets["vlm"] = _fake_ns(str(frozen_file))
    sched.set_workspace_component_registry(reg)

    spec = _spec(marginal_vram_mb=0)
    spec["active_workspace_dir"] = str(overlay_dir)
    spec["_shared_urls"] = {"vlm": "http://localhost:10000"}
    run_id = sched.submit(spec)

    with patch.object(sched, "_spawn") as spawn:
        spawn.side_effect = lambda q, ephem_tag=None: sched._running.setdefault(
            q.run_id,
            _RunningJob(
                run_id=q.run_id,
                proc=type("P", (), {"pid": 1, "poll": lambda self=None: None})(),
                pgid=1,
                marginal_vram_mb=q.marginal_vram_mb,
                exclusive_gpu=q.exclusive_gpu,
                started_at=time.time(),
                ephem_tag=ephem_tag,
            ),
        )
        asyncio.run(sched._admit())

    # No ephemeral spawned; frozen URL preserved.
    assert reg.spawn_calls == []
    post = read_shared_urls(tmp_path / "runs" / run_id)
    assert post == {"vlm": "http://localhost:10000"}
    assert sched._running[run_id].ephem_tag is None


def test_reap_releases_ephemerals(tmp_path: Path) -> None:
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    reg = _FakeRegistry()
    sched.set_workspace_component_registry(reg)

    # Submit + manually wire a _RunningJob with an ephem_tag, then simulate
    # subprocess exit + _DONE so _reap finalizes it.
    run_id = sched.submit(_spec(marginal_vram_mb=0))
    run_dir = tmp_path / run_id
    summary = read_summary(run_dir) or {}
    summary["status"] = "completed"
    atomic_write_json(run_dir / "summary.json", summary)
    touch_done(run_dir)

    sched._running[run_id] = _RunningJob(
        run_id=run_id,
        proc=type("P", (), {"pid": 1, "poll": lambda self=None: 0})(),
        pgid=1,
        marginal_vram_mb=0,
        exclusive_gpu=False,
        started_at=time.time(),
        ephem_tag=f"ephem-{run_id}",
    )
    asyncio.run(sched._reap())
    assert reg.unload_calls == [f"ephem-{run_id}"]
    assert run_id not in sched._running


def test_reap_skips_release_when_no_ephem_tag(tmp_path: Path) -> None:
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    reg = _FakeRegistry()
    sched.set_workspace_component_registry(reg)

    run_id = sched.submit(_spec(marginal_vram_mb=0))
    run_dir = tmp_path / run_id
    summary = read_summary(run_dir) or {}
    summary["status"] = "completed"
    atomic_write_json(run_dir / "summary.json", summary)
    touch_done(run_dir)
    sched._running[run_id] = _RunningJob(
        run_id=run_id,
        proc=type("P", (), {"pid": 1, "poll": lambda self=None: 0})(),
        pgid=1,
        marginal_vram_mb=0,
        exclusive_gpu=False,
        started_at=time.time(),
        ephem_tag=None,  # no ephemerals
    )
    asyncio.run(sched._reap())
    assert reg.unload_calls == []


def test_admit_no_overlay_skips_ephemeral(tmp_path: Path) -> None:
    sched = JobScheduler(eval_runs_dir=tmp_path / "runs", usable_vram_mb=20000)
    reg = _FakeRegistry()
    sched.set_workspace_component_registry(reg)

    spec = _spec(marginal_vram_mb=0)
    # active_workspace_dir omitted entirely.
    spec["_shared_urls"] = {"vlm": "http://localhost:10000"}
    sched.submit(spec)

    with patch.object(sched, "_spawn") as spawn:
        spawn.side_effect = lambda q, ephem_tag=None: sched._running.setdefault(
            q.run_id,
            _RunningJob(
                run_id=q.run_id,
                proc=type("P", (), {"pid": 1, "poll": lambda self=None: None})(),
                pgid=1,
                marginal_vram_mb=q.marginal_vram_mb,
                exclusive_gpu=q.exclusive_gpu,
                started_at=time.time(),
                ephem_tag=ephem_tag,
            ),
        )
        asyncio.run(sched._admit())

    assert reg.spawn_calls == []
    assert reg.unload_calls == []


# ── Shared-singleton auto-unload (refcounted) ──


def _stub_running(run_id: str, *, exited: bool, shared_nodesets: tuple[str, ...]) -> _RunningJob:
    rc = 0 if exited else None
    return _RunningJob(
        run_id=run_id,
        proc=type("P", (), {"pid": 1, "poll": lambda self=None, _rc=rc: _rc})(),
        pgid=1,
        marginal_vram_mb=0,
        exclusive_gpu=False,
        started_at=time.time(),
        shared_nodesets=shared_nodesets,
    )


def _mark_done(run_dir: Path, status: str = "completed") -> None:
    summary = read_summary(run_dir) or {}
    summary["status"] = status
    atomic_write_json(run_dir / "summary.json", summary)
    touch_done(run_dir)


def test_auto_unload_shared_singleton_on_last_job_reap(tmp_path: Path) -> None:
    """One job loaded VLM; on reap the scheduler unloads it."""
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    reg = _FakeRegistry()
    sched.set_workspace_component_registry(reg)

    spec = _spec(marginal_vram_mb=0)
    spec["_shared_urls"] = {"vlm": "http://localhost:9001"}
    spec["_shared_loaded_by_us"] = ["vlm"]
    run_id = sched.submit(spec)

    # Simulate admission: move into _running with shared_nodesets set.
    q = sched._queue.pop(0)
    sched._running[run_id] = _stub_running(run_id, exited=False, shared_nodesets=q.shared_nodesets)

    # Job finishes.
    _mark_done(tmp_path / run_id)
    sched._running[run_id].proc.poll = lambda self=None: 0  # type: ignore[assignment]
    asyncio.run(sched.tick())

    assert reg.unload_nodeset_calls == ["vlm"]
    assert "vlm" not in sched._shared_loaded_by_jobs


def test_auto_unload_holds_when_second_job_still_uses_it(tmp_path: Path) -> None:
    """Job A loaded VLM, Job B attaches without reloading. A reaps → VLM
    stays. B reaps → VLM unloads.
    """
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    reg = _FakeRegistry()
    sched.set_workspace_component_registry(reg)

    # Job A: freshly loaded VLM.
    specA = _spec(run_id="A", marginal_vram_mb=0)
    specA["_shared_urls"] = {"vlm": "http://localhost:9001"}
    specA["_shared_loaded_by_us"] = ["vlm"]
    sched.submit(specA)

    # Job B: VLM was already loaded by A; B only consumes.
    specB = _spec(run_id="B", marginal_vram_mb=0)
    specB["_shared_urls"] = {"vlm": "http://localhost:9001"}
    specB["_shared_loaded_by_us"] = []  # not freshly loaded
    sched.submit(specB)

    # Both admitted into _running.
    for q in list(sched._queue):
        sched._running[q.run_id] = _stub_running(
            q.run_id, exited=False, shared_nodesets=q.shared_nodesets
        )
    sched._queue.clear()

    # Job A finishes.
    _mark_done(tmp_path / "A")
    sched._running["A"].proc.poll = lambda self=None: 0  # type: ignore[assignment]
    asyncio.run(sched.tick())

    # B still references VLM → must not unload yet.
    assert reg.unload_nodeset_calls == []
    assert sched._shared_consumer_count.get("vlm") == 1
    assert "vlm" in sched._shared_loaded_by_jobs

    # Job B finishes.
    _mark_done(tmp_path / "B")
    sched._running["B"].proc.poll = lambda self=None: 0  # type: ignore[assignment]
    asyncio.run(sched.tick())

    # Now no consumers — VLM should be unloaded.
    assert reg.unload_nodeset_calls == ["vlm"]
    assert "vlm" not in sched._shared_loaded_by_jobs


def test_canvas_loaded_shared_stays_after_job_reap(tmp_path: Path) -> None:
    """A shared singleton NOT in _shared_loaded_by_us was loaded by canvas
    Play, not by this job, so it must NOT be auto-unloaded on reap.
    """
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    reg = _FakeRegistry()
    sched.set_workspace_component_registry(reg)

    spec = _spec(marginal_vram_mb=0)
    spec["_shared_urls"] = {"vlm": "http://localhost:9001"}
    spec["_shared_loaded_by_us"] = []  # canvas-Play loaded it
    run_id = sched.submit(spec)

    q = sched._queue.pop(0)
    sched._running[run_id] = _stub_running(run_id, exited=False, shared_nodesets=q.shared_nodesets)

    _mark_done(tmp_path / run_id)
    sched._running[run_id].proc.poll = lambda self=None: 0  # type: ignore[assignment]
    asyncio.run(sched.tick())

    assert reg.unload_nodeset_calls == []


def test_cancel_queued_releases_shared_refcount(tmp_path: Path) -> None:
    """Cancelling a queued job decrements its shared refcount; if it was
    the only consumer of a freshly-loaded singleton, the singleton gets
    auto-unloaded on the next tick.
    """
    sched = JobScheduler(eval_runs_dir=tmp_path, usable_vram_mb=20000)
    reg = _FakeRegistry()
    sched.set_workspace_component_registry(reg)

    spec = _spec(marginal_vram_mb=0)
    spec["_shared_urls"] = {"vlm": "http://localhost:9001"}
    spec["_shared_loaded_by_us"] = ["vlm"]
    run_id = sched.submit(spec)

    assert sched.cancel(run_id) == "cancelled"
    asyncio.run(sched._drain_pending_unloads())
    assert reg.unload_nodeset_calls == ["vlm"]
