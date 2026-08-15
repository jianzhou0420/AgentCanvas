"""§14.14 — /snapshot endpoint: identity passthrough + partial-write flagging.

The producer (depth_toolset._publish_snapshot) promises never to publish a
manifest before every versioned file it names is on disk; the endpoint's own
duty is to REPORT any violation (`consistent: false` + `missing`) so the
monitor never silently stitches manifest N+1 to files of N. These tests pin
that contract, plus verbatim passthrough of the new `identity` block.

Runnable two ways:
    ~/miniconda3/envs/agentcanvas/bin/python -m pytest tests/test_snapshot_identity.py -x
    ~/miniconda3/envs/agentcanvas/bin/python tests/test_snapshot_identity.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.execution import coding_agent as ca  # noqa: E402

FILE_KEYS = ("rgb_file", "topdown_file", "accumulated_map_file",
             "model_map_file", "depth_json_file")

IDENTITY = {"run_id": "run_a", "episode": "live_0", "executor": "mini",
            "action_id": "goto#2", "sensor_frame": 41}


def _manifest(n: int, step: int) -> dict:
    return {
        "obs_id": n, "env_step": step, "map_version": n + 10, "gotos": 2,
        "rgb_file": f"obs_{n:04d}_step{step:03d}.png",
        "topdown_file": f"topdown_{n:04d}.png",
        "accumulated_map_file": f"map_{n:04d}.png",
        "model_map_file": f"map_model_{n:04d}.png",
        "depth_json_file": f"depth_{n:04d}.json",
        "identity": IDENTITY, "t": 1.0,
    }


def _write_version(live: Path, snap: dict, *, skip: str | None = None) -> None:
    for key in FILE_KEYS:
        if key == skip:
            continue
        (live / snap[key]).write_bytes(b"x")
    (live / "live_snapshot.json").write_text(json.dumps(snap))


def _call(run: str, index: int = 0):
    return asyncio.run(ca.snapshot(run, index, source="eharness"))


def _with_tmp_root(fn):
    original = ca.SOURCE_ROOTS.get("eharness")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        live = root / "run_a" / "live_0"
        live.mkdir(parents=True)
        ca.SOURCE_ROOTS["eharness"] = root
        try:
            fn(live)
        finally:
            ca.SOURCE_ROOTS["eharness"] = original


def test_no_snapshot_404():
    def body(live: Path):
        try:
            _call("run_a")
        except Exception as e:  # HTTPException
            assert getattr(e, "status_code", None) == 404
        else:
            raise AssertionError("expected 404 when no snapshot exists")
    _with_tmp_root(body)


def test_complete_version_consistent_and_identity_passthrough():
    def body(live: Path):
        snap = _manifest(3, 12)
        _write_version(live, snap)
        out = _call("run_a")
        assert out["consistent"] is True
        assert "missing" not in out
        assert out["identity"] == IDENTITY          # verbatim, not reshaped
        assert out["rgb_file"] == "obs_0003_step012.png"
    _with_tmp_root(body)


def test_partial_next_version_is_flagged_not_mixed():
    def body(live: Path):
        _write_version(live, _manifest(3, 12))
        # v4 lands with one file missing (a crash between writes would look
        # like this if the producer's publish-last rule were ever violated)
        _write_version(live, _manifest(4, 15), skip="model_map_file")
        out = _call("run_a")
        # the endpoint serves the MANIFEST's version and says it is broken —
        # it must never quietly point the reader at a mix of v3 and v4 files
        assert out["obs_id"] == 4
        assert out["consistent"] is False
        assert out["missing"] == ["model_map_file"]
    _with_tmp_root(body)


def test_completed_next_version_switches_clean():
    def body(live: Path):
        _write_version(live, _manifest(3, 12))
        snap4 = _manifest(4, 15)
        _write_version(live, snap4, skip="model_map_file")
        (live / snap4["model_map_file"]).write_bytes(b"x")   # write completes
        out = _call("run_a")
        assert out["obs_id"] == 4 and out["consistent"] is True
    _with_tmp_root(body)


def test_identity_absent_old_runs_tolerated():
    def body(live: Path):
        snap = _manifest(3, 12)
        del snap["identity"]
        _write_version(live, snap)
        out = _call("run_a")
        assert out["consistent"] is True and "identity" not in out
    _with_tmp_root(body)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} passed")
