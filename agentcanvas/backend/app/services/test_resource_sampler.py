"""ResourceSampler parser tests — the pmon format assumption, GPU-free."""

from __future__ import annotations

from .resource_sampler import _parse_pmon

_PMON_SAMPLE = """\
# gpu         pid   type     fb   ccpm    command
# Idx           #    C/G     MB     MB    name
    0       1604     G    157      0    Xorg
    0    3432889     C  10120      0    python
    0    3433103     G    295      0    python
"""

_PMON_IDLE = """\
# gpu         pid   type     fb   ccpm    command
# Idx           #    C/G     MB     MB    name
    0          -     -      -      -    -
"""


def test_parse_pmon_includes_graphics_contexts() -> None:
    rows = {r["pid"]: r for r in _parse_pmon(_PMON_SAMPLE)}
    assert rows[3432889] == {"pid": 3432889, "mem_mb": 10120, "gpu_ctx": "C"}
    # The EGL renderer (type G) must be present — the whole point of pmon.
    assert rows[3433103] == {"pid": 3433103, "mem_mb": 295, "gpu_ctx": "G"}
    assert rows[1604]["gpu_ctx"] == "G"


def test_parse_pmon_idle_gpu_placeholder_rows() -> None:
    assert _parse_pmon(_PMON_IDLE) == []


def test_parse_pmon_sums_multi_gpu_pids() -> None:
    text = _PMON_SAMPLE + "    1    3432889     C   2000      0    python\n"
    rows = {r["pid"]: r for r in _parse_pmon(text)}
    assert rows[3432889]["mem_mb"] == 12120
