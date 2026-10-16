"""AutoServerApp's handler factory: a ``serialized`` node's calls run one
at a time under the app's inference lock; every other node's calls overlap."""

from __future__ import annotations

import asyncio
from typing import Any

from app.components.bases import BaseCanvasNode, BaseNodeSet, PortDef
from app.server.auto_server_app import AutoServerApp


class _Overlap(BaseCanvasNode):
    """Records how many calls were in flight at once."""

    node_type = "asa__overlap"
    input_ports = [PortDef("x", "ANY", "")]
    output_ports = [PortDef("y", "ANY", "")]
    in_flight = 0
    peak = 0

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        cls = type(self)
        cls.in_flight += 1
        cls.peak = max(cls.peak, cls.in_flight)
        await asyncio.sleep(0.01)
        cls.in_flight -= 1
        return {"y": inputs["x"]}


class _Serialized(_Overlap):
    node_type = "asa__serialized"
    serialized = True
    in_flight = 0
    peak = 0


class _NS(BaseNodeSet):
    name = "asa"

    def get_tools(self) -> list:
        return [_Overlap(), _Serialized()]


def _handlers() -> dict[str, Any]:
    app = AutoServerApp(_NS)
    return {f.name: f.handler for f in app.get_functions()}


async def _fan_out(handler: Any, n: int = 4) -> list[dict]:
    return await asyncio.gather(*(handler({"x": i}, {}) for i in range(n)))


def test_plain_node_calls_overlap() -> None:
    _Overlap.peak = 0
    outs = asyncio.run(_fan_out(_handlers()["asa__overlap"]))
    assert [o["y"] for o in outs] == [0, 1, 2, 3]
    assert _Overlap.peak == 4


def test_serialized_node_calls_run_one_at_a_time() -> None:
    _Serialized.peak = 0
    outs = asyncio.run(_fan_out(_handlers()["asa__serialized"]))
    assert [o["y"] for o in outs] == [0, 1, 2, 3]
    assert _Serialized.peak == 1


def test_inference_lock_is_created_inside_the_running_loop() -> None:
    app = AutoServerApp(_NS)
    assert app._inference_lock is None  # not bound to a loop at construction

    async def first_use() -> asyncio.Lock:
        return app._get_inference_lock()

    lock = asyncio.run(first_use())
    assert isinstance(lock, asyncio.Lock) and app._get_inference_lock() is lock
