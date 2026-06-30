"""Round-trip tests for the server-mode wire codec.

Covers Move 1 (msgpack + ``blob`` ExtType) and the retained JSON migration
path. Run from the backend dir:

    python -m pytest app/server/test_serialization.py -v
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from app.server import serialization as S
from app.server.serialization import (
    deserialize_value,
    pack_body,
    serialize_value,
    unpack_body,
)


def test_msgpack_available() -> None:
    assert S.MSGPACK_OK is True


def test_roundtrip_scalars_and_containers() -> None:
    obj = {
        "i": 3,
        "f": 2.5,
        "b": True,
        "s": "hi",
        "n": None,
        "lst": [1, 2, [3, "x"]],
        "d": {"k": "v", "nested": {"z": 9}},
    }
    assert unpack_body(pack_body(obj)) == obj


def test_roundtrip_bytes_native_bin() -> None:
    out = unpack_body(pack_body({"raw": b"\x00\x01\x02\xff"}))
    assert out["raw"] == b"\x00\x01\x02\xff"
    assert isinstance(out["raw"], (bytes, bytearray))


@pytest.mark.parametrize("dtype", ["uint8", "float32", "float64", "int64", "int16"])
def test_roundtrip_ndarray(dtype: str) -> None:
    arr = np.arange(24).reshape(2, 3, 4).astype(dtype)
    out = unpack_body(pack_body({"a": arr}))["a"]
    assert isinstance(out, np.ndarray)
    assert out.dtype == arr.dtype
    assert out.shape == arr.shape
    np.testing.assert_array_equal(out, arr)
    out[0, 0, 0] = 7  # decoded array must be writable (frombuffer().copy())


def test_roundtrip_depth_float32_exact() -> None:
    # DEPTH is float32 metres — exact metric scale must survive (no PNG path).
    depth = (np.random.rand(5, 7) * 10.0).astype(np.float32)
    out = unpack_body(pack_body({"d": depth}))["d"]
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, depth)


def test_numpy_scalars_become_python() -> None:
    out = unpack_body(pack_body({"i": np.int64(5), "f": np.float32(1.5), "b": np.bool_(True)}))
    assert isinstance(out["i"], int) and out["i"] == 5
    assert out["f"] == pytest.approx(1.5)
    assert out["b"] is True


def test_roundtrip_torch() -> None:
    torch = pytest.importorskip("torch")
    t = torch.arange(6).reshape(2, 3).to(torch.float32)
    out = unpack_body(pack_body({"t": t}))["t"]
    assert isinstance(out, torch.Tensor)
    np.testing.assert_array_equal(out.numpy(), t.numpy())


def test_torch_receiver_degrade(monkeypatch: pytest.MonkeyPatch) -> None:
    """A torch blob decoded in an env without torch comes back as ndarray."""
    torch = pytest.importorskip("torch")
    t = torch.arange(6).reshape(2, 3).to(torch.float32)
    packed = pack_body({"t": t})  # encoded with the torch tag (torch present)
    monkeypatch.setitem(sys.modules, "torch", None)  # make `import torch` raise
    out = unpack_body(packed)["t"]
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, t.numpy())


def test_roundtrip_pil() -> None:
    Image = pytest.importorskip("PIL.Image")
    arr = np.zeros((4, 5, 3), dtype=np.uint8)
    arr[1, 2] = (10, 20, 30)
    out = unpack_body(pack_body({"img": Image.fromarray(arr)}))["img"]
    assert isinstance(out, Image.Image)
    np.testing.assert_array_equal(np.array(out), arr)


def test_unserializable_raises() -> None:
    class Weird:
        pass

    with pytest.raises(TypeError):
        pack_body({"x": Weird()})


def test_json_path_still_roundtrips() -> None:
    # Migration window: the legacy __ndarray__ marker must still work.
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    enc = serialize_value(arr, "ANY")
    assert isinstance(enc, dict) and "__ndarray__" in enc
    np.testing.assert_array_equal(deserialize_value(enc, "ANY"), arr)


# ── /call route content-type negotiation (integration) ──


def _echo_app():
    from app.server.manifest import PortSchema
    from app.server.server_app import ServerApp, ServerFunction

    class EchoApp(ServerApp):
        name = "echo"

        def get_functions(self):
            async def echo(inputs, config):
                return {"y": inputs.get("x")}

            return [
                ServerFunction(
                    name="echo__id",
                    input_ports=[PortSchema(name="x", wire_type="ANY")],
                    output_ports=[PortSchema(name="y", wire_type="ANY")],
                    handler=echo,
                )
            ]

    return EchoApp()._build_app()


def test_call_msgpack_roundtrip() -> None:
    from fastapi.testclient import TestClient

    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    with TestClient(_echo_app()) as client:
        body = pack_body({"inputs": {"x": arr}, "config": {}})
        r = client.post(
            "/call/echo__id", content=body, headers={"Content-Type": S.MSGPACK_CONTENT_TYPE}
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(S.MSGPACK_CONTENT_TYPE)
    out = unpack_body(r.content)["outputs"]["y"]
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)


def test_call_json_migration_window() -> None:
    from fastapi.testclient import TestClient

    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    with TestClient(_echo_app()) as client:
        body = {"inputs": {"x": serialize_value(arr, "ANY")}, "config": {}}
        r = client.post("/call/echo__id", json=body)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    out = deserialize_value(r.json()["outputs"]["y"], "ANY")
    np.testing.assert_array_equal(out, arr)


# ── sub-home /containers/* msgpack hop (Move 2 container path) ──


class _FakeContainer:
    def __init__(self) -> None:
        self.store: dict = {}

    def read(self, name, key=None):
        if name not in self.store:
            raise KeyError(name)
        return self.store[name]

    def write(self, name, data, key=None) -> None:
        self.store[name] = data

    def evict(self, key) -> None:
        self.store.clear()


def _owner_app():
    from app.server.server_app import ServerApp

    class OwnerApp(ServerApp):
        name = "owner"

        def __init__(self) -> None:
            super().__init__()
            self._c = _FakeContainer()

        def get_functions(self):
            return []

        def get_owned_container(self, container_id: str):
            return self._c if container_id == "c1" else None

    return OwnerApp()._build_app()


def test_container_endpoints_msgpack_roundtrip() -> None:
    from fastapi.testclient import TestClient

    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    with TestClient(_owner_app()) as client:
        w = client.post(
            "/containers/c1/write",
            content=pack_body({"name": "voxels", "data": arr}),
            headers={"Content-Type": S.MSGPACK_CONTENT_TYPE},
        )
        assert w.status_code == 200
        assert unpack_body(w.content) == {"ok": True}

        r = client.post(
            "/containers/c1/read",
            content=pack_body({"name": "voxels"}),
            headers={"Content-Type": S.MSGPACK_CONTENT_TYPE},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(S.MSGPACK_CONTENT_TYPE)
    out = unpack_body(r.content)["value"]
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)


def test_container_unknown_id_404() -> None:
    from fastapi.testclient import TestClient

    with TestClient(_owner_app()) as client:
        r = client.post(
            "/containers/nope/read",
            content=pack_body({"name": "x"}),
            headers={"Content-Type": S.MSGPACK_CONTENT_TYPE},
        )
    assert r.status_code == 404
