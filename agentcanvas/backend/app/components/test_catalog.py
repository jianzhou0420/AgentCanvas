"""The node catalog through its interface, on a private instance."""

from __future__ import annotations

from .bases import BaseCanvasNode, ConfigField, NodeUIConfig, PortDef
from .catalog import NodeCatalog, ports_for


class _Plain(BaseCanvasNode):
    node_type = "t_plain"
    display_name = "Plain"
    category = "tool"
    input_ports = [PortDef("a", "TEXT", "first"), PortDef("b", "TEXT", "second", optional=True)]
    output_ports = [PortDef("out", "TEXT", "result"), PortDef("error", "TEXT", "", optional=True)]
    ui_config = NodeUIConfig(color="sky", config_fields=[ConfigField("k", "text", label="K")])

    async def forward(self, inputs, ctx):
        return {"out": inputs["a"]}


class _Configurable(BaseCanvasNode):
    """Instance ports replace the class ports when ``config.ports`` is set."""

    node_type = "t_configurable"
    display_name = "Configurable"
    category = "tool"
    input_ports = [PortDef("x", "TEXT", "class-level, required")]
    output_ports = [PortDef("y", "TEXT", "")]

    @classmethod
    def _resolve_ports(cls, config: dict) -> tuple[list, list]:
        ports = config.get("ports")
        if ports:
            ins = [
                PortDef(p["name"], p.get("wire_type", "TEXT"), optional=not p.get("required"))
                for p in ports
            ]
            return (ins, [PortDef("y", ports[0].get("wire_type", "TEXT"), "")])
        return (cls.input_ports, cls.output_ports)

    async def forward(self, inputs, ctx):
        return {"y": ""}


class _Broken(BaseCanvasNode):
    node_type = "t_broken"
    display_name = "Broken"
    category = "tool"
    input_ports = [PortDef("p", "TEXT", "")]
    output_ports = []

    @classmethod
    def _resolve_ports(cls, config: dict) -> tuple[list, list]:
        raise RuntimeError("resolver bug")

    async def forward(self, inputs, ctx):
        return {}


# ── Registration ───────────────────────────────────────────────────────


def test_register_get_contains_and_order():
    cat = NodeCatalog([_Plain])
    cat.register(_Configurable)
    assert cat.get("t_plain") is _Plain
    assert "t_configurable" in cat and "nope" not in cat
    assert cat.types() == ["t_plain", "t_configurable"]
    assert len(cat) == 2
    assert cat.get("nope") is None


def test_last_registration_wins():
    class _Plain2(_Plain):
        pass

    cat = NodeCatalog([_Plain])
    cat.register(_Plain2)
    assert cat.get("t_plain") is _Plain2
    assert len(cat) == 1


def test_unregister_reports_what_was_present():
    cat = NodeCatalog([_Plain, _Configurable])
    assert cat.unregister(["t_plain", "ghost"]) == ["t_plain"]
    assert "t_plain" not in cat and "t_configurable" in cat
    assert cat.unregister(iter(["t_configurable"])) == ["t_configurable"]
    assert len(cat) == 0


# ── Ports: the one rule ────────────────────────────────────────────────


def test_ports_of_unknown_type_is_none():
    cat = NodeCatalog()
    assert cat.ports_of("nope") is None
    assert cat.required_inputs("nope") == set()
    assert cat.wire_type("nope", "a", "in") is None


def test_ports_of_defaults_to_class_ports():
    cat = NodeCatalog([_Plain])
    ports = cat.ports_of("t_plain")
    assert [p.name for p in ports.inputs] == ["a", "b"]
    assert [p.name for p in ports.outputs] == ["out", "error"]
    assert ports.required == {"a"}
    assert cat.required_inputs("t_plain", {"anything": 1}) == {"a"}


def test_instance_ports_replace_class_ports():
    cat = NodeCatalog([_Configurable])
    assert cat.required_inputs("t_configurable") == {"x"}
    cfg = {"ports": [{"name": "p", "wire_type": "IMAGE", "required": True}, {"name": "q"}]}
    ports = cat.ports_of("t_configurable", cfg)
    assert [p.name for p in ports.inputs] == ["p", "q"]
    # The class-level required port is gone once the instance resolver answers.
    assert ports.required == {"p"}
    assert cat.wire_type("t_configurable", "p", "in", cfg) == "IMAGE"
    assert cat.wire_type("t_configurable", "y", "out", cfg) == "IMAGE"
    assert cat.wire_type("t_configurable", "x", "in", cfg) is None


def test_raising_resolver_falls_back_to_class_ports():
    cat = NodeCatalog([_Broken])
    ports = cat.ports_of("t_broken", {"ports": [{"name": "z"}]})
    assert [p.name for p in ports.inputs] == ["p"]
    assert cat.required_inputs("t_broken") == {"p"}


def test_wire_type_by_direction():
    cat = NodeCatalog([_Plain])
    assert cat.wire_type("t_plain", "a", "in") == "TEXT"
    assert cat.wire_type("t_plain", "a", "out") is None
    assert cat.wire_type("t_plain", "error", "out") == "TEXT"


def test_ports_for_works_on_an_unregistered_class():
    ports = ports_for(_Configurable, {"ports": [{"name": "only"}]})
    assert [p.name for p in ports.inputs] == ["only"]
    assert ports_for(_Plain).required == {"a"}


# ── Serialization ──────────────────────────────────────────────────────


def test_schema_shape():
    cat = NodeCatalog([_Plain])
    s = cat.schema("t_plain")
    assert s["type"] == "t_plain" and s["display_name"] == "Plain" and s["category"] == "tool"
    assert s["kind"] == "block" and s["ports_mode"] == "mirror"
    assert s["input_ports"] == [
        {"name": "a", "wire_type": "TEXT", "description": "first", "optional": False},
        {"name": "b", "wire_type": "TEXT", "description": "second", "optional": True},
    ]
    assert [p["name"] for p in s["output_ports"]] == ["out", "error"]
    assert s["ui_config"]["color"] == "sky"
    assert s["ui_config"]["config_fields"][0]["name"] == "k"
    assert cat.schema("nope") is None


def test_schemas_follow_registration_order():
    cat = NodeCatalog([_Configurable, _Plain])
    assert [s["type"] for s in cat.schemas()] == ["t_configurable", "t_plain"]
