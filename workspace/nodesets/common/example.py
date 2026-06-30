"""Example nodeset — demonstrates the execution log system.

Five nodes with different voluntary logging patterns:
- TextSource:          seed node (no inputs), outputs configurable text
- WordCounter:         logs simple scalar values
- SentimentTag:        logs structured data (dicts)
- TextSummary:         logs multiple entries with format hints
- RandomTriangleImage: generates an IMAGE, demonstrates asset logging

Load:  POST /api/components/nodesets/example/load
"""

from __future__ import annotations

import random as _random
from typing import Any, ClassVar

import numpy as np

from app.components import BaseCanvasNode, BaseNodeSet, NodeUIConfig, PortDef
from app.components.bases import ConfigField

# ── Node 1: Seed node — outputs configurable text ──


class TextSource(BaseCanvasNode):
    """Seed node that outputs user-configured text. No inputs required."""

    node_type = "example__source"
    display_name = "Text Source"
    description = "Output a configurable text string (seed node)"
    category = "example"
    icon = "Type"

    input_ports: ClassVar[list] = []
    output_ports: ClassVar[list] = [
        PortDef("text", "TEXT", "The configured text"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="sky",
        config_fields=[
            ConfigField(
                "text",
                "textarea",
                label="Input Text",
                default="The weather today is great and wonderful but the traffic is terrible and the roads are in poor condition",
                placeholder="Enter text to analyze...",
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        text = self.config.get(
            "text",
            "The weather today is great and wonderful but the traffic is terrible",
        )
        self._self_log("text_length", len(text))
        self._self_log("preview", text[:80])
        return {"text": text}


# ── Node 2: Simple scalar logging ──


class WordCounter(BaseCanvasNode):
    """Count words in text. Logs the word count and unique word count."""

    node_type = "example__word_count"
    display_name = "Word Counter"
    description = "Count words in input text"
    category = "example"
    icon = "Hash"

    input_ports: ClassVar[list] = [PortDef("text", "TEXT", "Input text to analyze")]
    output_ports: ClassVar[list] = [
        PortDef("count", "TEXT", "Word count as string"),
        PortDef("text", "TEXT", "Pass-through text"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="blue")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        text = inputs.get("text", "")
        words = text.split()
        unique = {w.lower() for w in words}

        # Voluntary logging — simple scalars
        self._self_log("word_count", len(words))
        self._self_log("unique_words", len(unique))
        self._self_log(
            "avg_word_length",
            round(
                sum(len(w) for w in words) / max(len(words), 1),
                1,
            ),
        )

        return {"count": str(len(words)), "text": text}


# ── Node 3: Structured data logging ──


class SentimentTag(BaseCanvasNode):
    """Tag text with a simple keyword-based sentiment. Logs analysis details."""

    node_type = "example__sentiment"
    display_name = "Sentiment Tagger"
    description = "Tag text with positive/negative/neutral sentiment"
    category = "example"
    icon = "ThumbsUp"

    input_ports: ClassVar[list] = [PortDef("text", "TEXT", "Text to analyze")]
    output_ports: ClassVar[list] = [
        PortDef("sentiment", "TEXT", "Sentiment label"),
        PortDef("text", "TEXT", "Pass-through text"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="green")

    POSITIVE = {"good", "great", "excellent", "happy", "love", "wonderful", "best"}
    NEGATIVE = {"bad", "terrible", "awful", "hate", "worst", "poor", "horrible"}

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        text = inputs.get("text", "")
        words = set(text.lower().split())

        pos_hits = words & self.POSITIVE
        neg_hits = words & self.NEGATIVE

        if len(pos_hits) > len(neg_hits):
            sentiment = "positive"
        elif len(neg_hits) > len(pos_hits):
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Voluntary logging — structured dict
        self._self_log(
            "analysis",
            {
                "positive_matches": sorted(pos_hits),
                "negative_matches": sorted(neg_hits),
                "positive_score": len(pos_hits),
                "negative_score": len(neg_hits),
                "result": sentiment,
            },
        )

        return {"sentiment": sentiment, "text": text}


# ── Node 4: Multiple log entries ──


class TextSummary(BaseCanvasNode):
    """Generate a text summary. Logs the full analysis pipeline."""

    node_type = "example__summary"
    display_name = "Text Summary"
    description = "Summarize text analysis results"
    category = "example"
    icon = "FileText"

    input_ports: ClassVar[list] = [
        PortDef("text", "TEXT", "Original text"),
        PortDef("word_count", "TEXT", "Word count"),
        PortDef("sentiment", "TEXT", "Sentiment label"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("summary", "TEXT", "Analysis summary"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="violet",
        config_fields=[
            ConfigField(
                "format",
                "select",
                label="Output format",
                options=[
                    {"value": "brief", "label": "Brief"},
                    {"value": "detailed", "label": "Detailed"},
                ],
                default="brief",
            ),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        text = inputs.get("text", "")
        count = inputs.get("word_count", "0")
        sentiment = inputs.get("sentiment", "neutral")
        fmt = self.config.get("format", "brief")

        # Log the assembly process step by step
        self._self_log(
            "input_lengths",
            {
                "text": len(text),
                "word_count": count,
                "sentiment": sentiment,
            },
        )
        self._self_log("format_mode", fmt)

        if fmt == "detailed":
            summary = (
                f"Analysis of {count}-word text:\n"
                f"  Sentiment: {sentiment}\n"
                f"  Preview: {text[:100]}{'...' if len(text) > 100 else ''}"
            )
        else:
            summary = f"{count} words, {sentiment} sentiment"

        self._self_log("output_length", len(summary))

        return {"summary": summary}


# ── Node 5: Image generation — demonstrates IMAGE wire type + asset logging ──


class RandomTriangleImage(BaseCanvasNode):
    """Generate a random colored triangle on a white canvas.

    Demonstrates IMAGE wire type logging — the executor saves the output
    as a sidecar JPEG file in ``outputs/runs/{id}/assets/``.
    """

    node_type = "example__random_triangle"
    display_name = "Random Triangle"
    description = "Generate a random colored triangle image"
    category = "example"
    icon = "Triangle"

    input_ports: ClassVar[list] = []
    output_ports: ClassVar[list] = [
        PortDef("rgb", "IMAGE", "Generated RGB image"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="orange",
        config_fields=[
            ConfigField("width", "slider", label="Width", default=320, min=64, max=640, step=64),
            ConfigField("height", "slider", label="Height", default=240, min=64, max=480, step=64),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        w = int(self.config.get("width", 320))
        h = int(self.config.get("height", 240))

        # White canvas
        img = np.full((h, w, 3), 255, dtype=np.uint8)

        # Random triangle color
        color = [_random.randint(30, 230) for _ in range(3)]

        # Three random vertices
        pts = [(_random.randint(0, w - 1), _random.randint(0, h - 1)) for _ in range(3)]

        # Fill triangle using scanline (no PIL/cv2 dependency)
        _fill_triangle(img, pts, color)

        self._self_log("image_shape", list(img.shape))
        self._self_log("triangle_color", color)
        self._self_log("vertices", pts)

        return {"rgb": img}


def _fill_triangle(
    img: np.ndarray,
    pts: list[tuple[int, int]],
    color: list[int],
) -> None:
    """Fill a triangle on img using scanline rasterization."""
    h, w = img.shape[:2]
    # Sort by y
    pts_sorted = sorted(pts, key=lambda p: p[1])
    (_x0, y0), (_x1, _y1), (_x2, y2) = pts_sorted

    def _interp_x(ya: int, yb: int, xa: int, xb: int, y: int) -> int:
        if ya == yb:
            return xa
        return xa + (xb - xa) * (y - ya) // (yb - ya)

    for y in range(max(0, y0), min(h, y2 + 1)):
        # Find x bounds on this scanline
        xs: list[int] = []
        for (ax, ay), (bx, by) in [
            (pts_sorted[0], pts_sorted[1]),
            (pts_sorted[1], pts_sorted[2]),
            (pts_sorted[0], pts_sorted[2]),
        ]:
            if ay == by:
                if y == ay:
                    xs.extend([ax, bx])
            elif min(ay, by) <= y <= max(ay, by):
                xs.append(_interp_x(ay, by, ax, bx, y))
        if len(xs) >= 2:
            x_min = max(0, min(xs))
            x_max = min(w - 1, max(xs))
            img[y, x_min : x_max + 1] = color


# ══════════════════════════════════════════════════════════════════════
# Tutorial companion nodes (Component Cookbook)
# ══════════════════════════════════════════════════════════════════════


class GreetNode(BaseCanvasNode):
    """Ch 1 — Your First Node: generate a greeting message."""

    node_type = "example__greet"
    display_name = "Greet"
    description = "Generate a greeting for a given name"
    category = "example"
    icon = "Hand"

    input_ports: ClassVar[list] = [
        PortDef("name", "TEXT", "Person's name"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("greeting", "TEXT", "The greeting message"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="sky")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        name = inputs.get("name", "World")
        greeting = f"Hello, {name}!"
        self._self_log("name_received", name)
        return {"greeting": greeting}


class ShoutNode(BaseCanvasNode):
    """Ch 7 — NodeSet grouping: convert text to uppercase."""

    node_type = "example__shout"
    display_name = "Shout"
    description = "Convert text to uppercase"
    category = "example"
    icon = "Volume2"

    input_ports: ClassVar[list] = [
        PortDef("text", "TEXT", "Input text"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("shouted", "TEXT", "UPPERCASED text"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="orange")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        text = str(inputs.get("text", ""))
        return {"shouted": text.upper()}


class TemperatureTextNode(BaseCanvasNode):
    """Ch 2 — Configuring Nodes: all six ConfigField types."""

    node_type = "example__temp_text"
    display_name = "Temperature Text"
    description = "Convert a temperature value to descriptive text"
    category = "example"
    icon = "Thermometer"

    input_ports: ClassVar[list] = [
        PortDef("value", "TEXT", "Temperature in Celsius"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("description", "TEXT", "Descriptive text"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="orange",
        config_fields=[
            ConfigField(
                "unit",
                "select",
                label="Unit",
                options=[
                    {"value": "celsius", "label": "Celsius"},
                    {"value": "fahrenheit", "label": "Fahrenheit"},
                ],
                default="celsius",
            ),
            ConfigField(
                "threshold", "slider", label="Hot threshold", default=30, min=0, max=50, step=1
            ),
            ConfigField("verbose", "toggle", label="Verbose output", default=False),
        ],
    )

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        temp = float(inputs.get("value", "0"))
        unit = self.config.get("unit", "celsius")
        threshold = self.config.get("threshold", 30)
        verbose = self.config.get("verbose", False)

        if unit == "fahrenheit":
            temp = (temp - 32) * 5 / 9

        if temp >= threshold:
            label = "hot"
        elif temp >= threshold - 15:
            label = "mild"
        else:
            label = "cold"

        self._self_log("converted_temp", round(temp, 1))
        self._self_log("label", label)

        if verbose:
            return {"description": f"{temp:.1f}°C is {label} (threshold: {threshold}°C)"}
        return {"description": label}


class TimestampNode(BaseCanvasNode):
    """Ch 1 — Seed node example: output current timestamp."""

    node_type = "example__timestamp"
    display_name = "Current Time"
    description = "Output the current timestamp (seed node — no inputs)"
    category = "example"
    icon = "Clock"

    input_ports: ClassVar[list] = []
    output_ports: ClassVar[list] = [
        PortDef("time", "TEXT", "ISO timestamp"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(color="cyan")

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        from datetime import datetime, timezone

        return {"time": datetime.now(timezone.utc).isoformat()}


class FlexibleMergeNode(BaseCanvasNode):
    """Ch 6 — Dynamic Ports: merge N text inputs via config.ports."""

    node_type = "example__flex_merge"
    display_name = "Flexible Merge"
    description = "Merge N text inputs into a single output (dynamic ports)"
    category = "example"
    icon = "Merge"

    input_ports: ClassVar[list] = [
        PortDef("a", "TEXT", "First input"),
        PortDef("b", "TEXT", "Second input"),
    ]
    output_ports: ClassVar[list] = [
        PortDef("merged", "TEXT", "Merged text"),
    ]

    ui_config: ClassVar[NodeUIConfig] = NodeUIConfig(
        color="sky",
        config_fields=[
            ConfigField("separator", "text", label="Separator", default="\n"),
            ConfigField("ports", "port_list", label="Input Ports"),
        ],
    )

    @classmethod
    def _resolve_ports(cls, config: dict) -> tuple[list, list]:
        custom_ports = config.get("ports")
        if custom_ports:
            inputs = [
                PortDef(p["name"], p.get("wire_type", "TEXT"), p.get("name", ""))
                for p in custom_ports
            ]
            return (inputs, cls.output_ports)
        return (cls.input_ports, cls.output_ports)

    async def forward(self, inputs: dict, ctx: Any) -> dict:
        sep = self.config.get("separator", "\n")
        parts = [str(v) for v in inputs.values() if v is not None]
        self._self_log("input_count", len(parts))
        return {"merged": sep.join(parts)}


# ── NodeSet Registration ──


class ExampleNodeSet(BaseNodeSet):
    name = "example"
    description = "Example tools — execution log demos + Component Cookbook companions"

    def get_tools(self) -> list:
        return [
            # Execution log demos (tutorials/execution-logs.md)
            TextSource(),
            WordCounter(),
            SentimentTag(),
            TextSummary(),
            RandomTriangleImage(),
            # Component Cookbook companions (tutorials/component-cookbook.md)
            GreetNode(),
            ShoutNode(),
            TemperatureTextNode(),
            TimestampNode(),
            FlexibleMergeNode(),
        ]
