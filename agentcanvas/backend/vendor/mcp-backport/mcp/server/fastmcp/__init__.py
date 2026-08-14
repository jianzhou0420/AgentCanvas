"""FastMCP - A more ergonomic interface for MCP servers."""
from __future__ import annotations

from importlib.metadata import version

from mcp.types import Icon

from .server import Context, FastMCP
from .utilities.types import Audio, Image

try:
    __version__ = version("mcp")
except Exception:  # vendored py3.9 backport runs from source, no dist-info
    __version__ = "1.27.0+py39"
__all__ = ["FastMCP", "Context", "Image", "Audio", "Icon"]
