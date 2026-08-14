from __future__ import annotations
from .lowlevel import NotificationOptions, Server
from .models import InitializationOptions

try:  # FastMCP needs typing_inspection, which has no py3.8 distribution.
    from .fastmcp import FastMCP
except ImportError:  # py3.8 vendored build: lowlevel Server only
    FastMCP = None

__all__ = ["Server", "FastMCP", "NotificationOptions", "InitializationOptions"]
