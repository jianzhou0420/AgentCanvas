"""Custom exceptions for FastMCP."""
from __future__ import annotations


class FastMCPError(Exception):
    """Base error for FastMCP."""


class ValidationError(FastMCPError):
    """Error in validating parameters or return values."""


class ResourceError(FastMCPError):
    """Error in resource operations."""


class ToolError(FastMCPError):
    """Error in tool operations."""


class InvalidSignature(Exception):
    """Invalid signature for use with FastMCP."""
