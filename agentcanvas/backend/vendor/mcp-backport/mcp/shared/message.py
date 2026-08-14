"""
Message wrapper with metadata support.

This module defines a wrapper type that combines JSONRPCMessage with metadata
to support transport-specific features like resumability.
"""
from __future__ import annotations
import typing
from typing import Union

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from mcp.types import JSONRPCMessage, RequestId

ResumptionToken = str

ResumptionTokenUpdateCallback = typing.Callable[[ResumptionToken], typing.Awaitable[None]]

# Callback type for closing SSE streams without terminating
CloseSSEStreamCallback = typing.Callable[[], typing.Awaitable[None]]


@dataclass
class ClientMessageMetadata:
    """Metadata specific to client messages."""

    resumption_token: ResumptionToken | None = None
    on_resumption_token_update: Callable[[ResumptionToken], Awaitable[None]] | None = None


@dataclass
class ServerMessageMetadata:
    """Metadata specific to server messages."""

    related_request_id: RequestId | None = None
    # Request-specific context (e.g., headers, auth info)
    request_context: object | None = None
    # Callback to close SSE stream for the current request without terminating
    close_sse_stream: CloseSSEStreamCallback | None = None
    # Callback to close the standalone GET SSE stream (for unsolicited notifications)
    close_standalone_sse_stream: CloseSSEStreamCallback | None = None


MessageMetadata = Union[ClientMessageMetadata, ServerMessageMetadata, None]


@dataclass
class SessionMessage:
    """A message with specific metadata for transport-specific features."""

    message: JSONRPCMessage
    metadata: MessageMetadata = None
