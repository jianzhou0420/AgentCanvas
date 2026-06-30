"""AgentCanvas Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ── WebSocket ──


class WSMessage(BaseModel):
    type: str
    data: Any = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_id: str | None = None
    source: str | None = None  # "canvas" | "eval" — for frontend event routing
