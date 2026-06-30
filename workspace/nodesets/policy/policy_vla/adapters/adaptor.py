"""Composed Adaptor that bridges RobotAdaptor and ModelAdaptor.

Vendored from vlaworkspace/adaptors/adaptor.py — no upstream import.

The Adaptor class composes a RobotAdaptor (robot-specific) and a ModelAdaptor
(model-specific) via the canonical intermediate format.

Data flow:
    Dataset -> Robot.dataset_to_canonical() -> Model.canonical_to_model() -> Model
    Env     -> Robot.env_to_canonical()     -> Model.canonical_to_model() -> Model
    Model   -> Model.model_to_canonical()   -> Robot.canonical_to_env()   -> Env
"""

from __future__ import annotations

import logging

from .models.base_model import ModelAdaptor
from .robots.base_robot import RobotAdaptor

logger = logging.getLogger(__name__)


class Adaptor:
    """Composed adaptor bridging Robot + Model via canonical format."""

    def __init__(
        self,
        *,
        robot: RobotAdaptor,
        model: ModelAdaptor,
    ) -> None:
        self.robot = robot
        self.model = model
        self._norm_stats = model.get_norm_stats()

        logger.info(
            f"Adaptor created: robot={type(robot).__name__}, model={type(model).__name__}"
        )

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def datasets_input_transforms(self, data: dict) -> dict:
        canonical = self.robot.dataset_to_canonical(data)
        return self.model.canonical_to_model(canonical)

    def env_input_transforms(self, data: dict) -> dict:
        canonical = self.robot.env_to_canonical(data)
        return self.model.canonical_to_model(canonical)

    def output_transforms(self, data: dict) -> dict:
        info = self.robot.get_canonical_info()
        canonical_action = self.model.model_to_canonical(data, info)

        state = None
        if "state" in data:
            state = data

        return self.robot.canonical_to_env(canonical_action, state=state)

    def get_state_dim(self) -> int:
        return self.robot.get_state_dim()

    def get_action_dim(self) -> int:
        return self.robot.get_action_dim()

    def get_norm_stats(self) -> dict | None:
        return self.model.get_norm_stats()

    def get_norm_stats_keys(self) -> tuple[str, ...]:
        return self.model.get_norm_stats_keys()

    def get_norm_stats_mode(self) -> str:
        return self.model.get_norm_stats_mode()
