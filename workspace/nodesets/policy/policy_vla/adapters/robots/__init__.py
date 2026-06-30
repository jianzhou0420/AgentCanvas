"""Robot adapter sub-package — env-coupled side of the Adapter system.

Each .py file under this folder that defines a `RobotAdaptor` subclass and a
module-level `DEFAULT_KWARGS` dict is auto-discovered by the canvas
`adapt_env_to_canonical` node and surfaces as a dropdown option.

Drop in / drop out:
    +  add a new .py here with `DEFAULT_KWARGS = {...}` → new dropdown entry
    -  delete the .py → option vanishes (after `POST /api/components/reload`)
"""

from __future__ import annotations

from .base_robot import RobotAdaptor
from .libero_robot import LiberoRobot
from .simpler_robot import SimplerRobot

__all__ = ["LiberoRobot", "RobotAdaptor", "SimplerRobot"]
