from dataclasses import dataclass
from typing import Any

import pygame

from visualization.robot_scene import VisualMicrorobot


@dataclass
class RobotAgent:

    robot_id: int
    visual: VisualMicrorobot

    current_cell: Any = None

    state: str = "SEARCHING"

    scanning: bool = False
    scan_timer: float = 0.0

    result: Any = None
    decision: Any = None

    cells_scanned: int = 0
    targets: int = 0
    passes: int = 0
    uncertain: int = 0

    exploration_target: pygame.Vector2 | None = None
    exploration_step: int = 0

    def clear_target(self) -> None:

        self.current_cell = None
        self.result = None
        self.decision = None

        self.scanning = False
        self.scan_timer = 0.0

        self.state = "SEARCHING"

    def set_exploration_target(
        self,
        x: float,
        y: float,
    ) -> None:

        self.exploration_target = (
            pygame.Vector2(x, y)
        )

        self.state = "EXPLORING"

    def clear_exploration_target(self) -> None:

        self.exploration_target = None
