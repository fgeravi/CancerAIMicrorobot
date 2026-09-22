from dataclasses import dataclass

import pygame

from src.models.cell import Cell


@dataclass
class VisualCell:
    cell: Cell
    x: float
    y: float
    radius: int = 18

    targeted: bool = False
    scanned: bool = False
    uncertain: bool = False

    # ID of the robot currently traveling to or
    # scanning this cell.
    claimed_by: int | None = None

    def position(self) -> pygame.Vector2:
        return pygame.Vector2(self.x, self.y)

    def draw(
        self,
        surface: pygame.Surface,
        selected: bool = False,
        show_ground_truth: bool = True,
    ) -> None:

        center = (
            int(self.x),
            int(self.y),
        )

        # When ground truth is hidden, cells look similar.
        if not show_ground_truth:
            color = (135, 120, 150)

        elif self.cell.actual_cancer:
            color = (190, 72, 82)

        else:
            color = (75, 165, 120)

        # A targeted cell becomes visibly damaged/dimmed.
        if self.targeted:
            color = tuple(
                max(35, component // 2)
                for component in color
            )

        pygame.draw.circle(
            surface,
            color,
            center,
            self.radius,
        )

        pygame.draw.circle(
            surface,
            (225, 225, 235),
            center,
            self.radius,
            2,
        )

        # Nucleus
        pygame.draw.circle(
            surface,
            (58, 55, 78),
            center,
            max(5, self.radius // 3),
        )

        # Previously scanned cell
        if self.scanned:
            pygame.draw.circle(
                surface,
                (95, 155, 205),
                center,
                self.radius + 4,
                2,
            )

        # Uncertain cells retain a yellow ring.
        if self.uncertain:
            pygame.draw.circle(
                surface,
                (225, 185, 70),
                center,
                self.radius + 6,
                2,
            )

        # Current robot destination
        if selected:
            pygame.draw.circle(
                surface,
                (245, 215, 90),
                center,
                self.radius + 9,
                3,
            )

        # Cross over cells receiving simulated targeting.
        if self.targeted:
            size = self.radius - 4

            pygame.draw.line(
                surface,
                (235, 95, 95),
                (
                    center[0] - size,
                    center[1] - size,
                ),
                (
                    center[0] + size,
                    center[1] + size,
                ),
                3,
            )

            pygame.draw.line(
                surface,
                (235, 95, 95),
                (
                    center[0] + size,
                    center[1] - size,
                ),
                (
                    center[0] - size,
                    center[1] + size,
                ),
                3,
            )
