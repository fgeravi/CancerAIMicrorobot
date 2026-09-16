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

    def position(self) -> pygame.Vector2:
        return pygame.Vector2(
            self.x,
            self.y,
        )

    def draw(
        self,
        surface: pygame.Surface,
        selected: bool = False,
    ) -> None:

        center = (
            int(self.x),
            int(self.y),
        )

        if self.targeted:
            color = (110, 110, 110)

        elif self.cell.actual_cancer:
            color = (195, 70, 80)

        else:
            color = (80, 175, 125)

        pygame.draw.circle(
            surface,
            color,
            center,
            self.radius,
        )

        pygame.draw.circle(
            surface,
            (230, 230, 235),
            center,
            self.radius,
            2,
        )

        # Nucleus
        pygame.draw.circle(
            surface,
            (65, 65, 85),
            center,
            max(5, self.radius // 3),
        )

        if self.scanned:
            pygame.draw.circle(
                surface,
                (120, 170, 220),
                center,
                self.radius + 4,
                2,
            )

        if selected:
            pygame.draw.circle(
                surface,
                (245, 215, 90),
                center,
                self.radius + 8,
                3,
            )
