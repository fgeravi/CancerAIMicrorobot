import pygame


class VisualMicrorobot:

    def __init__(
        self,
        x: float,
        y: float,
        speed: float = 150.0,
    ):
        self.position = pygame.Vector2(
            x,
            y,
        )

        self.speed = speed

    def move_toward(
        self,
        target: pygame.Vector2,
        delta_time: float,
    ) -> bool:

        difference = (
            target - self.position
        )

        distance = difference.length()

        if distance < 2.0:
            self.position = target.copy()
            return True

        direction = difference.normalize()

        movement = (
            direction
            * self.speed
            * delta_time
        )

        if movement.length() >= distance:
            self.position = target.copy()
            return True

        self.position += movement

        return False

    def draw(
        self,
        surface: pygame.Surface,
    ) -> None:

        center = (
            int(self.position.x),
            int(self.position.y),
        )

        # Outer robot body
        pygame.draw.circle(
            surface,
            (75, 165, 220),
            center,
            12,
        )

        pygame.draw.circle(
            surface,
            (220, 240, 250),
            center,
            12,
            2,
        )

        # Inner core
        pygame.draw.circle(
            surface,
            (25, 70, 100),
            center,
            5,
        )

        # Sensor arms
        pygame.draw.line(
            surface,
            (150, 210, 240),
            (center[0] - 16, center[1]),
            (center[0] + 16, center[1]),
            2,
        )

        pygame.draw.line(
            surface,
            (150, 210, 240),
            (center[0], center[1] - 16),
            (center[0], center[1] + 16),
            2,
        )
