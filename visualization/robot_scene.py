import math

import pygame


class VisualMicrorobot:

    def __init__(
        self,
        x: float,
        y: float,
        speed: float = 155.0,
        sensor_range: float = 170.0,
    ):
        self.position = pygame.Vector2(x, y)

        self.speed = speed
        self.sensor_range = sensor_range

        self.angle = 0.0
        self.distance_traveled = 0.0

    def move_toward(
        self,
        target: pygame.Vector2,
        delta_time: float,
    ) -> bool:

        difference = target - self.position
        distance = difference.length()

        if distance < 2.0:
            self.distance_traveled += distance
            self.position = target.copy()
            return True

        direction = difference.normalize()

        self.angle = math.atan2(
            direction.y,
            direction.x,
        )

        movement = (
            direction
            * self.speed
            * delta_time
        )

        if movement.length() >= distance:

            self.distance_traveled += distance

            self.position = target.copy()

            return True

        self.position += movement

        self.distance_traveled += (
            movement.length()
        )

        return False

    def reset(
        self,
        x: float,
        y: float,
    ) -> None:

        self.position.update(x, y)

        self.angle = 0.0
        self.distance_traveled = 0.0

    def draw_sensor_range(
        self,
        surface: pygame.Surface,
    ) -> None:

        center = (
            int(self.position.x),
            int(self.position.y),
        )

        overlay = pygame.Surface(
            surface.get_size(),
            pygame.SRCALPHA,
        )

        pygame.draw.circle(
            overlay,
            (75, 145, 195, 12),
            center,
            int(self.sensor_range),
        )

        pygame.draw.circle(
            overlay,
            (85, 155, 205, 40),
            center,
            int(self.sensor_range),
            1,
        )

        surface.blit(
            overlay,
            (0, 0),
        )

    def draw(
        self,
        surface: pygame.Surface,
        scanning: bool = False,
        pulse: float = 0.0,
        robot_id: int | None = None,
        font: pygame.font.Font | None = None,
    ) -> None:

        center = (
            int(self.position.x),
            int(self.position.y),
        )

        if scanning:

            scan_radius = int(
                22 + pulse * 24
            )

            pygame.draw.circle(
                surface,
                (75, 145, 195),
                center,
                scan_radius,
                2,
            )

        for offset in [
            (-18, 0),
            (18, 0),
            (0, -18),
            (0, 18),
        ]:

            pygame.draw.line(
                surface,
                (125, 195, 225),
                center,
                (
                    center[0] + offset[0],
                    center[1] + offset[1],
                ),
                3,
            )

        pygame.draw.circle(
            surface,
            (65, 155, 215),
            center,
            13,
        )

        pygame.draw.circle(
            surface,
            (215, 235, 245),
            center,
            13,
            2,
        )

        pygame.draw.circle(
            surface,
            (25, 65, 95),
            center,
            6,
        )

        direction = pygame.Vector2(
            math.cos(self.angle),
            math.sin(self.angle),
        )

        endpoint = (
            self.position
            + direction * 19
        )

        pygame.draw.line(
            surface,
            (235, 240, 245),
            center,
            (
                int(endpoint.x),
                int(endpoint.y),
            ),
            2,
        )

        if (
            robot_id is not None
            and font is not None
        ):

            label = font.render(
                f"R{robot_id}",
                True,
                (235, 240, 245),
            )

            label_rect = label.get_rect(
                center=(
                    center[0],
                    center[1] - 29,
                )
            )

            background = label_rect.inflate(
                8,
                4,
            )

            pygame.draw.rect(
                surface,
                (35, 42, 52),
                background,
                border_radius=5,
            )

            surface.blit(
                label,
                label_rect,
            )
