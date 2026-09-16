import random

import pygame

from src.ai.decision_engine import DecisionEngine
from src.ai.ml_classifier import MLCancerCellClassifier
from src.simulation.environment import CellEnvironment

from visualization.cell_scene import VisualCell
from visualization.robot_scene import VisualMicrorobot


WIDTH = 1200
HEIGHT = 760

SIMULATION_WIDTH = 820
PANEL_X = 850

FPS = 60

CELL_COUNT = 35

MODEL_PATH = (
    "models/cancer_classifier.joblib"
)


def create_visual_cells():

    environment = CellEnvironment(
        cancer_fraction=0.35,
        noise_level=0.08,
        seed=200,
    )

    cells = environment.generate_population(
        CELL_COUNT
    )

    random_generator = random.Random(200)

    visual_cells = []

    for cell in cells:

        visual_cells.append(
            VisualCell(
                cell=cell,
                x=random_generator.randint(
                    70,
                    SIMULATION_WIDTH - 70,
                ),
                y=random_generator.randint(
                    70,
                    HEIGHT - 70,
                ),
                radius=random_generator.randint(
                    15,
                    22,
                ),
            )
        )

    return visual_cells


def draw_text(
    surface,
    font,
    text,
    x,
    y,
    color=(225, 225, 230),
):

    rendered = font.render(
        text,
        True,
        color,
    )

    surface.blit(
        rendered,
        (x, y),
    )


def draw_feature_bar(
    surface,
    font,
    label,
    value,
    x,
    y,
):

    draw_text(
        surface,
        font,
        label,
        x,
        y,
    )

    bar_x = x
    bar_y = y + 24

    bar_width = 260
    bar_height = 12

    pygame.draw.rect(
        surface,
        (55, 60, 70),
        (
            bar_x,
            bar_y,
            bar_width,
            bar_height,
        ),
        border_radius=5,
    )

    pygame.draw.rect(
        surface,
        (80, 165, 215),
        (
            bar_x,
            bar_y,
            int(bar_width * value),
            bar_height,
        ),
        border_radius=5,
    )

    draw_text(
        surface,
        font,
        f"{value:.2f}",
        x + 215,
        y,
    )


def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "Cancer AI Microrobot Simulation"
    )

    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont(
        "Arial",
        26,
        bold=True,
    )

    font = pygame.font.SysFont(
        "Arial",
        18,
    )

    small_font = pygame.font.SysFont(
        "Arial",
        15,
    )

    classifier = (
        MLCancerCellClassifier.load(
            MODEL_PATH
        )
    )

    decision_engine = DecisionEngine(
        target_threshold=0.85,
        pass_threshold=0.30,
    )

    visual_cells = create_visual_cells()

    robot = VisualMicrorobot(
        x=40,
        y=HEIGHT / 2,
    )

    current_index = 0

    current_result = None
    current_decision = None

    scan_timer = 0.0
    scanning = False

    cells_scanned = 0
    targets = 0
    passes = 0
    uncertain = 0

    running = True

    while running:

        delta_time = (
            clock.tick(FPS) / 1000.0
        )

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                running = False

        if current_index < len(visual_cells):

            current_cell = (
                visual_cells[current_index]
            )

            if not scanning:

                arrived = robot.move_toward(
                    current_cell.position(),
                    delta_time,
                )

                if arrived:

                    current_result = (
                        classifier.predict(
                            current_cell.cell
                        )
                    )

                    current_decision = (
                        decision_engine.decide(
                            current_result
                        )
                    )

                    current_cell.scanned = True

                    scanning = True
                    scan_timer = 0.0

            else:

                scan_timer += delta_time

                if scan_timer >= 1.3:

                    action = (
                        current_decision.action
                    )

                    if action == "TARGET":

                        targets += 1
                        current_cell.targeted = True

                    elif action == "PASS":

                        passes += 1

                    else:

                        uncertain += 1

                    cells_scanned += 1
                    current_index += 1

                    scanning = False
                    scan_timer = 0.0

        screen.fill(
            (18, 21, 28)
        )

        # Tissue simulation region
        pygame.draw.rect(
            screen,
            (28, 32, 40),
            (
                20,
                20,
                SIMULATION_WIDTH,
                HEIGHT - 40,
            ),
            border_radius=12,
        )

        # Side information panel
        pygame.draw.rect(
            screen,
            (31, 35, 44),
            (
                PANEL_X,
                20,
                WIDTH - PANEL_X - 20,
                HEIGHT - 40,
            ),
            border_radius=12,
        )

        draw_text(
            screen,
            title_font,
            "MICROSCOPIC TISSUE",
            45,
            38,
        )

        for index, visual_cell in enumerate(
            visual_cells
        ):

            visual_cell.draw(
                screen,
                selected=(
                    index == current_index
                ),
            )

        robot.draw(screen)

        # Information panel
        draw_text(
            screen,
            title_font,
            "AI MICROROBOT",
            PANEL_X + 25,
            45,
        )

        draw_text(
            screen,
            font,
            f"Cells scanned: {cells_scanned}",
            PANEL_X + 25,
            95,
        )

        draw_text(
            screen,
            font,
            f"TARGET: {targets}",
            PANEL_X + 25,
            125,
        )

        draw_text(
            screen,
            font,
            f"PASS: {passes}",
            PANEL_X + 25,
            155,
        )

        draw_text(
            screen,
            font,
            f"UNCERTAIN: {uncertain}",
            PANEL_X + 25,
            185,
        )

        pygame.draw.line(
            screen,
            (75, 80, 90),
            (PANEL_X + 25, 220),
            (WIDTH - 45, 220),
            1,
        )

        if (
            current_index
            < len(visual_cells)
        ):

            cell = (
                visual_cells[
                    current_index
                ].cell
            )

            draw_text(
                screen,
                title_font,
                f"CELL {cell.cell_id}",
                PANEL_X + 25,
                245,
            )

            draw_feature_bar(
                screen,
                small_font,
                "Marker A",
                cell.marker_a,
                PANEL_X + 25,
                295,
            )

            draw_feature_bar(
                screen,
                small_font,
                "Marker B",
                cell.marker_b,
                PANEL_X + 25,
                350,
            )

            draw_feature_bar(
                screen,
                small_font,
                "Irregularity",
                cell.irregularity,
                PANEL_X + 25,
                405,
            )

            draw_feature_bar(
                screen,
                small_font,
                "Growth Signal",
                cell.growth_signal,
                PANEL_X + 25,
                460,
            )

        if (
            scanning
            and current_result
            and current_decision
        ):

            probability = (
                current_result
                .cancer_probability
            )

            draw_text(
                screen,
                font,
                "AI CANCER PROBABILITY",
                PANEL_X + 25,
                535,
            )

            draw_text(
                screen,
                title_font,
                f"{probability:.1%}",
                PANEL_X + 25,
                565,
            )

            action = (
                current_decision.action
            )

            if action == "TARGET":
                action_color = (
                    225,
                    85,
                    90,
                )

            elif action == "PASS":
                action_color = (
                    85,
                    205,
                    135,
                )

            else:
                action_color = (
                    235,
                    195,
                    75,
                )

            draw_text(
                screen,
                title_font,
                action,
                PANEL_X + 25,
                610,
                action_color,
            )

            ground_truth = (
                "CANCER"
                if cell.actual_cancer
                else "HEALTHY"
            )

            draw_text(
                screen,
                small_font,
                (
                    "Simulation truth: "
                    + ground_truth
                ),
                PANEL_X + 25,
                660,
                (150, 155, 165),
            )

        elif current_index >= len(
            visual_cells
        ):

            draw_text(
                screen,
                title_font,
                "SIMULATION COMPLETE",
                PANEL_X + 25,
                280,
                (100, 210, 145),
            )

            draw_text(
                screen,
                font,
                (
                    f"{cells_scanned} "
                    "cells analyzed"
                ),
                PANEL_X + 25,
                325,
            )

        draw_text(
            screen,
            small_font,
            "ESC: Exit",
            45,
            HEIGHT - 55,
            (145, 150, 160),
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
