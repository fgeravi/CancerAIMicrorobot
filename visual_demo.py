import math
import random

import pygame

from src.ai.decision_engine import DecisionEngine
from src.ai.ml_classifier import MLCancerCellClassifier
from src.simulation.environment import CellEnvironment
from src.simulation.swarm import RobotSwarm
from src.simulation.navigation import (
    claim_cell,
    find_nearest_available_cell,
    release_cell,
)
from src.simulation.exploration import (
    next_exploration_point,
)

from visualization.cell_scene import VisualCell


WIDTH = 1440
HEIGHT = 960

SIMULATION_LEFT = 20
SIMULATION_TOP = 20
SIMULATION_WIDTH = 870
SIMULATION_HEIGHT = 760

PANEL_X = 910
PANEL_WIDTH = WIDTH - PANEL_X - 20
FOOTER_Y = 850

FPS = 60
CELL_COUNT = 42

MODEL_PATH = "models/cancer_classifier.joblib"


def distance_between(x1, y1, x2, y2):
    return math.sqrt(
        (x2 - x1) ** 2
        + (y2 - y1) ** 2
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

    rng = random.Random(200)

    visual_cells = []

    tumor_center = pygame.Vector2(
        570,
        420,
    )

    tumor_radius = 245

    for cell in cells:

        radius = rng.randint(15, 21)

        placed = False

        for _ in range(1000):

            # Cancer cells are more likely to appear
            # inside the simulated tumor region.
            if cell.actual_cancer:

                angle = rng.uniform(
                    0,
                    math.tau,
                )

                distance = rng.uniform(
                    25,
                    tumor_radius - 30,
                )

                x = (
                    tumor_center.x
                    + math.cos(angle) * distance
                )

                y = (
                    tumor_center.y
                    + math.sin(angle) * distance
                )

            else:

                x = rng.randint(
                    SIMULATION_LEFT + 45,
                    SIMULATION_LEFT
                    + SIMULATION_WIDTH
                    - 45,
                )

                y = rng.randint(
                    SIMULATION_TOP + 100,
                    SIMULATION_TOP
                    + SIMULATION_HEIGHT
                    - 45,
                )

            collision = False

            # Keep cells away from UI labels.
            reserved_regions = [
                pygame.Rect(
                    35,
                    30,
                    390,
                    55,
                ),
                pygame.Rect(
                    640,
                    65,
                    220,
                    45,
                ),
            ]

            for region in reserved_regions:
                expanded_region = region.inflate(
                    radius * 2,
                    radius * 2,
                )

                if expanded_region.collidepoint(
                    x,
                    y,
                ):
                    collision = True
                    break

            for existing in visual_cells:

                minimum_distance = (
                    radius
                    + existing.radius
                    + 12
                )

                if distance_between(
                    x,
                    y,
                    existing.x,
                    existing.y,
                ) < minimum_distance:

                    collision = True
                    break

            if not collision:

                visual_cells.append(
                    VisualCell(
                        cell=cell,
                        x=x,
                        y=y,
                        radius=radius,
                    )
                )

                placed = True
                break

        if not placed:
            print(
                f"Warning: could not place "
                f"cell {cell.cell_id}"
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

    bar_width = 285

    pygame.draw.rect(
        surface,
        (54, 59, 68),
        (
            x,
            y + 22,
            bar_width,
            11,
        ),
        border_radius=5,
    )

    pygame.draw.rect(
        surface,
        (72, 155, 210),
        (
            x,
            y + 22,
            int(bar_width * value),
            11,
        ),
        border_radius=5,
    )

    draw_text(
        surface,
        font,
        f"{value:.2f}",
        x + 235,
        y,
    )


def draw_button(
    surface,
    font,
    rect,
    text,
    active=False,
):

    color = (
        (70, 125, 165)
        if active
        else (55, 60, 70)
    )

    pygame.draw.rect(
        surface,
        color,
        rect,
        border_radius=7,
    )

    pygame.draw.rect(
        surface,
        (100, 110, 125),
        rect,
        1,
        border_radius=7,
    )

    rendered = font.render(
        text,
        True,
        (235, 235, 240),
    )

    text_rect = rendered.get_rect(
        center=rect.center
    )

    surface.blit(
        rendered,
        text_rect,
    )


def draw_swarm_status(
    surface,
    font,
    swarm,
    x,
    y,
):
    """
    Draw live state and AI output for every
    simulated robot in the swarm.
    """

    state_colors = {
        "SEARCHING": (155, 165, 220),
        "EXPLORING": (105, 205, 185),
        "TRAVELING": (100, 175, 225),
        "SCANNING": (225, 185, 70),
        "DECIDING": (190, 135, 220),
    }

    action_colors = {
        "TARGET": (225, 85, 90),
        "PASS": (85, 205, 135),
        "UNCERTAIN": (230, 190, 70),
    }

    draw_text(
        surface,
        font,
        "SWARM AI STATUS",
        x,
        y,
        (165, 170, 180),
    )

    row_y = y + 26

    for agent in swarm.agents:

        target = "--"
        probability = "--"
        action = "--"

        if agent.current_cell is not None:
            target = (
                f"{agent.current_cell.cell.cell_id:02d}"
            )

        if agent.result is not None:
            probability = (
                f"{agent.result.cancer_probability:.1%}"
            )

        if agent.decision is not None:
            action = agent.decision.action

        state_color = state_colors.get(
            agent.state,
            (180, 180, 185),
        )

        draw_text(
            surface,
            font,
            f"R{agent.robot_id}",
            x,
            row_y,
            (225, 230, 240),
        )

        draw_text(
            surface,
            font,
            f"{agent.state:<9}",
            x + 32,
            row_y,
            state_color,
        )

        draw_text(
            surface,
            font,
            f"C:{target}",
            x + 125,
            row_y,
            (180, 185, 195),
        )

        draw_text(
            surface,
            font,
            probability,
            x + 180,
            row_y,
            (
                action_colors.get(
                    action,
                    (180, 185, 195),
                )
            ),
        )

        draw_text(
            surface,
            font,
            action,
            x + 255,
            row_y,
            (
                action_colors.get(
                    action,
                    (180, 185, 195),
                )
            ),
        )

        draw_text(
            surface,
            font,
            (
                f"{agent.visual.distance_traveled:.0f}px"
            ),
            x + 365,
            row_y,
            (145, 150, 160),
        )

        row_y += 24



def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT),
        pygame.RESIZABLE,
    )

    pygame.display.set_caption(
        "Cancer AI Microrobot Simulation"
    )

    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont(
        "Arial",
        24,
        bold=True,
    )

    font = pygame.font.SysFont(
        "Arial",
        17,
    )

    small_font = pygame.font.SysFont(
        "Arial",
        14,
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

    # All interactive controls live in the
    # dedicated footer below the information panel.
    pause_button = pygame.Rect(
        930,
        880,
        90,
        38,
    )

    restart_button = pygame.Rect(
        1030,
        880,
        90,
        38,
    )

    truth_button = pygame.Rect(
        1130,
        880,
        110,
        38,
    )

    speed_down_button = pygame.Rect(
        1300,
        880,
        42,
        38,
    )

    speed_up_button = pygame.Rect(
        1352,
        880,
        42,
        38,
    )

    def reset_simulation():

        visual_cells = create_visual_cells()

        swarm = RobotSwarm(
            robot_count=4,
            start_x=60,
            start_y=400,
            spacing=42,
            speed=155,
        )

        return {
            "cells": visual_cells,
            "swarm": swarm,
            "scanned": 0,
            "targets": 0,
            "passes": 0,
            "uncertain": 0,
            "correct_targets": 0,
            "false_targets": 0,
            "history": [],
        }

    state = reset_simulation()

    paused = False
    show_ground_truth = True
    simulation_speed = 1.0

    running = True

    while running:

        real_delta = (
            clock.tick(FPS) / 1000.0
        )

        delta_time = (
            real_delta
            * simulation_speed
        )

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    state = reset_simulation()

                elif event.key == pygame.K_g:
                    show_ground_truth = (
                        not show_ground_truth
                    )

                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_KP_PLUS,
                ):
                    simulation_speed = min(
                        3.0,
                        simulation_speed + 0.25,
                    )

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS,
                ):
                    simulation_speed = max(
                        0.25,
                        simulation_speed - 0.25,
                    )

            elif event.type == pygame.MOUSEBUTTONDOWN:

                if event.button == 1:

                    if pause_button.collidepoint(
                        event.pos
                    ):
                        paused = not paused

                    elif restart_button.collidepoint(
                        event.pos
                    ):
                        state = reset_simulation()

                    elif truth_button.collidepoint(
                        event.pos
                    ):
                        show_ground_truth = (
                            not show_ground_truth
                        )

                    elif speed_down_button.collidepoint(
                        event.pos
                    ):
                        simulation_speed = max(
                            0.25,
                            simulation_speed - 0.25,
                        )

                    elif speed_up_button.collidepoint(
                        event.pos
                    ):
                        simulation_speed = min(
                            3.0,
                            simulation_speed + 0.25,
                        )

        visual_cells = state["cells"]
        swarm = state["swarm"]

        if (
            not paused
            and state["scanned"]
            < len(visual_cells)
        ):

            for agent in swarm.agents:

                robot = agent.visual

                # SEARCHING
                if agent.current_cell is None:

                    agent.state = "SEARCHING"

                    candidate = (
                        find_nearest_available_cell(
                            robot,
                            visual_cells,
                        )
                    )

                    if (
                        candidate is not None
                        and claim_cell(
                            candidate,
                            agent.robot_id,
                        )
                    ):
                        agent.current_cell = candidate

                current_cell = agent.current_cell

                if current_cell is None:
                    continue

                # TRAVELING
                if not agent.scanning:

                    agent.state = "TRAVELING"

                    arrived = robot.move_toward(
                        current_cell.position(),
                        delta_time,
                    )

                    if arrived:

                        agent.state = "SCANNING"

                        agent.result = (
                            classifier.predict(
                                current_cell.cell
                            )
                        )

                        agent.decision = (
                            decision_engine.decide(
                                agent.result
                            )
                        )

                        current_cell.scanned = True

                        agent.scanning = True
                        agent.scan_timer = 0.0

                # SCANNING
                else:

                    agent.state = "SCANNING"

                    agent.scan_timer += delta_time

                    if agent.scan_timer >= 1.25:

                        agent.state = "DECIDING"

                        action = (
                            agent.decision.action
                        )

                        probability = (
                            agent.result
                            .cancer_probability
                        )

                        if action == "TARGET":

                            state["targets"] += 1
                            agent.targets += 1

                            current_cell.targeted = True

                            if (
                                current_cell
                                .cell
                                .actual_cancer
                            ):
                                state[
                                    "correct_targets"
                                ] += 1

                            else:
                                state[
                                    "false_targets"
                                ] += 1

                        elif action == "PASS":

                            state["passes"] += 1
                            agent.passes += 1

                        else:

                            state["uncertain"] += 1
                            agent.uncertain += 1

                            current_cell.uncertain = True

                        state["history"].insert(
                            0,
                            (
                                current_cell.cell.cell_id,
                                action,
                                probability,
                                agent.robot_id,
                            ),
                        )

                        state["history"] = (
                            state["history"][:3]
                        )

                        state["scanned"] += 1
                        agent.cells_scanned += 1

                        release_cell(
                            current_cell,
                            agent.robot_id,
                        )

                        agent.clear_target()

        screen.fill(
            (15, 18, 24)
        )

        # Tissue field
        tissue_rect = pygame.Rect(
            SIMULATION_LEFT,
            SIMULATION_TOP,
            SIMULATION_WIDTH,
            SIMULATION_HEIGHT,
        )

        pygame.draw.rect(
            screen,
            (27, 31, 39),
            tissue_rect,
            border_radius=12,
        )

        # Simulated tumor region
        tumor_surface = pygame.Surface(
            (
                SIMULATION_WIDTH,
                SIMULATION_HEIGHT,
            ),
            pygame.SRCALPHA,
        )

        pygame.draw.circle(
            tumor_surface,
            (130, 55, 70, 35),
            (
                570 - SIMULATION_LEFT,
                420 - SIMULATION_TOP,
            ),
            245,
        )

        pygame.draw.circle(
            tumor_surface,
            (170, 75, 90, 80),
            (
                570 - SIMULATION_LEFT,
                420 - SIMULATION_TOP,
            ),
            245,
            2,
        )

        screen.blit(
            tumor_surface,
            (
                SIMULATION_LEFT,
                SIMULATION_TOP,
            ),
        )

        draw_text(
            screen,
            title_font,
            "MICROSCOPIC TISSUE ENVIRONMENT",
            45,
            42,
        )

        draw_text(
            screen,
            small_font,
            "SIMULATED TUMOR REGION",
            655,
            82,
            (180, 105, 115),
        )

        for visual_cell in visual_cells:

            visual_cell.draw(
                screen,
                selected=any(
                    visual_cell
                    is agent.current_cell
                    for agent in swarm.agents
                ),
                show_ground_truth=(
                    show_ground_truth
                ),
            )

        # Draw navigation paths and sensor ranges
        # for every robot in the swarm.
        for agent in swarm.agents:

            robot = agent.visual
            current_cell = agent.current_cell

            if (
                current_cell is not None
                and not agent.scanning
            ):
                pygame.draw.line(
                    screen,
                    (70, 105, 125),
                    (
                        int(robot.position.x),
                        int(robot.position.y),
                    ),
                    (
                        int(current_cell.x),
                        int(current_cell.y),
                    ),
                    1,
                )

            robot.draw_sensor_range(
                screen
            )

        # Draw robot bodies after sensor overlays.
        for agent in swarm.agents:

            robot = agent.visual

            pulse = (
                math.sin(
                    agent.scan_timer * 7
                )
                + 1
            ) / 2

            robot.draw(
                screen,
                scanning=agent.scanning,
                pulse=pulse,
                robot_id=agent.robot_id,
                font=small_font,
            )

        # Panel
        pygame.draw.rect(
            screen,
            (29, 33, 41),
            (
                PANEL_X,
                20,
                PANEL_WIDTH,
                810,
            ),
            border_radius=12,
        )

        draw_text(
            screen,
            title_font,
            "AI MICROROBOT",
            935,
            42,
        )

        status = (
            "PAUSED"
            if paused
            else "RUNNING"
        )

        draw_text(
            screen,
            small_font,
            status,
            1325,
            48,
            (
                (225, 185, 70)
                if paused
                else (90, 205, 135)
            ),
        )

        draw_text(
            screen,
            font,
            f"Scanned: {state['scanned']} / "
            f"{len(visual_cells)}",
            935,
            85,
        )

        draw_text(
            screen,
            small_font,
            (
                f"Swarm distance: "
                f"{swarm.total_distance:.0f} px"
            ),
            935,
            108,
            (155, 165, 180),
        )

        draw_text(
            screen,
            font,
            f"TARGET      {state['targets']}",
            935,
            135,
            (220, 90, 95),
        )

        draw_text(
            screen,
            font,
            f"PASS        {state['passes']}",
            1115,
            135,
            (90, 200, 135),
        )

        draw_text(
            screen,
            font,
            f"UNCERTAIN   {state['uncertain']}",
            935,
            160,
            (225, 185, 70),
        )

        pygame.draw.line(
            screen,
            (70, 75, 85),
            (935, 180),
            (1245, 180),
        )

        active_agent = next(
            (
                agent
                for agent in swarm.agents
                if agent.current_cell is not None
            ),
            None,
        )

        if active_agent is not None:

            cell = active_agent.current_cell.cell

            draw_text(
                screen,
                title_font,
                f"CELL {cell.cell_id}",
                935,
                215,
            )

            draw_feature_bar(
                screen,
                small_font,
                "Marker A",
                cell.marker_a,
                935,
                255,
            )

            draw_feature_bar(
                screen,
                small_font,
                "Marker B",
                cell.marker_b,
                935,
                300,
            )

            draw_feature_bar(
                screen,
                small_font,
                "Irregularity",
                cell.irregularity,
                935,
                345,
            )

            draw_feature_bar(
                screen,
                small_font,
                "Growth Signal",
                cell.growth_signal,
                935,
                390,
            )

        draw_swarm_status(
            screen,
            small_font,
            swarm,
            935,
            445,
        )

        draw_text(
            screen,
            small_font,
            "RECENT DECISIONS",
            935,
            575,
            (160, 165, 175),
        )

        history_y = 602

        for (
            cell_id,
            action,
            probability,
            robot_id,
        ) in state["history"]:

            draw_text(
                screen,
                small_font,
                (
                    f"R{robot_id}  "
                    f"Cell {cell_id:02d}  "
                    f"{action:<9} "
                    f"{probability:>6.1%}"
                ),
                935,
                history_y,
            )

            history_y += 23

        draw_text(
            screen,
            small_font,
            (
                f"Correct targets: "
                f"{state['correct_targets']}"
            ),
            1245,
            602,
            (150, 155, 165),
        )

        draw_text(
            screen,
            small_font,
            (
                f"False targets: "
                f"{state['false_targets']}"
            ),
            1245,
            625,
            (150, 155, 165),
        )

        # Dedicated footer keeps controls separate
        # from simulation statistics.
        pygame.draw.rect(
            screen,
            (29, 33, 41),
            (
                20,
                850,
                WIDTH - 40,
                90,
            ),
            border_radius=12,
        )

        draw_text(
            screen,
            small_font,
            "CONTROLS",
            45,
            865,
            (160, 165, 175),
        )

        draw_text(
            screen,
            small_font,
            "SPACE Pause   R Restart   G Ground Truth   ESC Exit",
            45,
            895,
            (190, 195, 205),
        )

        draw_text(
            screen,
            small_font,
            f"SPEED  {simulation_speed:.2f}x",
            1280,
            855,
            (160, 165, 175),
        )

        draw_button(
            screen,
            font,
            speed_down_button,
            "−",
        )

        draw_button(
            screen,
            font,
            speed_up_button,
            "+",
        )

        draw_button(
            screen,
            small_font,
            pause_button,
            (
                "Resume"
                if paused
                else "Pause"
            ),
            active=paused,
        )

        draw_button(
            screen,
            small_font,
            restart_button,
            "Restart",
        )

        draw_button(
            screen,
            small_font,
            truth_button,
            (
                "Hide Truth"
                if show_ground_truth
                else "Show Truth"
            ),
            active=not show_ground_truth,
        )

        if state["scanned"] >= len(
            visual_cells
        ):

            draw_text(
                screen,
                title_font,
                "SIMULATION COMPLETE",
                350,
                735,
                (90, 205, 135),
            )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
