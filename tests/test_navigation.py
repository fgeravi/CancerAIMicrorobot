from src.models.cell import Cell
from src.simulation.navigation import (
    claim_cell,
    find_nearest_available_cell,
    release_cell,
)
from visualization.cell_scene import VisualCell
from visualization.robot_scene import VisualMicrorobot


def make_cell(
    cell_id,
    x,
    y,
):
    cell = Cell(
        cell_id=cell_id,
        marker_a=0.2,
        marker_b=0.2,
        irregularity=0.2,
        growth_signal=0.2,
    )

    return VisualCell(
        cell=cell,
        x=x,
        y=y,
    )


def test_robot_selects_nearest_available_cell():
    robot = VisualMicrorobot(
        0,
        0,
    )

    near = make_cell(
        1,
        10,
        0,
    )

    far = make_cell(
        2,
        100,
        0,
    )

    result = find_nearest_available_cell(
        robot,
        [far, near],
    )

    assert result is near


def test_claimed_cell_is_not_selected():
    robot = VisualMicrorobot(
        0,
        0,
    )

    claimed = make_cell(
        1,
        10,
        0,
    )

    available = make_cell(
        2,
        50,
        0,
    )

    claimed.claimed_by = 99

    result = find_nearest_available_cell(
        robot,
        [claimed, available],
    )

    assert result is available


def test_cell_claim_and_release():
    cell = make_cell(
        1,
        10,
        10,
    )

    assert claim_cell(
        cell,
        robot_id=1,
    )

    assert cell.claimed_by == 1

    release_cell(
        cell,
        robot_id=1,
    )

    assert cell.claimed_by is None
