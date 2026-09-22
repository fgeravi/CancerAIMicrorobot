def find_nearest_available_cell(
    robot,
    visual_cells,
    sensor_limited: bool = False,
):
    """
    Find the nearest available cell.

    When sensor_limited is True, only cells inside
    the robot's simulated sensor range are visible.

    Navigation never uses hidden cancer ground truth.
    """

    candidates = []

    for cell in visual_cells:

        if cell.scanned:
            continue

        if cell.claimed_by is not None:
            continue

        distance = (
            robot.position.distance_to(
                cell.position()
            )
        )

        if (
            sensor_limited
            and distance > robot.sensor_range
        ):
            continue

        candidates.append(
            (distance, cell)
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0][1]


def cells_in_sensor_range(
    robot,
    visual_cells,
):
    """
    Return currently detectable, unscanned cells.
    """

    return [
        cell
        for cell in visual_cells
        if (
            not cell.scanned
            and cell.claimed_by is None
            and robot.position.distance_to(
                cell.position()
            )
            <= robot.sensor_range
        )
    ]


def claim_cell(
    cell,
    robot_id: int,
) -> bool:

    if cell.scanned:
        return False

    if cell.claimed_by is not None:
        return False

    cell.claimed_by = robot_id

    return True


def release_cell(
    cell,
    robot_id: int,
) -> None:

    if cell.claimed_by == robot_id:
        cell.claimed_by = None
