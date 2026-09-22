def find_nearest_available_cell(
    robot,
    visual_cells,
):
    """
    Find the nearest cell that has not been scanned
    and is not already claimed by another robot.

    This navigation function uses spatial position only.
    It does not use hidden cancer ground truth.
    """

    candidates = [
        cell
        for cell in visual_cells
        if (
            not cell.scanned
            and cell.claimed_by is None
        )
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda cell: (
            robot.position.distance_to(
                cell.position()
            )
        ),
    )


def claim_cell(
    cell,
    robot_id: int,
) -> bool:
    """
    Attempt to reserve a cell for one robot.
    """

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
    """
    Release a reservation only if it belongs
    to the requesting robot.
    """

    if cell.claimed_by == robot_id:
        cell.claimed_by = None
