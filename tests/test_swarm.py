from src.simulation.swarm import RobotSwarm


def test_swarm_creates_requested_number_of_robots():
    swarm = RobotSwarm(robot_count=4)

    assert len(swarm.agents) == 4


def test_swarm_robot_ids_are_unique():
    swarm = RobotSwarm(robot_count=4)

    ids = [
        agent.robot_id
        for agent in swarm.agents
    ]

    assert len(ids) == len(set(ids))


def test_swarm_starts_with_zero_distance():
    swarm = RobotSwarm(robot_count=4)

    assert swarm.total_distance == 0.0


def test_swarm_starts_with_zero_scans():
    swarm = RobotSwarm(robot_count=4)

    assert swarm.cells_scanned == 0


def test_swarm_rejects_zero_robots():
    try:
        RobotSwarm(robot_count=0)
        assert False
    except ValueError:
        assert True
