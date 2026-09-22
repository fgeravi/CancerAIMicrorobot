from src.simulation.robot_agent import RobotAgent
from visualization.robot_scene import VisualMicrorobot


class RobotSwarm:
    """
    Coordinates multiple simulated autonomous microrobots.

    Robots share the same tissue environment but maintain
    independent position, targets, scan state, and statistics.
    """

    def __init__(
        self,
        robot_count: int = 4,
        start_x: float = 60.0,
        start_y: float = 400.0,
        spacing: float = 35.0,
        speed: float = 155.0,
    ):
        if robot_count <= 0:
            raise ValueError(
                "robot_count must be greater than zero."
            )

        self.agents = []

        for index in range(robot_count):

            y_offset = (
                index - (robot_count - 1) / 2
            ) * spacing

            visual = VisualMicrorobot(
                start_x,
                start_y + y_offset,
                speed=speed,
            )

            agent = RobotAgent(
                robot_id=index + 1,
                visual=visual,
            )

            self.agents.append(agent)

    @property
    def total_distance(self) -> float:
        return sum(
            agent.visual.distance_traveled
            for agent in self.agents
        )

    @property
    def cells_scanned(self) -> int:
        return sum(
            agent.cells_scanned
            for agent in self.agents
        )

    @property
    def targets(self) -> int:
        return sum(
            agent.targets
            for agent in self.agents
        )

    @property
    def passes(self) -> int:
        return sum(
            agent.passes
            for agent in self.agents
        )

    @property
    def uncertain(self) -> int:
        return sum(
            agent.uncertain
            for agent in self.agents
        )
