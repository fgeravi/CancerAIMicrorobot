from dataclasses import dataclass


@dataclass
class Cell:
    """
    Represents a simulated biological cell.

    All sensor values are synthetic normalized measurements from 0.0 to 1.0.
    They are NOT clinically validated cancer biomarkers.
    """

    cell_id: int

    marker_a: float
    marker_b: float
    irregularity: float
    growth_signal: float

    # Ground truth for evaluating the simulation.
    # The classifier does NOT use this value.
    actual_cancer: bool = False

    def features(self) -> list[float]:
        return [
            self.marker_a,
            self.marker_b,
            self.irregularity,
            self.growth_signal,
        ]
