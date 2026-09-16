import random

from src.models.cell import Cell


class CellEnvironment:
    """
    Generates synthetic cell populations for the simulation.

    These distributions are intentionally fictional and are NOT based on
    clinically validated biomarkers.
    """

    def __init__(
        self,
        cancer_fraction: float = 0.30,
        noise_level: float = 0.08,
        seed: int | None = 42,
    ):
        if not 0.0 <= cancer_fraction <= 1.0:
            raise ValueError("cancer_fraction must be between 0 and 1.")

        if noise_level < 0.0:
            raise ValueError("noise_level cannot be negative.")

        self.cancer_fraction = cancer_fraction
        self.noise_level = noise_level
        self.random = random.Random(seed)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _measurement(self, mean: float, spread: float) -> float:
        biological_variation = self.random.gauss(mean, spread)
        sensor_noise = self.random.gauss(0.0, self.noise_level)

        return self._clamp(
            biological_variation + sensor_noise
        )

    def generate_cell(
        self,
        cell_id: int,
        actual_cancer: bool | None = None,
    ) -> Cell:

        if actual_cancer is None:
            actual_cancer = (
                self.random.random() < self.cancer_fraction
            )

        if actual_cancer:
            # Synthetic cancer-like distribution.
            # Deliberately overlaps with the healthy distribution.
            means = {
                "marker_a": 0.78,
                "marker_b": 0.72,
                "irregularity": 0.76,
                "growth_signal": 0.74,
            }

            spread = 0.16

        else:
            # Synthetic healthy-like distribution.
            # Some healthy cells will appear suspicious.
            means = {
                "marker_a": 0.24,
                "marker_b": 0.28,
                "irregularity": 0.22,
                "growth_signal": 0.26,
            }

            spread = 0.16

        return Cell(
            cell_id=cell_id,
            marker_a=self._measurement(
                means["marker_a"],
                spread,
            ),
            marker_b=self._measurement(
                means["marker_b"],
                spread,
            ),
            irregularity=self._measurement(
                means["irregularity"],
                spread,
            ),
            growth_signal=self._measurement(
                means["growth_signal"],
                spread,
            ),
            actual_cancer=actual_cancer,
        )

    def generate_population(
        self,
        population_size: int,
    ) -> list[Cell]:

        if population_size <= 0:
            raise ValueError(
                "population_size must be greater than zero."
            )

        return [
            self.generate_cell(cell_id=i)
            for i in range(1, population_size + 1)
        ]
