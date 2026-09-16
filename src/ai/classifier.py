from dataclasses import dataclass

from src.models.cell import Cell


@dataclass
class ClassificationResult:
    cancer_probability: float
    explanation: dict[str, float]


class CancerCellClassifier:
    """
    Transparent baseline classifier.

    IMPORTANT:
    These weights are arbitrary simulation parameters.
    They are NOT medically validated.
    """

    WEIGHTS = {
        "marker_a": 0.35,
        "marker_b": 0.25,
        "irregularity": 0.20,
        "growth_signal": 0.20,
    }

    def predict(self, cell: Cell) -> ClassificationResult:
        contributions = {
            "marker_a": cell.marker_a * self.WEIGHTS["marker_a"],
            "marker_b": cell.marker_b * self.WEIGHTS["marker_b"],
            "irregularity": cell.irregularity * self.WEIGHTS["irregularity"],
            "growth_signal": cell.growth_signal * self.WEIGHTS["growth_signal"],
        }

        probability = sum(contributions.values())
        probability = max(0.0, min(1.0, probability))

        return ClassificationResult(
            cancer_probability=probability,
            explanation=contributions,
        )
