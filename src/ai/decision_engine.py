from dataclasses import dataclass

from src.ai.classifier import ClassificationResult


@dataclass
class Decision:
    action: str
    probability: float
    reason: str


class DecisionEngine:
    """
    Converts classifier output into a simulated action.

    TARGET:
        Simulated therapeutic action.

    PASS:
        Leave the cell alone.

    UNCERTAIN:
        Do not act because confidence is insufficient.
    """

    def __init__(
        self,
        target_threshold: float = 0.85,
        pass_threshold: float = 0.30,
    ):
        if not 0 <= pass_threshold < target_threshold <= 1:
            raise ValueError(
                "Thresholds must satisfy 0 <= pass < target <= 1."
            )

        self.target_threshold = target_threshold
        self.pass_threshold = pass_threshold

    def decide(self, result: ClassificationResult) -> Decision:
        probability = result.cancer_probability

        if probability >= self.target_threshold:
            return Decision(
                action="TARGET",
                probability=probability,
                reason="Cancer probability exceeded target threshold.",
            )

        if probability <= self.pass_threshold:
            return Decision(
                action="PASS",
                probability=probability,
                reason="Cancer probability was below pass threshold.",
            )

        return Decision(
            action="UNCERTAIN",
            probability=probability,
            reason="Evidence was insufficient for a simulated action.",
        )
