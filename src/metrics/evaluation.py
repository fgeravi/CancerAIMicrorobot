from dataclasses import dataclass

from src.ai.classifier import CancerCellClassifier
from src.ai.decision_engine import DecisionEngine
from src.models.cell import Cell


@dataclass
class EvaluationResult:
    total_cells: int

    cancer_cells: int
    healthy_cells: int

    target_decisions: int
    pass_decisions: int
    uncertain_decisions: int

    true_positives: int
    false_positives: int

    true_negatives: int
    false_negatives: int

    uncertain_cancer: int
    uncertain_healthy: int

    sensitivity: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_population(
    cells: list[Cell],
    classifier: CancerCellClassifier,
    decision_engine: DecisionEngine,
) -> EvaluationResult:

    true_positives = 0
    false_positives = 0

    true_negatives = 0
    false_negatives = 0

    uncertain_cancer = 0
    uncertain_healthy = 0

    target_decisions = 0
    pass_decisions = 0
    uncertain_decisions = 0

    cancer_cells = 0
    healthy_cells = 0

    for cell in cells:

        if cell.actual_cancer:
            cancer_cells += 1
        else:
            healthy_cells += 1

        classification = classifier.predict(cell)
        decision = decision_engine.decide(classification)

        if decision.action == "TARGET":
            target_decisions += 1

            if cell.actual_cancer:
                true_positives += 1
            else:
                false_positives += 1

        elif decision.action == "PASS":
            pass_decisions += 1

            if cell.actual_cancer:
                false_negatives += 1
            else:
                true_negatives += 1

        else:
            uncertain_decisions += 1

            if cell.actual_cancer:
                uncertain_cancer += 1
            else:
                uncertain_healthy += 1

    sensitivity = safe_divide(
        true_positives,
        cancer_cells,
    )

    specificity = safe_divide(
        true_negatives,
        healthy_cells,
    )

    false_positive_rate = safe_divide(
        false_positives,
        healthy_cells,
    )

    false_negative_rate = safe_divide(
        false_negatives,
        cancer_cells,
    )

    return EvaluationResult(
        total_cells=len(cells),

        cancer_cells=cancer_cells,
        healthy_cells=healthy_cells,

        target_decisions=target_decisions,
        pass_decisions=pass_decisions,
        uncertain_decisions=uncertain_decisions,

        true_positives=true_positives,
        false_positives=false_positives,

        true_negatives=true_negatives,
        false_negatives=false_negatives,

        uncertain_cancer=uncertain_cancer,
        uncertain_healthy=uncertain_healthy,

        sensitivity=sensitivity,
        specificity=specificity,

        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
    )
