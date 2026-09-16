from src.ai.classifier import CancerCellClassifier
from src.models.cell import Cell


def test_classifier_probability_is_valid():

    classifier = CancerCellClassifier()

    cell = Cell(
        cell_id=1,
        marker_a=0.5,
        marker_b=0.5,
        irregularity=0.5,
        growth_signal=0.5,
    )

    result = classifier.predict(cell)

    assert 0.0 <= result.cancer_probability <= 1.0


def test_high_signal_cell_scores_higher():

    classifier = CancerCellClassifier()

    low_signal = Cell(
        cell_id=1,
        marker_a=0.1,
        marker_b=0.1,
        irregularity=0.1,
        growth_signal=0.1,
    )

    high_signal = Cell(
        cell_id=2,
        marker_a=0.9,
        marker_b=0.9,
        irregularity=0.9,
        growth_signal=0.9,
    )

    low_result = classifier.predict(low_signal)
    high_result = classifier.predict(high_signal)

    assert (
        high_result.cancer_probability
        >
        low_result.cancer_probability
    )
