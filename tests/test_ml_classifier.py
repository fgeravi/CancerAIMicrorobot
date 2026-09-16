import numpy as np

from sklearn.linear_model import LogisticRegression

from src.ai.ml_classifier import MLCancerCellClassifier
from src.models.cell import Cell


def create_test_model():

    X = np.array([
        [0.1, 0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2],
        [0.8, 0.8, 0.8, 0.8],
        [0.9, 0.9, 0.9, 0.9],
    ])

    y = np.array([
        0,
        0,
        1,
        1,
    ])

    model = LogisticRegression()
    model.fit(X, y)

    return model


def test_ml_probability_valid():

    model = create_test_model()

    classifier = MLCancerCellClassifier(
        model
    )

    cell = Cell(
        cell_id=1,
        marker_a=0.8,
        marker_b=0.8,
        irregularity=0.8,
        growth_signal=0.8,
    )

    result = classifier.predict(cell)

    assert (
        0.0
        <= result.cancer_probability
        <= 1.0
    )


def test_ml_classifier_does_not_need_ground_truth():

    model = create_test_model()

    classifier = MLCancerCellClassifier(
        model
    )

    cell_a = Cell(
        cell_id=1,
        marker_a=0.8,
        marker_b=0.8,
        irregularity=0.8,
        growth_signal=0.8,
        actual_cancer=True,
    )

    cell_b = Cell(
        cell_id=2,
        marker_a=0.8,
        marker_b=0.8,
        irregularity=0.8,
        growth_signal=0.8,
        actual_cancer=False,
    )

    probability_a = (
        classifier.predict(
            cell_a
        ).cancer_probability
    )

    probability_b = (
        classifier.predict(
            cell_b
        ).cancer_probability
    )

    assert probability_a == probability_b
