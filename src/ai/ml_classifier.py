import joblib
import numpy as np

from src.ai.classifier import ClassificationResult
from src.models.cell import Cell


class MLCancerCellClassifier:
    """
    Wrapper around a trained scikit-learn classifier.

    The model receives only synthetic sensor measurements.
    Ground-truth labels are never passed into prediction.
    """

    FEATURE_NAMES = [
        "marker_a",
        "marker_b",
        "irregularity",
        "growth_signal",
    ]

    def __init__(self, model):
        self.model = model

    def predict(self, cell: Cell) -> ClassificationResult:
        """
        Predict one cell.

        Useful for individual robot interactions and the
        future visual simulation.
        """

        features = np.array(
            [cell.features()],
            dtype=float,
        )

        probability = float(
            self.model.predict_proba(features)[0][1]
        )

        return ClassificationResult(
            cancer_probability=probability,
            explanation={
                "model_probability": probability,
            },
        )

    def predict_batch(
        self,
        cells: list[Cell],
    ) -> list[ClassificationResult]:
        """
        Predict many cells in one model call.

        This is substantially faster than calling predict()
        thousands of times.
        """

        if not cells:
            return []

        features = np.array(
            [
                cell.features()
                for cell in cells
            ],
            dtype=float,
        )

        probabilities = self.model.predict_proba(
            features
        )[:, 1]

        return [
            ClassificationResult(
                cancer_probability=float(probability),
                explanation={
                    "model_probability": float(probability),
                },
            )
            for probability in probabilities
        ]

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        model = joblib.load(path)
        return cls(model)
