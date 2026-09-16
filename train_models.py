import csv

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.ai.ml_classifier import MLCancerCellClassifier
from src.simulation.environment import CellEnvironment


TRAIN_SIZE = 10000
TEST_SIZE = 5000

RANDOM_SEED = 42


def cells_to_arrays(cells):

    X = np.array(
        [cell.features() for cell in cells],
        dtype=float,
    )

    y = np.array(
        [
            int(cell.actual_cancer)
            for cell in cells
        ],
        dtype=int,
    )

    return X, y


def evaluate_binary_model(
    name,
    model,
    X_test,
    y_test,
):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


def main():

    print()
    print("CANCER AI MICROROBOT")
    print("ML TRAINING EXPERIMENT")
    print("=" * 65)

    # Separate generators/seeds ensure that the test
    # population is not the same as the training population.
    training_environment = CellEnvironment(
        cancer_fraction=0.30,
        noise_level=0.08,
        seed=RANDOM_SEED,
    )

    test_environment = CellEnvironment(
        cancer_fraction=0.30,
        noise_level=0.08,
        seed=RANDOM_SEED + 1,
    )

    training_cells = (
        training_environment.generate_population(
            TRAIN_SIZE
        )
    )

    test_cells = (
        test_environment.generate_population(
            TEST_SIZE
        )
    )

    X_train, y_train = cells_to_arrays(
        training_cells
    )

    X_test, y_test = cells_to_arrays(
        test_cells
    )

    print()
    print(f"Training cells: {len(X_train)}")
    print(f"Testing cells:  {len(X_test)}")

    models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
    }

    results = []

    print()
    print("TRAINING MODELS")
    print("-" * 65)

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(
            X_train,
            y_train,
        )

        result = evaluate_binary_model(
            name,
            model,
            X_test,
            y_test,
        )

        results.append(result)

    print()
    print("TEST SET RESULTS")
    print("=" * 65)

    print(
        f"{'Model':<24}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'ROC AUC':<12}"
    )

    print("-" * 65)

    for result in results:

        print(
            f"{result['model']:<24}"
            f"{result['accuracy']:<12.3f}"
            f"{result['precision']:<12.3f}"
            f"{result['recall']:<12.3f}"
            f"{result['roc_auc']:<12.3f}"
        )

    # Select by ROC AUC for this simulation experiment.
    best_result = max(
        results,
        key=lambda item: item["roc_auc"],
    )

    best_name = best_result["model"]
    best_model = models[best_name]

    wrapper = MLCancerCellClassifier(
        best_model
    )

    model_path = (
        "models/cancer_classifier.joblib"
    )

    wrapper.save(model_path)

    print()
    print(
        f"Selected model: {best_name}"
    )

    print(
        f"Saved locally to: {model_path}"
    )

    output_file = (
        "results/model_comparison.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "accuracy",
                "precision",
                "recall",
                "roc_auc",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"Results saved to: {output_file}"
    )

    print()


if __name__ == "__main__":
    main()
