from src.ai.decision_engine import DecisionEngine
from src.ai.ml_classifier import MLCancerCellClassifier
from src.metrics.evaluation import evaluate_population
from src.simulation.environment import CellEnvironment


MODEL_PATH = "models/cancer_classifier.joblib"


def main():

    print()
    print("TRAINED AI MICROROBOT SIMULATION")
    print("=" * 55)

    classifier = MLCancerCellClassifier.load(
        MODEL_PATH
    )

    environment = CellEnvironment(
        cancer_fraction=0.30,
        noise_level=0.08,
        seed=100,
    )

    cells = environment.generate_population(
        5000
    )

    decision_engine = DecisionEngine(
        target_threshold=0.85,
        pass_threshold=0.30,
    )

    results = evaluate_population(
        cells,
        classifier,
        decision_engine,
    )

    print()
    print("NEW SIMULATED POPULATION")
    print("-" * 55)

    print(
        f"Total cells:                 "
        f"{results.total_cells}"
    )

    print(
        f"Cancer cells:                "
        f"{results.cancer_cells}"
    )

    print(
        f"Healthy cells:               "
        f"{results.healthy_cells}"
    )

    print()
    print("AI ROBOT DECISIONS")
    print("-" * 55)

    print(
        f"TARGET:                      "
        f"{results.target_decisions}"
    )

    print(
        f"PASS:                        "
        f"{results.pass_decisions}"
    )

    print(
        f"UNCERTAIN:                   "
        f"{results.uncertain_decisions}"
    )

    print()
    print("SAFETY / DETECTION")
    print("-" * 55)

    print(
        f"Cancer correctly targeted:   "
        f"{results.true_positives}"
    )

    print(
        f"Healthy incorrectly targeted:"
        f" {results.false_positives}"
    )

    print(
        f"Cancer incorrectly passed:   "
        f"{results.false_negatives}"
    )

    print(
        f"Cancer marked uncertain:     "
        f"{results.uncertain_cancer}"
    )

    print()
    print(
        f"Sensitivity:                 "
        f"{results.sensitivity:.2%}"
    )

    print(
        f"Specificity:                 "
        f"{results.specificity:.2%}"
    )

    print(
        f"False positive rate:         "
        f"{results.false_positive_rate:.2%}"
    )

    print()


if __name__ == "__main__":
    main()
