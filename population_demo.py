import csv

from src.ai.classifier import CancerCellClassifier
from src.ai.decision_engine import DecisionEngine
from src.metrics.evaluation import evaluate_population
from src.simulation.environment import CellEnvironment


POPULATION_SIZE = 1000
CANCER_FRACTION = 0.30
NOISE_LEVEL = 0.08

TARGET_THRESHOLD = 0.85
PASS_THRESHOLD = 0.30


def export_population(cells, classifier, decision_engine):
    output_file = "data/synthetic_cells.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "cell_id",
            "marker_a",
            "marker_b",
            "irregularity",
            "growth_signal",
            "actual_cancer",
            "cancer_probability",
            "decision",
        ])

        for cell in cells:

            classification = classifier.predict(cell)
            decision = decision_engine.decide(
                classification
            )

            writer.writerow([
                cell.cell_id,
                round(cell.marker_a, 4),
                round(cell.marker_b, 4),
                round(cell.irregularity, 4),
                round(cell.growth_signal, 4),
                cell.actual_cancer,
                round(
                    classification.cancer_probability,
                    4,
                ),
                decision.action,
            ])

    return output_file


def print_results(result):
    print()
    print("CANCER AI MICROROBOT")
    print("POPULATION SIMULATION")
    print("=" * 55)

    print()
    print("POPULATION")
    print("-" * 55)

    print(f"Total cells:             {result.total_cells}")
    print(f"Cancer cells:            {result.cancer_cells}")
    print(f"Healthy cells:           {result.healthy_cells}")

    print()
    print("ROBOT DECISIONS")
    print("-" * 55)

    print(f"TARGET:                  {result.target_decisions}")
    print(f"PASS:                    {result.pass_decisions}")
    print(f"UNCERTAIN:               {result.uncertain_decisions}")

    print()
    print("CORRECT / INCORRECT ACTIONS")
    print("-" * 55)

    print(
        f"Cancer correctly targeted: "
        f"{result.true_positives}"
    )

    print(
        f"Healthy incorrectly targeted: "
        f"{result.false_positives}"
    )

    print(
        f"Healthy correctly passed: "
        f"{result.true_negatives}"
    )

    print(
        f"Cancer incorrectly passed: "
        f"{result.false_negatives}"
    )

    print()
    print("UNCERTAIN")
    print("-" * 55)

    print(
        f"Cancer cells marked uncertain: "
        f"{result.uncertain_cancer}"
    )

    print(
        f"Healthy cells marked uncertain: "
        f"{result.uncertain_healthy}"
    )

    print()
    print("METRICS")
    print("-" * 55)

    print(
        f"Sensitivity:             "
        f"{result.sensitivity:.2%}"
    )

    print(
        f"Specificity:             "
        f"{result.specificity:.2%}"
    )

    print(
        f"False positive rate:     "
        f"{result.false_positive_rate:.2%}"
    )

    print(
        f"False negative rate:     "
        f"{result.false_negative_rate:.2%}"
    )


def main():

    environment = CellEnvironment(
        cancer_fraction=CANCER_FRACTION,
        noise_level=NOISE_LEVEL,
        seed=42,
    )

    cells = environment.generate_population(
        POPULATION_SIZE
    )

    classifier = CancerCellClassifier()

    decision_engine = DecisionEngine(
        target_threshold=TARGET_THRESHOLD,
        pass_threshold=PASS_THRESHOLD,
    )

    result = evaluate_population(
        cells,
        classifier,
        decision_engine,
    )

    print_results(result)

    output_file = export_population(
        cells,
        classifier,
        decision_engine,
    )

    print()
    print("=" * 55)
    print(
        f"Synthetic dataset exported to: "
        f"{output_file}"
    )
    print()


if __name__ == "__main__":
    main()
