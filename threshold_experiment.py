from src.ai.classifier import CancerCellClassifier
from src.ai.decision_engine import DecisionEngine
from src.metrics.evaluation import evaluate_population
from src.simulation.environment import CellEnvironment


def main():

    environment = CellEnvironment(
        cancer_fraction=0.30,
        noise_level=0.08,
        seed=42,
    )

    cells = environment.generate_population(5000)

    classifier = CancerCellClassifier()

    thresholds = [
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    print()
    print("TARGET THRESHOLD EXPERIMENT")
    print("=" * 78)

    print(
        f"{'Threshold':<12}"
        f"{'Targeted':<12}"
        f"{'Cancer Hit':<14}"
        f"{'Healthy Hit':<14}"
        f"{'Uncertain':<12}"
        f"{'FP Rate':<10}"
    )

    print("-" * 78)

    for threshold in thresholds:

        engine = DecisionEngine(
            target_threshold=threshold,
            pass_threshold=0.30,
        )

        result = evaluate_population(
            cells,
            classifier,
            engine,
        )

        print(
            f"{threshold:<12.2f}"
            f"{result.target_decisions:<12}"
            f"{result.true_positives:<14}"
            f"{result.false_positives:<14}"
            f"{result.uncertain_decisions:<12}"
            f"{result.false_positive_rate:<10.2%}"
        )

    print()
    print(
        "Higher target thresholds require stronger "
        "classifier evidence before simulated targeting."
    )
    print()


if __name__ == "__main__":
    main()
