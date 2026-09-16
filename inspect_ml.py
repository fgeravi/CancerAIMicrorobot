from src.ai.decision_engine import DecisionEngine
from src.ai.ml_classifier import MLCancerCellClassifier
from src.simulation.environment import CellEnvironment


classifier = MLCancerCellClassifier.load(
    "models/cancer_classifier.joblib"
)

engine = DecisionEngine(
    target_threshold=0.85,
    pass_threshold=0.30,
)

environment = CellEnvironment(
    cancer_fraction=0.30,
    noise_level=0.08,
    seed=999,
)

cells = environment.generate_population(10)


print()
print("INDIVIDUAL AI DECISIONS")
print("=" * 70)


for cell in cells:

    result = classifier.predict(cell)
    decision = engine.decide(result)

    print()
    print(f"Cell {cell.cell_id}")

    print(
        f"Measurements: "
        f"{[round(x, 3) for x in cell.features()]}"
    )

    print(
        f"AI probability: "
        f"{result.cancer_probability:.2%}"
    )

    print(
        f"Decision: "
        f"{decision.action}"
    )

    print(
        f"Hidden ground truth: "
        f"{'CANCER' if cell.actual_cancer else 'HEALTHY'}"
    )


print()
