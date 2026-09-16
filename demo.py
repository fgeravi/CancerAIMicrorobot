from src.models.cell import Cell
from src.models.microrobot import Microrobot
from src.ai.classifier import CancerCellClassifier
from src.ai.decision_engine import DecisionEngine


def main():

    # Synthetic cells used to test the system.
    cells = [
        Cell(
            cell_id=1,
            marker_a=0.95,
            marker_b=0.90,
            irregularity=0.92,
            growth_signal=0.88,
            actual_cancer=True,
        ),

        Cell(
            cell_id=2,
            marker_a=0.10,
            marker_b=0.15,
            irregularity=0.08,
            growth_signal=0.12,
            actual_cancer=False,
        ),

        Cell(
            cell_id=3,
            marker_a=0.65,
            marker_b=0.55,
            irregularity=0.70,
            growth_signal=0.60,
            actual_cancer=True,
        ),
    ]

    classifier = CancerCellClassifier()

    decision_engine = DecisionEngine(
        target_threshold=0.85,
        pass_threshold=0.30,
    )

    robot = Microrobot(robot_id=1)

    print()
    print("CANCER AI MICROROBOT SIMULATION")
    print("=" * 50)

    for cell in cells:

        # Robot receives simulated measurements.
        result = classifier.predict(cell)

        # Safety layer decides whether to act.
        decision = decision_engine.decide(result)

        robot.record_decision(decision.action)

        print()
        print(f"CELL {cell.cell_id}")
        print("-" * 30)

        print(f"Ground truth cancer: {cell.actual_cancer}")
        print(f"Cancer probability: {result.cancer_probability:.1%}")
        print(f"Robot decision: {decision.action}")
        print(f"Reason: {decision.reason}")

        print("Sensor contributions:")

        for feature, contribution in result.explanation.items():
            print(f"  {feature}: {contribution:.3f}")

    print()
    print("=" * 50)
    print("ROBOT SUMMARY")
    print("=" * 50)

    print(f"Cells scanned:       {robot.cells_scanned}")
    print(f"Targets selected:    {robot.targets_selected}")
    print(f"Cells passed:        {robot.cells_passed}")
    print(f"Uncertain cells:     {robot.uncertain_cells}")

    print()


if __name__ == "__main__":
    main()
