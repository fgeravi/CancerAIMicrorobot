from dataclasses import dataclass


@dataclass
class Microrobot:
    robot_id: int

    cells_scanned: int = 0
    targets_selected: int = 0
    cells_passed: int = 0
    uncertain_cells: int = 0

    def record_decision(self, decision: str) -> None:
        self.cells_scanned += 1

        if decision == "TARGET":
            self.targets_selected += 1

        elif decision == "PASS":
            self.cells_passed += 1

        elif decision == "UNCERTAIN":
            self.uncertain_cells += 1
