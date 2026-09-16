from src.simulation.environment import CellEnvironment


def test_population_size():

    environment = CellEnvironment(seed=42)

    cells = environment.generate_population(100)

    assert len(cells) == 100


def test_measurements_stay_normalized():

    environment = CellEnvironment(
        noise_level=0.50,
        seed=42,
    )

    cells = environment.generate_population(500)

    for cell in cells:

        for feature in cell.features():

            assert 0.0 <= feature <= 1.0


def test_reproducible_seed():

    environment_a = CellEnvironment(seed=42)
    environment_b = CellEnvironment(seed=42)

    cells_a = environment_a.generate_population(10)
    cells_b = environment_b.generate_population(10)

    assert cells_a == cells_b
