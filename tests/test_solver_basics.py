import numpy as np

from sensorlab.solvers import LaplaceSolver

from sensorlab.physics.fields import PotentialField
from sensorlab.physics.geometry import Grid2D


# ==========================================================
# Initialization
# ==========================================================


def test_initialize_returns_potential_field():

    solver = LaplaceSolver()

    grid = Grid2D(
        xmin=-1,
        xmax=1,
        ymin=-1,
        ymax=1,
        nx=11,
        ny=21,
    )

    field = solver._initialize(grid)

    assert isinstance(field, PotentialField)


def test_initialize_preserves_grid():

    solver = LaplaceSolver()

    grid = Grid2D(
        xmin=-1,
        xmax=1,
        ymin=-1,
        ymax=1,
        nx=13,
        ny=17,
    )

    field = solver._initialize(grid)

    assert field.grid is grid


def test_initialize_creates_correct_shape():

    solver = LaplaceSolver()

    grid = Grid2D(
        xmin=-1,
        xmax=1,
        ymin=-1,
        ymax=1,
        nx=15,
        ny=25,
    )

    field = solver._initialize(grid)

    assert field.shape == (25, 15)


def test_initialize_sets_zero_potential_everywhere():

    solver = LaplaceSolver()

    grid = Grid2D(
        xmin=-1,
        xmax=1,
        ymin=-1,
        ymax=1,
        nx=7,
        ny=9,
    )

    field = solver._initialize(grid)

    assert np.all(field.values == 0.0)


# ==========================================================
# Convergence
# ==========================================================


def test_converged_returns_true_for_small_error():

    solver = LaplaceSolver(
        tolerance=1e-6,
    )

    assert solver._converged(1e-7)


def test_converged_returns_false_for_large_error():

    solver = LaplaceSolver(
        tolerance=1e-6,
    )

    assert not solver._converged(1e-3)