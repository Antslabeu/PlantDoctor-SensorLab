import numpy as np

from sensorlab.solvers import LaplaceSolver

from sensorlab.physics.fields import PotentialField

from sensorlab.physics.geometry import Grid2D

from sensorlab.physics.boundary_conditions import (
    DirichletBoundaryConditions,
)

from sensorlab.physics.electrodes import RectangleElectrode

from sensorlab.physics.materials import COPPER

from sensorlab.physics.quantities.electrical import Voltage
from sensorlab.physics.quantities.geometry import (
    Coordinate,
    Length,
)

from sensorlab.physics.geometry import (
    Point3D,
    Rectangle,
)


def point(x, y):

    return Point3D(
        Coordinate(x),
        Coordinate(y),
        Coordinate(0),
    )


def electrode(v, x):

    return RectangleElectrode(
        material=COPPER,
        potential=Voltage(v),
        rectangles=[
            Rectangle(
                center=point(x, 0),
                width=Length(0.1),
                height=Length(2.0),
            )
        ],
    )


def make_grid():

    return Grid2D(
        xmin=-1,
        xmax=1,
        ymin=-1,
        ymax=1,
        nx=41,
        ny=41,
    )


def make_bc():

    return DirichletBoundaryConditions(
        electrodes=[
            electrode(1.0, -0.9),
            electrode(0.0, +0.9),
        ]
    )


# ==========================================================
# Initialization
# ==========================================================


def test_initialize():

    solver = LaplaceSolver()

    field = solver._initialize(
        make_grid(),
    )

    assert isinstance(
        field,
        PotentialField,
    )

    assert np.allclose(
        field.values,
        0.0,
    )


def test_initialize_shape():

    solver = LaplaceSolver()

    field = solver._initialize(
        make_grid(),
    )

    assert field.values.shape == (
        41,
        41,
    )


# ==========================================================
# Boundary conditions
# ==========================================================


def test_apply_boundary_conditions_creates_fixed_nodes():

    solver = LaplaceSolver()

    solver._potential = solver._initialize(
        make_grid(),
    )

    solver._apply_boundary_conditions(
        make_bc(),
    )

    assert solver._fixed_nodes.shape == (
        41,
        41,
    )

    assert solver._fixed_nodes.dtype == bool


# ==========================================================
# Convergence
# ==========================================================


def test_converged_true():

    solver = LaplaceSolver(
        tolerance=1e-6,
    )

    assert solver._converged(
        1e-7,
    )


def test_converged_false():

    solver = LaplaceSolver(
        tolerance=1e-6,
    )

    assert not solver._converged(
        1e-4,
    )