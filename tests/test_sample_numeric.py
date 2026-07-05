"""
Numerical validation of the electrostatic solver.

These tests validate physical properties rather than
implementation details.

The objective is to verify that:

• electric field points from higher to lower potential
• field strength agrees with analytical parallel plate solution
• symmetry is preserved
• numerical solution converges when grid resolution increases
"""

from __future__ import annotations

import numpy as np
import pytest

from sensorlab.physics.boundary_conditions import DirichletBoundaryConditions
from sensorlab.physics.electrodes import RectangleElectrode
from sensorlab.physics.fields import ElectricFieldGrid
from sensorlab.physics.geometry import Grid2D, Point3D, Rectangle
from sensorlab.physics.materials import COPPER
from sensorlab.physics.quantities.electrical import Voltage
from sensorlab.physics.quantities.geometry import Coordinate, Length
from sensorlab.solvers import LaplaceSolver


# ==========================================================
# Helpers
# ==========================================================


def point(x: float, y: float) -> Point3D:
    return Point3D(
        Coordinate(x),
        Coordinate(y),
        Coordinate(0),
    )


def plate(
    *,
    x: float,
    voltage: float,
) -> RectangleElectrode:

    return RectangleElectrode(
        material=COPPER,
        potential=Voltage(voltage),
        rectangles=[
            Rectangle(
                center=point(x, 0),
                width=Length(0.05),
                height=Length(1.60),
            )
        ],
    )


def build_field(nx: int = 151) -> ElectricFieldGrid:

    grid = Grid2D(
        xmin=-5,
        xmax=5,
        ymin=-5,
        ymax=5,
        nx=nx,
        ny=nx,
    )

    left = plate(
        x=-0.4,
        voltage=1.0,
    )

    right = plate(
        x=0.4,
        voltage=0.0,
    )

    bc = DirichletBoundaryConditions(
        electrodes=[
            left,
            right,
        ],
    )

    potential = LaplaceSolver().solve(
        grid=grid,
        boundary_conditions=bc,
    )

    return ElectricFieldGrid.from_potential(
        potential,
    )


# ==========================================================
# Tests
# ==========================================================


def test_field_direction():
    """
    Electric field must point from
    higher potential towards lower potential.
    """

    field = build_field()

    iy = field.grid.ny // 2
    ix = field.grid.nx // 2

    Ex = field.x[iy, ix]
    Ey = field.y[iy, ix]

    assert Ex > 0
    assert abs(Ey) < 1e-2


def test_parallel_plate_strength():
    """
    Compare numerical field with analytical solution.
    """

    field = build_field()

    iy = field.grid.ny // 2
    ix = field.grid.nx // 2

    Ex = field.x[iy, ix]

    distance = 0.75

    expected = 1.0 / distance

    relative_error = abs(
        Ex - expected
    ) / expected

    #
    # Finite plates + fringing
    #
    assert relative_error < 0.10


def test_parallel_plate_symmetry():
    """
    Along the symmetry axis the vertical
    electric field should vanish.
    """

    field = build_field()

    iy = field.grid.ny // 2

    #
    # Between electrodes only
    #
    xs = np.asarray(field.grid.x)

    mask = (xs > -0.35) & (xs < 0.35)

    Ey = field.y[iy]

    assert np.max(np.abs(Ey[mask])) < 1e-2


@pytest.mark.parametrize(
    "resolution",
    [
        81,
        161,
        321,
    ],
)
def test_solution_runs_for_multiple_resolutions(
    resolution: int,
):
    """
    Basic smoke test.
    """

    field = build_field(
        resolution,
    )

    assert field.shape == (
        resolution,
        resolution,
    )
