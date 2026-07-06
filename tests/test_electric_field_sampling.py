from __future__ import annotations

import numpy as np
import pytest

from sensorlab.physics.fields import ElectricFieldGrid
from sensorlab.physics.geometry import Grid2D, Point3D
from sensorlab.physics.quantities.geometry import Coordinate


# ============================================================
# Helpers
# ============================================================


def point(
    x: float,
    y: float,
) -> Point3D:

    return Point3D(
        Coordinate(x),
        Coordinate(y),
        Coordinate(0.0),
    )


def create_uniform_field(
    ex: float = 3.0,
    ey: float = -2.0,
) -> ElectricFieldGrid:

    grid = Grid2D(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        nx=11,
        ny=11,
    )

    x = np.full(
        (grid.ny, grid.nx),
        ex,
    )

    y = np.full(
        (grid.ny, grid.nx),
        ey,
    )

    return ElectricFieldGrid(
        grid=grid,
        x=x,
        y=y,
    )


# ============================================================
# Tests
# ============================================================


def test_sample_returns_exact_node_value():
    """
    Sampling exactly at a grid node should return
    the stored value.
    """

    field = create_uniform_field()

    sampled = field.sample(
        point(0.5, 0.5),
    )

    assert sampled.x == pytest.approx(3.0)
    assert sampled.y == pytest.approx(-2.0)


def test_uniform_field_is_exact_everywhere():
    """
    Bilinear interpolation must preserve
    a constant field exactly.
    """

    field = create_uniform_field()

    coordinates = (
        (0.13, 0.27),
        (0.21, 0.81),
        (0.99, 0.41),
        (0.44, 0.72),
        (0.05, 0.95),
    )

    for x, y in coordinates:

        sampled = field.sample(
            point(x, y),
        )

        assert sampled.x == pytest.approx(3.0)
        assert sampled.y == pytest.approx(-2.0)


def test_linear_field_is_exact():
    """
    Bilinear interpolation reproduces
    linear functions exactly.
    """

    grid = Grid2D(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        nx=21,
        ny=21,
    )

    X, Y = np.meshgrid(
        grid.x,
        grid.y,
    )

    #
    # Linear field
    #

    ex = 2.0 * X + 3.0 * Y
    ey = -X + 4.0 * Y

    field = ElectricFieldGrid(
        grid=grid,
        x=ex,
        y=ey,
    )

    p = point(
        0.37,
        0.64,
    )

    sampled = field.sample(p)

    expected_x = (
        2.0 * p.x.value
        + 3.0 * p.y.value
    )

    expected_y = (
        -p.x.value
        + 4.0 * p.y.value
    )

    assert sampled.x == pytest.approx(expected_x)
    assert sampled.y == pytest.approx(expected_y)


@pytest.mark.parametrize(
    "x,y",
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (1.0, 0.0),
    ],
)
def test_sampling_on_boundary_does_not_fail(
    x,
    y,
):
    """
    Sampling on the domain boundary
    should never raise an exception.
    """

    field = create_uniform_field()

    sampled = field.sample(
        point(x, y),
    )

    assert sampled.x == pytest.approx(3.0)
    assert sampled.y == pytest.approx(-2.0)


def test_sampling_outside_grid_is_clamped():
    """
    Points outside the computational domain
    should be clamped to the nearest cell.
    """

    field = create_uniform_field()

    sampled = field.sample(
        point(
            -100,
            100,
        )
    )

    assert sampled.x == pytest.approx(3.0)
    assert sampled.y == pytest.approx(-2.0)