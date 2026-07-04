import pytest

from sensorlab.physics.boundary_conditions import (
    DirichletBoundaryConditions,
)

from sensorlab.physics.electrodes import (
    RectangleElectrode,
)

from sensorlab.physics.geometry import (
    Point3D,
    Rectangle,
)

from sensorlab.physics.materials import COPPER

from sensorlab.physics.quantities.electrical import Voltage
from sensorlab.physics.quantities.geometry import (
    Coordinate,
    Length,
)


# ==========================================================
# Helpers
# ==========================================================


def point(
    x: float,
    y: float,
) -> Point3D:

    return Point3D(
        Coordinate(x),
        Coordinate(y),
        Coordinate(0),
    )


def rectangle(
    x: float,
    y: float,
    width: float = 1,
    height: float = 1,
) -> Rectangle:

    return Rectangle(
        center=point(x, y),
        width=Length(width),
        height=Length(height),
    )


def electrode(
    voltage: float,
    x: float,
) -> RectangleElectrode:

    return RectangleElectrode(
        material=COPPER,
        potential=Voltage(voltage),
        rectangles=[
            rectangle(x, 0),
        ],
    )


# ==========================================================
# Construction
# ==========================================================


def test_single_electrode():

    left = electrode(1.0, -1)

    bc = DirichletBoundaryConditions(
        electrodes=[left],
    )

    assert len(bc) == 1


def test_two_electrodes():

    left = electrode(1.0, -1)
    right = electrode(0.0, 1)

    bc = DirichletBoundaryConditions(
        electrodes=[
            left,
            right,
        ],
    )

    assert len(bc) == 2


def test_empty_boundary_conditions_are_rejected():

    with pytest.raises(ValueError):

        DirichletBoundaryConditions(
            electrodes=[],
        )


def test_duplicate_electrode_is_rejected():

    left = electrode(1.0, -1)

    with pytest.raises(ValueError):

        DirichletBoundaryConditions(
            electrodes=[
                left,
                left,
            ],
        )


# ==========================================================
# Iteration
# ==========================================================


def test_iteration():

    left = electrode(1.0, -1)
    right = electrode(0.0, 1)

    bc = DirichletBoundaryConditions(
        electrodes=[
            left,
            right,
        ],
    )

    electrodes = list(bc)

    assert electrodes[0] is left
    assert electrodes[1] is right


def test_indexing():

    left = electrode(1.0, -1)
    right = electrode(0.0, 1)

    bc = DirichletBoundaryConditions(
        electrodes=[
            left,
            right,
        ],
    )

    assert bc[0] is left
    assert bc[1] is right


# ==========================================================
# Potentials
# ==========================================================


def test_potentials_property():

    left = electrode(5.0, -1)
    right = electrode(0.0, 1)

    bc = DirichletBoundaryConditions(
        electrodes=[
            left,
            right,
        ],
    )

    assert bc.potentials == [
        Voltage(5.0),
        Voltage(0.0),
    ]


def test_potential_of_left():

    left = electrode(3.3, -1)
    right = electrode(0.0, 1)

    bc = DirichletBoundaryConditions(
        electrodes=[
            left,
            right,
        ],
    )

    assert bc.potential_of(left) == Voltage(3.3)


def test_potential_of_right():

    left = electrode(3.3, -1)
    right = electrode(-1.2, 1)

    bc = DirichletBoundaryConditions(
        electrodes=[
            left,
            right,
        ],
    )

    assert bc.potential_of(right) == Voltage(-1.2)


def test_unknown_electrode_is_rejected():

    left = electrode(1.0, -1)
    right = electrode(0.0, 1)

    bc = DirichletBoundaryConditions(
        electrodes=[
            left,
        ],
    )

    with pytest.raises(ValueError):

        bc.potential_of(right)


# ==========================================================
# Real capacitor
# ==========================================================


def test_parallel_plate_capacitor():

    left = RectangleElectrode(
        material=COPPER,
        potential=Voltage(1.0),
        rectangles=[
            rectangle(-1, 0, 0.1, 10),
        ],
    )

    right = RectangleElectrode(
        material=COPPER,
        potential=Voltage(0.0),
        rectangles=[
            rectangle(1, 0, 0.1, 10),
        ],
    )

    bc = DirichletBoundaryConditions(
        electrodes=[
            left,
            right,
        ],
    )

    assert len(bc) == 2

    assert bc.potential_of(left) == Voltage(1.0)
    assert bc.potential_of(right) == Voltage(0.0)
    assert bc.voltage_span == Voltage(1.0)