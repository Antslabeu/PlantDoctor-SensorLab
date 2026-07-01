import math

import pytest

from sensorlab.constants import COULOMB_CONSTANT
from sensorlab.physics.electrostatics import PointCharge, electric_potential, electric_field
from sensorlab.physics.geometry import Point3D
from sensorlab.physics.quantities.electrical import (
    Charge,
    Potential,
)
from sensorlab.physics.quantities.geometry import Coordinate


def point_charge(
    q: float,
    x: float,
    y: float = 0.0,
    z: float = 0.0,
) -> PointCharge:
    return PointCharge(
        charge=Charge(q),
        position=Point3D(
            Coordinate(x),
            Coordinate(y),
            Coordinate(z),
        ),
    )


def test_positive_charge_produces_positive_potential():

    source = point_charge(1e-6, 0)

    potential = electric_potential(
        source,
        Point3D(
            Coordinate(1),
            Coordinate(0),
            Coordinate(0),
        ),
    )

    assert potential.value > 0


def test_negative_charge_produces_negative_potential():

    source = point_charge(-1e-6, 0)

    potential = electric_potential(
        source,
        Point3D(
            Coordinate(1),
            Coordinate(0),
            Coordinate(0),
        ),
    )

    assert potential.value < 0


def test_zero_charge_produces_zero_potential():

    source = point_charge(0.0, 0)

    potential = electric_potential(
        source,
        Point3D(
            Coordinate(2),
            Coordinate(0),
            Coordinate(0),
        ),
    )

    assert potential == Potential(0.0)


def test_inverse_distance_law():

    source = point_charge(1e-6, 0)

    v1 = electric_potential(
        source,
        Point3D(
            Coordinate(1),
            Coordinate(0),
            Coordinate(0),
        ),
    )

    v2 = electric_potential(
        source,
        Point3D(
            Coordinate(2),
            Coordinate(0),
            Coordinate(0),
        ),
    )

    assert math.isclose(
        v1.value / v2.value,
        2.0,
        rel_tol=1e-12,
    )


def test_linearity_with_charge():

    point = Point3D(
        Coordinate(1),
        Coordinate(0),
        Coordinate(0),
    )

    v1 = electric_potential(
        point_charge(1e-6, 0),
        point,
    )

    v2 = electric_potential(
        point_charge(2e-6, 0),
        point,
    )

    assert math.isclose(
        v2.value,
        2 * v1.value,
        rel_tol=1e-12,
    )


def test_translation_invariance():

    q1 = point_charge(1e-6, 0)

    q2 = point_charge(
        1e-6,
        100,
        50,
        -20,
    )

    p1 = Point3D(
        Coordinate(1),
        Coordinate(0),
        Coordinate(0),
    )

    p2 = Point3D(
        Coordinate(101),
        Coordinate(50),
        Coordinate(-20),
    )

    v1 = electric_potential(q1, p1)
    v2 = electric_potential(q2, p2)

    assert math.isclose(
        v1.value,
        v2.value,
        rel_tol=1e-12,
    )


def test_potential_depends_only_on_distance():

    source = point_charge(1e-6, 0)

    v1 = electric_potential(
        source,
        Point3D(
            Coordinate(1),
            Coordinate(0),
            Coordinate(0),
        ),
    )

    v2 = electric_potential(
        source,
        Point3D(
            Coordinate(0),
            Coordinate(1),
            Coordinate(0),
        ),
    )

    v3 = electric_potential(
        source,
        Point3D(
            Coordinate(0),
            Coordinate(0),
            Coordinate(1),
        ),
    )

    assert math.isclose(
        v1.value,
        v2.value,
        rel_tol=1e-12,
    )

    assert math.isclose(
        v2.value,
        v3.value,
        rel_tol=1e-12,
    )


def test_matches_analytical_solution():

    charge = 3e-6
    distance = 2.5

    source = point_charge(
        charge,
        0,
    )

    potential = electric_potential(
        source,
        Point3D(
            Coordinate(distance),
            Coordinate(0),
            Coordinate(0),
        ),
    )

    expected = (
        COULOMB_CONSTANT
        * charge
        / distance
    )

    assert math.isclose(
        potential.value,
        expected,
        rel_tol=1e-12,
    )


def test_same_position_raises():

    source = point_charge(
        1e-6,
        0,
    )

    with pytest.raises(ValueError):
        electric_potential(
            source,
            source.position,
        )


def test_potential_is_scalar():

    source = point_charge(
        1e-6,
        0,
    )

    potential = electric_potential(
        source,
        Point3D(
            Coordinate(1),
            Coordinate(0),
            Coordinate(0),
        ),
    )

    assert isinstance(
        potential,
        Potential,
    )

def test_electric_field_is_negative_gradient_of_potential():

    source = point_charge(
        1e-6,
        0,
    )

    x = 1.0
    dx = 1e-6

    p1 = Point3D(
        Coordinate(x - dx),
        Coordinate(0),
        Coordinate(0),
    )

    p2 = Point3D(
        Coordinate(x + dx),
        Coordinate(0),
        Coordinate(0),
    )
    center = Point3D(
        Coordinate(x),
        Coordinate(0),
        Coordinate(0),
    )

    potential1 = electric_potential(
        source,
        p1,
    )

    potential2 = electric_potential(
        source,
        p2,
    )

    numerical_field = -(
        potential2.value - potential1.value
    ) / (2 * dx)

    analytical_field = electric_field(
        source,
        center,
    )

    assert math.isclose(
        numerical_field,
        analytical_field.x,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )

    assert math.isclose(
        analytical_field.y,
        0.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        analytical_field.z,
        0.0,
        abs_tol=1e-12,
    )