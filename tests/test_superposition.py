import math

import pytest

from sensorlab.physics.electrostatics import (
    PointCharge,
    electric_field,
    electric_potential,
)
from sensorlab.physics.geometry import (
    ElectricFieldVector,
    Point3D,
)
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


POINT = Point3D(
    Coordinate(2),
    Coordinate(0),
    Coordinate(0),
)


# ============================================================
# A
# ============================================================

def test_single_source_matches_original_field():

    charge = point_charge(
        1e-6,
        0,
    )

    single = electric_field(
        charge,
        POINT,
    )

    collection = electric_field(
        [charge],
        POINT,
    )

    assert single.almost_equal(collection)


# ============================================================
# B
# ============================================================

def test_single_element_collection_matches_single_charge():

    charge = point_charge(
        1e-6,
        0,
    )

    assert electric_field(
        charge,
        POINT,
    ).almost_equal(
        electric_field(
            [charge],
            POINT,
        )
    )


# ============================================================
# C
# ============================================================

def test_two_identical_charges_double_the_field():

    charge = point_charge(
        1e-6,
        0,
    )

    single = electric_field(
        charge,
        POINT,
    )

    doubled = electric_field(
        [
            charge,
            charge,
        ],
        POINT,
    )

    assert math.isclose(
        doubled.x,
        2 * single.x,
        rel_tol=1e-12,
    )

    assert math.isclose(
        doubled.y,
        2 * single.y,
        rel_tol=1e-12,
    )

    assert math.isclose(
        doubled.z,
        2 * single.z,
        rel_tol=1e-12,
    )


# ============================================================
# D
# ============================================================

def test_opposite_charges_cancel_potential():

    charges = [
        point_charge(
            1e-6,
            -1,
        ),
        point_charge(
            -1e-6,
            -1,
        ),
    ]

    potential = electric_potential(
        charges,
        POINT,
    )

    assert math.isclose(
        potential.value,
        0.0,
        abs_tol=1e-12,
    )


# ============================================================
# E
# ============================================================

def test_potential_superposition_is_linear():

    charge = point_charge(
        1e-6,
        0,
    )

    single = electric_potential(
        charge,
        POINT,
    )

    doubled = electric_potential(
        [
            charge,
            charge,
        ],
        POINT,
    )

    assert math.isclose(
        doubled.value,
        2 * single.value,
        rel_tol=1e-12,
    )


# ============================================================
# F
# ============================================================

def test_order_of_sources_does_not_matter():

    q1 = point_charge(
        1e-6,
        0,
    )

    q2 = point_charge(
        -2e-6,
        1,
    )

    q3 = point_charge(
        3e-6,
        -2,
    )

    field1 = electric_field(
        [
            q1,
            q2,
            q3,
        ],
        POINT,
    )

    field2 = electric_field(
        [
            q3,
            q1,
            q2,
        ],
        POINT,
    )

    assert field1.almost_equal(field2)


# ============================================================
# G
# ============================================================

def test_empty_collection_raises():

    with pytest.raises(ValueError):

        electric_field(
            [],
            POINT,
        )

    with pytest.raises(ValueError):

        electric_potential(
            [],
            POINT,
        )


# ============================================================
# H
# ============================================================

def test_return_types():

    charge = point_charge(
        1e-6,
        0,
    )

    field = electric_field(
        [
            charge,
        ],
        POINT,
    )

    potential = electric_potential(
        [
            charge,
        ],
        POINT,
    )

    assert isinstance(
        field,
        ElectricFieldVector,
    )

    assert isinstance(
        potential,
        Potential,
    )