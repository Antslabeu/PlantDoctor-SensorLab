"""
Validation tests for homogeneous dielectric materials.

These tests verify physical laws rather than implementation details.
"""

import math

from sensorlab.physics.electrostatics import (
    PointCharge,
    electric_field,
    electric_potential,
)

from sensorlab.physics.geometry import Point3D

from sensorlab.physics.materials import (
    AIR,
    FR4,
    VACUUM,
    WATER,
    Dielectric,
)

from sensorlab.physics.quantities.electrical import (
    Charge,
    RelativePermittivity,
)

from sensorlab.physics.quantities.geometry import Coordinate


# ==========================================================
# Helpers
# ==========================================================


def point(
    x: float,
    y: float,
    z: float = 0.0,
) -> Point3D:

    return Point3D(
        Coordinate(x),
        Coordinate(y),
        Coordinate(z),
    )


def charge(
    q: float = 1e-6,
) -> PointCharge:

    return PointCharge(
        charge=Charge(q),
        position=point(0, 0),
    )


# ==========================================================
# Vacuum compatibility
# ==========================================================


def test_vacuum_matches_default_field():

    source = charge()
    p = point(1, 0)

    field_default = electric_field(
        source,
        p,
    )

    field_vacuum = electric_field(
        source,
        p,
        material=VACUUM,
    )

    assert field_default == field_vacuum


def test_vacuum_matches_default_potential():

    source = charge()
    p = point(1, 0)

    potential_default = electric_potential(
        source,
        p,
    )

    potential_vacuum = electric_potential(
        source,
        p,
        material=VACUUM,
    )

    assert potential_default == potential_vacuum


# ==========================================================
# Field scaling
# ==========================================================


def test_field_scales_with_relative_permittivity():

    source = charge()
    p = point(1, 0)

    field_air = electric_field(
        source,
        p,
        material=AIR,
    )

    field_fr4 = electric_field(
        source,
        p,
        material=FR4,
    )

    expected_ratio = (
        FR4.relative_permittivity.value
        / AIR.relative_permittivity.value
    )

    measured_ratio = (
        field_air.magnitude().value
        / field_fr4.magnitude().value
    )

    assert math.isclose(
        measured_ratio,
        expected_ratio,
        rel_tol=1e-12,
    )


def test_field_is_strongly_reduced_in_water():

    source = charge()
    p = point(1, 0)

    field_vacuum = electric_field(
        source,
        p,
        material=VACUUM,
    )

    field_water = electric_field(
        source,
        p,
        material=WATER,
    )

    expected = (
        field_vacuum.magnitude().value
        / WATER.relative_permittivity.value
    )

    assert math.isclose(
        field_water.magnitude().value,
        expected,
        rel_tol=1e-12,
    )


# ==========================================================
# Potential scaling
# ==========================================================


def test_potential_scales_with_relative_permittivity():

    source = charge()
    p = point(1, 0)

    vacuum = electric_potential(
        source,
        p,
        material=VACUUM,
    )

    fr4 = electric_potential(
        source,
        p,
        material=FR4,
    )

    expected = (
        vacuum.value
        / FR4.relative_permittivity.value
    )

    assert math.isclose(
        fr4.value,
        expected,
        rel_tol=1e-12,
    )


# ==========================================================
# Direction
# ==========================================================


def test_dielectric_preserves_field_direction():

    source = charge()
    p = point(1, 1)

    vacuum = electric_field(
        source,
        p,
        material=VACUUM,
    ).normalize()

    water = electric_field(
        source,
        p,
        material=WATER,
    ).normalize()

    assert vacuum.almost_equal(water)


# ==========================================================
# Superposition
# ==========================================================


def test_superposition_still_holds_inside_dielectric():

    charges = [

        PointCharge(
            Charge(+1e-6),
            point(-0.5, 0),
        ),

        PointCharge(
            Charge(-2e-6),
            point(+0.5, 0),
        ),

    ]

    p = point(0, 1)

    total = electric_field(
        charges,
        p,
        material=FR4,
    )

    separate = (
        electric_field(
            charges[0],
            p,
            material=FR4,
        )
        +
        electric_field(
            charges[1],
            p,
            material=FR4,
        )
    )

    assert total == separate


# ==========================================================
# Limiting case
# ==========================================================


def test_extremely_large_permittivity_reduces_field():

    material = Dielectric(
        name="Test",
        relative_permittivity=RelativePermittivity(
            1e9,
        ),
    )

    field = electric_field(
        charge(),
        point(1, 0),
        material=material,
    )

    assert field.magnitude().value < 1e-4


def test_extremely_large_permittivity_reduces_potential():

    material = Dielectric(
        name="Test",
        relative_permittivity=RelativePermittivity(
            1e9,
        ),
    )

    potential = electric_potential(
        charge(),
        point(1, 0),
        material=material,
    )

    assert potential.value < 1e-2


# ==========================================================
# Physical law
# ==========================================================


def test_field_ratio_equals_inverse_permittivity_ratio():

    source = charge()
    p = point(1, 0)

    fr4 = electric_field(
        source,
        p,
        material=FR4,
    )

    water = electric_field(
        source,
        p,
        material=WATER,
    )

    measured = (
        fr4.magnitude().value
        / water.magnitude().value
    )

    expected = (
        WATER.relative_permittivity.value
        / FR4.relative_permittivity.value
    )

    assert math.isclose(
        measured,
        expected,
        rel_tol=1e-12,
    )