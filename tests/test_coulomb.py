import math

import pytest

from sensorlab.constants import COULOMB_CONSTANT
from sensorlab.physics.electrostatics import PointCharge, coulomb_force
from sensorlab.physics.geometry import Point3D, Vector3D, ForceVector
from sensorlab.physics.quantities.electrical import Charge
from sensorlab.physics.quantities.geometry import Length
from sensorlab.physics.quantities.geometry import Coordinate
from sensorlab.physics.quantities.electrical import Charge


def point_charge(q: float, x: float, y: float = 0.0, z: float = 0.0) -> PointCharge:
    return PointCharge(
        charge=Charge(q),
        position=Point3D(
            Coordinate(x),
            Coordinate(y),
            Coordinate(z),
        ),
    )


def test_equal_charges_repel():

    source = point_charge(+1e-6, 0)
    target = point_charge(+1e-6, 1)

    force = coulomb_force(source, target)

    assert force.x > 0
    assert force.y == 0
    assert force.z == 0


def test_opposite_charges_attract():

    source = point_charge(+1e-6, 0)
    target = point_charge(-1e-6, 1)

    force = coulomb_force(source, target)

    assert force.x < 0
    assert force.y == 0
    assert force.z == 0


def test_zero_charge_produces_zero_force():

    source = point_charge(0, 0)
    target = point_charge(1e-6, 1)

    force = coulomb_force(source, target)

    assert force == ForceVector(0, 0, 0)


def test_inverse_square_law():

    source = point_charge(1e-6, 0)

    target1 = point_charge(1e-6, 1)
    target2 = point_charge(1e-6, 2)

    f1 = coulomb_force(source, target1)
    f2 = coulomb_force(source, target2)

    assert math.isclose(
        f1.magnitude().value,
        4 * f2.magnitude().value,
        rel_tol=1e-12,
    )


def test_force_is_linear_with_charge():

    source = point_charge(1e-6, 0)

    target1 = point_charge(1e-6, 1)
    target2 = point_charge(2e-6, 1)

    f1 = coulomb_force(source, target1)
    f2 = coulomb_force(source, target2)

    assert math.isclose(
        f2.magnitude().value,
        2 * f1.magnitude().value,
        rel_tol=1e-12,
    )


def test_newtons_third_law():

    q1 = point_charge(1e-6, 0)
    q2 = point_charge(-1e-6, 1)

    f12 = coulomb_force(q1, q2)
    f21 = coulomb_force(q2, q1)
    result = f12 + f21

    assert result.is_zero()


def test_same_position_raises():

    q1 = point_charge(1e-6, 0)
    q2 = point_charge(2e-6, 0)

    with pytest.raises(ValueError):
        coulomb_force(q1, q2)


def test_force_along_y_axis():

    source = point_charge(1e-6, 0, 0)
    target = point_charge(1e-6, 0, 1)

    force = coulomb_force(source, target)

    assert force.x == 0
    assert force.y > 0
    assert force.z == 0


def test_force_direction_is_correct():

    source = point_charge(1e-6, 0, 0)
    target = point_charge(1e-6, 1, 1)

    force = coulomb_force(source, target)

    assert math.isclose(
        force.x,
        force.y,
        rel_tol=1e-12,
    )


def test_force_matches_analytical_solution():

    source = point_charge(2e-6, 0)
    target = point_charge(3e-6, 2)

    force = coulomb_force(source, target)

    expected = (
        COULOMB_CONSTANT
        * 2e-6
        * 3e-6
        / (2**2)
    )

    assert math.isclose(
        force.magnitude().value,
        expected,
        rel_tol=1e-12,
    )


def test_translation_invariance():

    q1 = point_charge(1e-6, 0)
    q2 = point_charge(1e-6, 1)

    q3 = point_charge(1e-6, 100, 50, -20)
    q4 = point_charge(1e-6, 101, 50, -20)

    f1 = coulomb_force(q1, q2)
    f2 = coulomb_force(q3, q4)

    assert f1 == f2


def test_force_is_parallel_to_displacement():
    source = point_charge(1e-6, 0, 0, 0)
    target = point_charge(1e-6, 3, 4, 5)
    force = coulomb_force(source, target)
    displacement = source.position.vector_to(target.position)
    cross = force.cross(displacement)
    assert cross.is_zero()