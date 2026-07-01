from sensorlab.physics.geometry import Point3D
from sensorlab.physics.geometry import Vector3D
from sensorlab.physics.quantities.geometry import Length, Coordinate

import math
import pytest


def test_create_point():
    p = Point3D(
        Length(1),
        Length(2),
        Length(3),
    )

    assert p.x.value == 1
    assert p.y.value == 2
    assert p.z.value == 3

def test_create_vector():
    v = Vector3D(
        1,
        2,
        3,
    )

    assert v.x == 1
    assert v.y == 2
    assert v.z == 3

def test_vector_between_points():
    p1 = Point3D(
        Length(1),
        Length(2),
        Length(3),
    )
    p2 = Point3D(
        Length(4),
        Length(6),
        Length(8),
    )

    v = p1.vector_to(p2)

    assert v.x == 3
    assert v.y == 4
    assert v.z == 5

def test_distance_between_points():

    p1 = Point3D(
        Length(0),
        Length(0),
        Length(0),
    )

    p2 = Point3D(
        Length(3),
        Length(4),
        Length(0),
    )

    d = p1.distance_to(p2)

    assert d.value == 5

def test_vector_magnitude():

    v = Vector3D(
        3,
        4,
        0,
    )

    assert v.magnitude().value == 5


def test_vector_magnitude_squared():

    v = Vector3D(
        3,
        4,
        0,
    )

    assert v.magnitude_squared() == 25

def test_vector_normalization():

    v = Vector3D(
        3,
        4,
        0,
    )

    u = v.normalize()

    assert math.isclose(
        u.magnitude().value,
        1.0,
        rel_tol=1e-12,
    )

def test_zero_vector_normalization():

    v = Vector3D(
        0,
        0,
        0,
    )

    with pytest.raises(ValueError):
        v.normalize()

def test_vector_addition():

    v1 = Vector3D(
        1,
        2,
        3,
    )

    v2 = Vector3D(
        4,
        5,
        6,
    )

    result = v1 + v2

    assert result == Vector3D(
        5,
        7,
        9,
    )

def test_vector_subtraction():

    v1 = Vector3D(
        5,
        7,
        9,
    )

    v2 = Vector3D(
        1,
        2,
        3,
    )

    result = v1 - v2

    assert result == Vector3D(
        4,
        5,
        6,
    )

def test_vector_negation():

    v = Vector3D(
        1,
        -2,
        3,
    )

    assert -v == Vector3D(
        -1,
        2,
        -3,
    )

def test_vector_scaling():

    v = Vector3D(
        1,
        2,
        3,
    )

    result = v * 2

    assert result == Vector3D(
        2,
        4,
        6,
    )

def test_left_scalar_multiplication():

    v = Vector3D(
        1,
        2,
        3,
    )

    result = 2 * v

    assert result == Vector3D(
        2,
        4,
        6,
    )

def test_vector_division():

    v = Vector3D(
        2,
        4,
        6,
    )

    result = v / 2

    assert result == Vector3D(
        1,
        2,
        3,
    )

def test_vector_division_by_zero():

    v = Vector3D(
        1,
        2,
        3,
    )

    with pytest.raises(ZeroDivisionError):
        v / 0


def test_dot_product():

    v1 = Vector3D(
        1,
        2,
        3,
    )

    v2 = Vector3D(
        4,
        5,
        6,
    )

    assert v1 * v2 == 32

def test_perpendicular_vectors_have_zero_dot_product():

    v1 = Vector3D(
        1,
        0,
        0,
    )

    v2 = Vector3D(
        0,
        1,
        0,
    )

    assert v1 * v2 == 0

def test_cross_product():

    x = Vector3D(
        1,
        0,
        0,
    )

    y = Vector3D(
        0,
        1,
        0,
    )

    assert x.cross(y) == Vector3D(
        0,
        0,
        1,
    )


def test_cross_product_is_anti_commutative():

    a = Vector3D(
        2,
        3,
        4,
    )

    b = Vector3D(
        5,
        6,
        7,
    )

    assert a.cross(b) == -(b.cross(a))


def test_distance_to_same_point_is_zero():

    p = Point3D(
        Length(2),
        Length(5),
        Length(8),
    )

    assert p.distance_to(p).value == 0


def test_vector_normalization_preserves_direction():

    v = Vector3D(3, 4, 0)

    u = v.normalize()

    assert math.isclose(
        v.cross(u).magnitude().value,
        0.0,
        abs_tol=1e-12,
    )

def test_displacement_between_points():

    p1 = Point3D(
        Coordinate(1),
        Coordinate(2),
        Coordinate(3),
    )

    p2 = Point3D(
        Coordinate(4),
        Coordinate(6),
        Coordinate(8),
    )

    displacement = p1.displacement_to(p2)

    assert displacement == Vector3D(
        3,
        4,
        5,
    )

def test_displacement_is_antisymmetric():

    p1 = Point3D(
        Coordinate(2),
        Coordinate(5),
        Coordinate(7),
    )

    p2 = Point3D(
        Coordinate(8),
        Coordinate(9),
        Coordinate(1),
    )

    assert (
        p1.displacement_to(p2)
        ==
        -p2.displacement_to(p1)
    )

def test_direction_is_unit_vector():

    p1 = Point3D(
        Coordinate(0),
        Coordinate(0),
        Coordinate(0),
    )

    p2 = Point3D(
        Coordinate(3),
        Coordinate(4),
        Coordinate(0),
    )

    direction, distance = p1.direction_to(p2)

    assert math.isclose(
        direction.magnitude().value,
        1.0,
        rel_tol=1e-12,
    )

    assert distance.value == 5

def test_direction_points_towards_target():

    p1 = Point3D(
        Coordinate(0),
        Coordinate(0),
        Coordinate(0),
    )

    p2 = Point3D(
        Coordinate(5),
        Coordinate(0),
        Coordinate(0),
    )

    direction, _ = p1.direction_to(p2)

    assert direction == Vector3D(
        1,
        0,
        0,
    )

def test_direction_is_opposite_when_points_are_swapped():

    p1 = Point3D(
        Coordinate(0),
        Coordinate(0),
        Coordinate(0),
    )

    p2 = Point3D(
        Coordinate(3),
        Coordinate(4),
        Coordinate(5),
    )

    d12, _ = p1.direction_to(p2)
    d21, _ = p2.direction_to(p1)

    assert d12 == -d21

def test_distance_is_symmetric():

    p1 = Point3D(
        Coordinate(-5),
        Coordinate(1),
        Coordinate(9),
    )

    p2 = Point3D(
        Coordinate(7),
        Coordinate(4),
        Coordinate(-2),
    )

    _, d1 = p1.direction_to(p2)
    _, d2 = p2.direction_to(p1)

    assert d1 == d2

def test_direction_is_parallel_to_displacement():

    p1 = Point3D(
        Coordinate(1),
        Coordinate(2),
        Coordinate(3),
    )

    p2 = Point3D(
        Coordinate(4),
        Coordinate(8),
        Coordinate(9),
    )

    displacement = p1.displacement_to(p2)

    direction, _ = p1.direction_to(p2)

    assert displacement.cross(direction) == Vector3D(
        0,
        0,
        0,
    )

def test_direction_same_point_raises():

    p = Point3D(
        Coordinate(1),
        Coordinate(2),
        Coordinate(3),
    )

    with pytest.raises(ValueError):
        p.direction_to(p)

def test_zero_displacement():

    p = Point3D(
        Coordinate(1),
        Coordinate(2),
        Coordinate(3),
    )

    assert p.displacement_to(p) == Vector3D(
        0,
        0,
        0,
    )

def test_direction_times_distance_equals_displacement():

    p1 = Point3D(
        Coordinate(1),
        Coordinate(2),
        Coordinate(3),
    )

    p2 = Point3D(
        Coordinate(4),
        Coordinate(6),
        Coordinate(8),
    )

    displacement = p1.displacement_to(p2)

    direction, distance = p1.direction_to(p2)

    reconstructed = direction * distance.value

    assert reconstructed == displacement