import pytest

from sensorlab.physics.electrodes import RectangleElectrode
from sensorlab.physics.geometry import Point3D, Rectangle
from sensorlab.physics.materials import COPPER
from sensorlab.physics.quantities.electrical import Voltage
from sensorlab.physics.quantities.geometry import (
    Area,
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
    width: float,
    height: float,
) -> Rectangle:

    return Rectangle(
        center=point(x, y),
        width=Length(width),
        height=Length(height),
    )


# ==========================================================
# Rectangle
# ==========================================================


def test_rectangle_area():

    r = rectangle(
        0,
        0,
        2,
        5,
    )

    assert r.area == Area(10)


def test_rectangle_left():

    r = rectangle(
        10,
        0,
        4,
        2,
    )

    assert r.left == 8


def test_rectangle_right():

    r = rectangle(
        10,
        0,
        4,
        2,
    )

    assert r.right == 12


def test_rectangle_bottom():

    r = rectangle(
        0,
        10,
        2,
        6,
    )

    assert r.bottom == 7


def test_rectangle_top():

    r = rectangle(
        0,
        10,
        2,
        6,
    )

    assert r.top == 13


# ==========================================================
# RectangleElectrode
# ==========================================================


def test_single_rectangle_is_stored():

    rect = rectangle(
        0,
        0,
        1,
        1,
    )

    electrode = RectangleElectrode(
        material=COPPER,
        potential=Voltage(1),
        rectangles=[rect],
    )

    assert len(electrode.rectangles) == 1


def test_multiple_rectangles_are_stored():

    electrode = RectangleElectrode(
        material=COPPER,
        potential=Voltage(1),
        rectangles=[
            rectangle(0, 0, 1, 1),
            rectangle(2, 0, 1, 1),
            rectangle(4, 0, 1, 1),
        ],
    )

    assert len(electrode.rectangles) == 3


def test_material_is_preserved():

    electrode = RectangleElectrode(
        material=COPPER,
        potential=Voltage(2),
        rectangles=[
            rectangle(0, 0, 1, 1),
        ],
    )

    assert electrode.material is COPPER


def test_potential_is_preserved():

    electrode = RectangleElectrode(
        material=COPPER,
        potential=Voltage(3.3),
        rectangles=[
            rectangle(0, 0, 1, 1),
        ],
    )

    assert electrode.potential == Voltage(3.3)


def test_total_area_single_rectangle():

    electrode = RectangleElectrode(
        material=COPPER,
        potential=Voltage(1),
        rectangles=[
            rectangle(0, 0, 2, 5),
        ],
    )

    assert electrode.area == Area(10)


def test_total_area_multiple_rectangles():

    electrode = RectangleElectrode(
        material=COPPER,
        potential=Voltage(1),
        rectangles=[
            rectangle(0, 0, 2, 5),   # 10
            rectangle(5, 0, 1, 4),   # 4
            rectangle(9, 0, 3, 2),   # 6
        ],
    )

    assert electrode.area == Area(20)


def test_comb_like_geometry():

    electrode = RectangleElectrode(
        material=COPPER,
        potential=Voltage(1),
        rectangles=[
            rectangle(0, 0, 10, 1),   # bus
            rectangle(-4, 2, 1, 4),
            rectangle(-2, 2, 1, 4),
            rectangle(0, 2, 1, 4),
            rectangle(2, 2, 1, 4),
            rectangle(4, 2, 1, 4),
        ],
    )

    assert len(electrode.rectangles) == 6

    assert electrode.area == Area(
        10 + 5 * 4
    )