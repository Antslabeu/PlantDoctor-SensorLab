"""
Tests for SurfaceChargeDensity.

These tests validate only the construction of the
SurfaceChargeDensity object and its interaction with
SurfaceSampler and ElectricDisplacementField.

They intentionally do NOT validate electrostatic theory.
"""

from __future__ import annotations

import numpy as np

from sensorlab.physics.electrostatics import SurfaceSampler
from sensorlab.physics.electrodes import RectangleElectrode
from sensorlab.physics.fields import (
    ElectricDisplacementField,
    ElectricFieldGrid,
    SurfaceChargeDensity,
)
from sensorlab.physics.geometry import Grid2D, Point3D, Rectangle
from sensorlab.physics.materials import COPPER
from sensorlab.physics.quantities import Coordinate, Length, Voltage


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
        Coordinate(0),
    )


def build_electrode() -> RectangleElectrode:
    return RectangleElectrode(
        material=COPPER,
        potential=Voltage(1.0),
        rectangles=[
            Rectangle(
                center=point(0.0, 0.0),
                width=Length(0.20),
                height=Length(0.60),
            )
        ],
    )

def check_type():
    print("------------------------")
    field = build_displacement_field()
    print(type(field.grid.dx))
    print(field.grid.dx)
    print("------------------------")

def build_displacement_field() -> ElectricDisplacementField:

    grid = Grid2D(
        xmin=-1,
        xmax=1,
        ymin=-1,
        ymax=1,
        nx=51,
        ny=51,
    )

    #
    # Constant field.
    #

    Ex = np.full(
        (grid.ny, grid.nx),
        2.0,
    )

    Ey = np.full(
        (grid.ny, grid.nx),
        -3.0,
    )

    electric = ElectricFieldGrid(
        grid=grid,
        x=Ex,
        y=Ey,
    )

    return ElectricDisplacementField.from_electric_field(
        electric,
    )


# ============================================================
# Tests
# ============================================================


def test_surface_charge_density_has_same_number_of_values_as_samples():

    field = build_displacement_field()

    electrode = build_electrode()

    charge = SurfaceChargeDensity.from_displacement_field(
        field=field,
        electrode=electrode,
        spacing=field.grid.dx,
    )

    print(charge)

    assert len(charge.samples) == len(charge.sigma)
    assert charge.values.shape == (len(charge.samples),)


def test_points_property_matches_samples():

    field = build_displacement_field()

    electrode = build_electrode()

    charge = SurfaceChargeDensity.from_displacement_field(
        field=field,
        electrode=electrode,
        spacing=field.grid.dx,
    )

    for point, sample in zip(
        charge.points,
        charge.samples,
    ):
        assert point == sample.point


def test_sigma_contains_only_finite_values():

    field = build_displacement_field()

    electrode = build_electrode()

    charge = SurfaceChargeDensity.from_displacement_field(
        field=field,
        electrode=electrode,
        spacing=field.grid.dx,
    )

    assert np.all(np.isfinite(charge.sigma))


def test_sigma_has_expected_shape():

    field = build_displacement_field()

    electrode = build_electrode()

    charge = SurfaceChargeDensity.from_displacement_field(
        field=field,
        electrode=electrode,
        spacing=field.grid.dx,
    )

    assert charge.sigma.ndim == 1


def test_surface_sampler_and_charge_density_have_identical_lengths():

    field = build_displacement_field()

    electrode = build_electrode()

    samples = SurfaceSampler.sample(
        electrode=electrode,
        spacing=field.grid.dx,
    )

    charge = SurfaceChargeDensity.from_displacement_field(
        field=field,
        electrode=electrode,
        spacing=field.grid.dx,
    )

    assert len(samples) == len(charge.samples)


def test_surface_charge_density_does_not_crash_for_small_spacing():

    field = build_displacement_field()

    electrode = build_electrode()

    charge = SurfaceChargeDensity.from_displacement_field(
        field=field,
        electrode=electrode,
        spacing=field.grid.dx / 2,
    )

    assert len(charge.samples) > 0
    assert np.all(np.isfinite(charge.sigma))