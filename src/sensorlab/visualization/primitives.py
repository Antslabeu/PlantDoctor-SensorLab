from __future__ import annotations

from dataclasses import dataclass

from sensorlab.physics.electrostatics import PointCharge
from sensorlab.physics.geometry import Point3D
from sensorlab.physics.geometry import Vector3D

from sensorlab.physics.quantities.electrical import Voltage
from sensorlab.physics.quantities.geometry import Length


# ============================================================
# Base class
# ============================================================


@dataclass(frozen=True)
class Primitive:
    """
    Base class for every drawable object.

    A primitive represents a physical object in the scene.
    It contains no rendering logic.
    """


# ============================================================
# Point
# ============================================================


@dataclass(frozen=True)
class PointPrimitive(Primitive):
    """
    A point in world coordinates.
    """

    position: Point3D

    name: str | None = None


# ============================================================
# Charge
# ============================================================


@dataclass(frozen=True)
class ChargePrimitive(Primitive):
    """
    A point electric charge.
    """
    charge: PointCharge
    name: str | None = None


# ============================================================
# Vector
# ============================================================


@dataclass(frozen=True)
class VectorPrimitive(Primitive):
    """
    A vector originating from a point.
    """

    origin: Point3D
    vector: Vector3D
    name: str | None = None


# ============================================================
# Scalar Field
# ============================================================

@dataclass(frozen=True)
class ScalarFieldPrimitive:
    field: PotentialField


# ============================================================
# Electrode (Rectangle)
# ============================================================

@dataclass(frozen=True)
class ElectrodePrimitive(Primitive):
    """
    A rectangular electrode.
    """

    center: Point3D
    width: Length
    height: Length
    potential: Voltage
    name: str | None = None