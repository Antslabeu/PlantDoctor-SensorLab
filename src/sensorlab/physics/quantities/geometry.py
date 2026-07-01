from dataclasses import dataclass

from .base import Quantity


@dataclass(frozen=True)
class Length(Quantity):
    unit = "m"


@dataclass(frozen=True)
class Area(Quantity):
    unit = "m²"


@dataclass(frozen=True)
class Volume(Quantity):
    unit = "m³"

@dataclass(frozen=True)
class Coordinate(Quantity):
    """Cartesian coordinate."""
    unit = "m"
    allow_negative = True