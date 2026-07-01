from __future__ import annotations
from dataclasses import dataclass
from .vector import Vector3D


@dataclass(frozen=True)
class ElectricFieldVector(Vector3D):
    """
    Electric field vector.

    Components are expressed in volts per metre (V/m).
    """