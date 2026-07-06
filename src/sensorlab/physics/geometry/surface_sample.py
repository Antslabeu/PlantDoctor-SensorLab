from dataclasses import dataclass

from .point import Point3D
from .vector import Vector3D


@dataclass(frozen=True)
class SurfaceSample:
    """
    One sampling point on the electrode boundary.
    """

    point: Point3D
    normal: Vector3D
    length: float