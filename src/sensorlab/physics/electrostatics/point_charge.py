from dataclasses import dataclass

from sensorlab.physics.geometry import Point3D
from sensorlab.physics.quantities.electrical import Charge


@dataclass(frozen=True)
class PointCharge:
    """
    Point electric charge located in 3D space.
    """

    charge: Charge
    position: Point3D