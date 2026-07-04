from __future__ import annotations

from dataclasses import dataclass

from sensorlab.physics.geometry import Rectangle
from sensorlab.physics.materials import Conductor

from sensorlab.physics.quantities.electrical import Voltage
from sensorlab.physics.quantities.geometry import Area

from .base import Electrode


@dataclass(frozen=True)
class RectangleElectrode(Electrode):
    """
    Rectangular PCB copper electrode.

    The geometry may later become one element of a larger
    multi-segment electrode.
    """

    rectangles: list[Rectangle]

    def __post_init__(self):
        if not self.rectangles:
            raise ValueError("Electrode must contain at least one rectangle.")

    @property
    def area(self) -> Area:
        return Area(sum(rectangle.area.value for rectangle in self.rectangles))

    def contains(self, point: Point3D, ) -> bool:
        """
        Return True if the point belongs to the electrode.
        """

        return any(
            rectangle.contains(point)
            for rectangle in self.rectangles
        )
