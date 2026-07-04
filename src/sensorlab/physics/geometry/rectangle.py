from __future__ import annotations

from dataclasses import dataclass

from sensorlab.physics.geometry.point import Point3D

from sensorlab.physics.quantities.geometry import (
    Area,
    Length,
)


@dataclass(frozen=True)
class Rectangle:
    """
    Axis-aligned rectangle.

    Used to describe PCB copper geometry.
    """

    center: Point3D

    width: Length

    height: Length

    @property
    def area(self) -> Area:
        """
        Rectangle area.
        """
        return Area(
            self.width.value * self.height.value,
        )

    @property
    def left(self) -> float:
        return self.center.x.value - self.width.value / 2

    @property
    def right(self) -> float:
        return self.center.x.value + self.width.value / 2

    @property
    def bottom(self) -> float:
        return self.center.y.value - self.height.value / 2

    @property
    def top(self) -> float:
        return self.center.y.value + self.height.value / 2

    def contains(self, point: Point3D, ) -> bool:
        """
        Check if a point lies inside the rectangle.
        """

        half_width = self.width.value / 2
        half_height = self.height.value / 2

        return (
            self.center.x.value - half_width
            <= point.x.value
            <= self.center.x.value + half_width
            and
            self.center.y.value - half_height
            <= point.y.value
            <= self.center.y.value + half_height
        )