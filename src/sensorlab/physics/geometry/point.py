from dataclasses import dataclass
import math

from sensorlab.physics.quantities.geometry import Length
from sensorlab.physics.quantities.geometry import Coordinate


@dataclass(frozen=True)
class Point3D:
    """
    Point in Cartesian 3D space.
    """

    x: Coordinate
    y: Coordinate
    z: Coordinate

    def is_same_position(self, other: Vector3D) -> bool:
        """
        Check if two vectors represent the same position in space.
        """

        return (
            math.isclose(self.x, other.x)
            and
            math.isclose(self.y, other.y)
            and
            math.isclose(self.z, other.z)
        )

    def vector_to(self, other: "Point3D") -> "Vector3D":
        """
        Return vector from this point to another point.
        """
        from .vector import Vector3D

        return Vector3D(
            other.x.value - self.x.value,
            other.y.value - self.y.value,
            other.z.value - self.z.value,
        )

    def distance_to(self, other: "Point3D") -> Length:
        """
        Euclidean distance to another point.
        """
        return self.vector_to(other).magnitude()
    
    def direction_to(self, other: "Point3D", ) -> tuple["Vector3D", Length]:
        """
        Return the unit direction vector and distance to another point.

        Returns
        -------
        tuple
            (direction, distance)
        """

        vector = self.vector_to(other)

        distance = vector.magnitude()

        if distance.value == 0:
            raise ValueError("Points must not occupy the same position.")

        direction = vector.normalize()

        return direction, distance

    def displacement_to(self, other: Point3D, ) -> Vector3D:
        """
        Return the displacement vector from this point to another point.
        """
        return self.vector_to(other)

