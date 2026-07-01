from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self, TypeVar

from sensorlab.physics.quantities.geometry import Length


@dataclass(frozen=True)
class Vector3D:
    """
    Vector in Cartesian 3D space.
    """

    x: float
    y: float
    z: float

    T = TypeVar("T", bound="Vector3D")

    def cast(self, cls: type[T]) -> T:
        """
        Cast vector into another Vector3D-derived class.
        """
        return cls(
            self.x,
            self.y,
            self.z,
        )

    def magnitude(self) -> Length:
        """
        Euclidean vector length.
        """
        return Length(
            math.sqrt(
                self.x**2 +
                self.y**2 +
                self.z**2
            )
        )

    def magnitude_squared(self) -> float:
        """
        Squared Euclidean vector length.
        """
        return (
            self.x**2 +
            self.y**2 +
            self.z**2
        )

    def normalize(self) -> Self:
        """
        Unit vector.
        """

        length = self.magnitude().value

        if length == 0:
            raise ValueError("Cannot normalize zero vector.")

        return self.__class__(
            self.x / length,
            self.y / length,
            self.z / length,
        )

    def __add__(self, other: Vector3D) -> Self:
        return self.__class__(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: Vector3D) -> Self:
        return self.__class__(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(self, other: Vector3D | float) -> Self | float:
        """
        Multiply by a scalar or compute the dot product.
        """

        if isinstance(other, Vector3D):
            return self.dot(other)

        if isinstance(other, (int, float)):
            return self.scale(other)

        raise TypeError(
            f"Unsupported operand type(s) for *: "
            f"'{self.__class__.__name__}' "
            f"and '{type(other).__name__}'"
        )

    def __rmul__(self, scalar: float) -> Self:
        return self.scale(scalar)

    def __truediv__(self, scalar: float) -> Self:

        if scalar == 0:
            raise ZeroDivisionError(
                "Cannot divide vector by zero."
            )

        return self.__class__(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar,
        )

    def __neg__(self) -> Self:
        return self.__class__(
            -self.x,
            -self.y,
            -self.z,
        )

    def dot(self, other: Vector3D) -> float:
        """
        Scalar (dot) product.
        """
        return (
            self.x * other.x +
            self.y * other.y +
            self.z * other.z
        )

    def scale(self, scalar: float) -> Self:
        """
        Scale the vector by a scalar.
        """
        return self.__class__(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )

    def cross(self, other: Vector3D) -> Self:
        """
        Vector (cross) product.
        """
        return self.__class__(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def is_zero(
        self,
        abs_tol: float = 1e-12,
    ) -> bool:

        return (
            math.isclose(
                self.x,
                0.0,
                abs_tol=abs_tol,
            )
            and
            math.isclose(
                self.y,
                0.0,
                abs_tol=abs_tol,
            )
            and
            math.isclose(
                self.z,
                0.0,
                abs_tol=abs_tol,
            )
        )

    def almost_equal(
        self,
        other: Vector3D,
        rel_tol: float = 1e-9,
        abs_tol: float = 1e-12,
    ) -> bool:
        """
        Check if two vectors are almost equal.
        """

        return (
            math.isclose(
                self.x,
                other.x,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            and
            math.isclose(
                self.y,
                other.y,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            and
            math.isclose(
                self.z,
                other.z,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        )