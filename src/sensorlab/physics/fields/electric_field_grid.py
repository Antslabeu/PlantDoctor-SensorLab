from __future__ import annotations
from dataclasses import dataclass
import numpy as np


from sensorlab.physics.geometry import Grid2D, Vector3D
from sensorlab.physics.fields import PotentialField
from sensorlab.physics.numerics import gradient


@dataclass(frozen=True)
class ElectricFieldGrid:
    """
    Electric field sampled on a regular 2D grid.

    Each grid node stores the Cartesian components of the
    electric field:

        Ex(x, y)
        Ey(x, y)

    The field is typically obtained from the gradient
    of an electrostatic potential.
    """

    grid: Grid2D

    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:

        if self.x.shape != self.y.shape:
            raise ValueError("Electric field components must have identical shapes.")

        if self.x.shape != (
            self.grid.ny,
            self.grid.nx,
        ):
            raise ValueError("Electric field shape does not match Grid2D.")

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of the field arrays.
        """

        return self.x.shape

    @property
    def magnitude(self) -> np.ndarray:
        """
        Magnitude of the electric field.
        """

        return np.sqrt(
            self.x**2 +
            self.y**2
        )

    @property
    def max_magnitude(self) -> float:
        """
        Maximum electric field magnitude.
        """

        return float(np.max(self.magnitude, ))

    @property
    def min_magnitude(self) -> float:
        """
        Minimum electric field magnitude.
        """

        return float(np.min(self.magnitude,))

    @classmethod
    def from_potential(
        cls,
        potential: PotentialField,
    ) -> "ElectricFieldGrid":
        """
        Compute the electric field from an electrostatic potential.

        The electric field is obtained as the negative gradient
        of the scalar potential:

            E = -∇V
        """

        dVdx, dVdy = gradient(
            values=potential.values,
            grid=potential.grid,
        )

        return cls(
            grid=potential.grid,
            x=-dVdx,
            y=-dVdy,
        )


    def sample(self, point: Point3D, ) -> Vector3D:
        """
        Bilinearly interpolate electric field
        at an arbitrary point.
        """

        grid = self.grid

        #
        # Local coordinates
        #

        fx = (point.x.value - grid.xmin) / grid.dx

        fy = (point.y.value - grid.ymin) / grid.dy

        #
        # Lower indices
        #

        ix = int(fx)
        iy = int(fy)

        #
        # Clamp
        #

        ix = max(
            0,
            min(ix, grid.nx - 2),
        )

        iy = max(
            0,
            min(iy, grid.ny - 2),
        )

        tx = fx - ix
        ty = fy - iy

        #
        # Ex
        #

        ex00 = self.x[iy, ix]
        ex10 = self.x[iy, ix + 1]
        ex01 = self.x[iy + 1, ix]
        ex11 = self.x[iy + 1, ix + 1]

        ex0 = ex00 + tx * (ex10 - ex00)
        ex1 = ex01 + tx * (ex11 - ex01)

        ex = ex0 + ty * (ex1 - ex0)

        #
        # Ey
        #

        ey00 = self.y[iy, ix]
        ey10 = self.y[iy, ix + 1]
        ey01 = self.y[iy + 1, ix]
        ey11 = self.y[iy + 1, ix + 1]

        ey0 = ey00 + tx * (ey10 - ey00)
        ey1 = ey01 + tx * (ey11 - ey01)

        ey = ey0 + ty * (ey1 - ey0)

        return Vector3D(
            ex,
            ey,
            0
        )