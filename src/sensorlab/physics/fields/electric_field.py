from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensorlab.physics.geometry import Grid2D, Vector3D



@dataclass(frozen=True)
class ElectricField:
    """
    Electric field sampled on a 2D grid.

    This class is produced from PotentialField by
    computing the spatial gradient.

    It is intentionally minimal.
    """

    grid: Grid2D

    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:

        expected = (self.grid.ny, self.grid.nx)

        if self.x.shape != expected:
            raise ValueError(
                "Electric field X component shape does not match Grid2D."
            )

        if self.y.shape != expected:
            raise ValueError(
                "Electric field Y component shape does not match Grid2D."
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape

    def value(
        self,
        ix: int,
        iy: int,
    ) -> Vector3D:
        """
        Electric field vector at grid node.
        """

        return Vector3D(
            self.x[iy, ix],
            self.y[iy, ix],
            0.0,
        )

    @classmethod
    def from_potential(
        cls,
        potential: PotentialField,
    ) -> ElectricField:
        ex, ey = gradient(
            potential.values,
            potential.grid,
        )

        return cls(
            grid=potential.grid,
            ex=ex,
            ey=ey,
        )