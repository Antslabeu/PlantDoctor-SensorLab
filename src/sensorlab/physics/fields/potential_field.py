from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensorlab.physics.geometry import Grid2D
from sensorlab.physics.quantities.electrical import Voltage


@dataclass(frozen=True)
class PotentialField:
    """
    Electric potential sampled on a 2D grid.
    """

    grid: Grid2D
    values: np.ndarray

    def __post_init__(self) -> None:

        if self.values.shape != (self.grid.ny, self.grid.nx):
            raise ValueError(
                "Potential field shape does not match Grid2D."
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def value(
        self,
        ix: int,
        iy: int,
    ) -> Voltage:
        """
        Potential at grid node (ix, iy).
        """
        return Voltage(
            self.values[iy, ix]
        )

    @property
    def copy(self) -> "PotentialField":
        return PotentialField(
            grid=self.grid,
            values=self.values.copy(),
        )