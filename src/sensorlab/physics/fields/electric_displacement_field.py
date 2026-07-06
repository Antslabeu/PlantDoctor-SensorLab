from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensorlab.physics.fields.electric_field_grid import ElectricFieldGrid
from sensorlab.physics.geometry import Grid2D
from sensorlab.constants import VACUUM_PERMITTIVITY


@dataclass(frozen=True)
class ElectricDisplacementField:
    """
    Electric displacement field D = εE.
    """

    grid: Grid2D

    x: np.ndarray
    y: np.ndarray

    @classmethod
    def from_electric_field(
        cls,
        field: ElectricFieldGrid,
        relative_permittivity: float = 1.0,
    ) -> "ElectricDisplacementField":

        epsilon = VACUUM_PERMITTIVITY * relative_permittivity

        return cls(
            grid=field.grid,
            x=epsilon * field.x,
            y=epsilon * field.y,
        )

    @property
    def magnitude(self) -> np.ndarray:
        return np.sqrt(
            self.x**2 +
            self.y**2
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape