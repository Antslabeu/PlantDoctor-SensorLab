from __future__ import annotations

from typing import Any, Callable

from sensorlab.physics.geometry import Grid2D, Point3D
from sensorlab.physics.fields import PotentialField
import numpy as np


class FieldSampler2D:
    """
    Evaluate physical quantities on a 2D grid.

    This class is intentionally independent of any specific
    physical model. It simply evaluates arbitrary functions
    defined on Point3D objects.
    """

    def __init__(
        self,
        grid: Grid2D,
    ) -> None:

        self.grid = grid

    # ==========================================================
    # Scalar field
    # ==========================================================

    def sample_scalar_field(
        self,
        function: Callable[..., Any],
        *,
        exclude: Callable[[Point3D], bool] | None = None,
        **kwargs,
    ) -> PotentialField:
        """
        Sample a scalar-valued function on the grid.
        """

        values = np.zeros(
            (
                self.grid.ny,
                self.grid.nx,
            ),
            dtype=float,
        )

        for iy, ix, point in self.grid.indices():
            if exclude is not None and exclude(point):
                continue

            value = function(
                point=point,
                **kwargs,
            )

            if hasattr(value, "value"):
                value = value.value

            values[iy, ix] = float(value)

        return PotentialField(
            grid=self.grid,
            values=values,
        )

    # ==========================================================
    # Vector field
    # ==========================================================

    def sample_vector_field(
        self,
        function: Callable[..., Any],
        *,
        exclude: Callable[[Point3D], bool] | None = None,
        **kwargs,
    ) -> PotentialField:
        """
        Sample a vector-valued function on the grid.
        """

        ex = np.zeros(
            (
                self.grid.ny,
                self.grid.nx,
            ),
            dtype=float,
        )
        ey = np.zeros(
            (
                self.grid.ny,
                self.grid.nx,
            ),
            dtype=float,
        )

        for point in self.grid:
            if exclude is not None and exclude(point):
                continue

            vector = function(
                point=point,
                **kwargs,
            )

            samples.append(
                (
                    point,
                    vector,
                )
            )

        return ElectricField(
            grid=self.grid,
            ex=ex,
            ey=ey,
        )