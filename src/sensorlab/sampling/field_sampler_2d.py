from __future__ import annotations

from typing import Any, Callable

from sensorlab.physics.geometry import Grid2D, Point3D


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
    ) -> list[tuple[Point3D, float]]:
        """
        Sample a scalar-valued function on the grid.
        """

        samples: list[tuple[Point3D, float]] = []

        for point in self.grid:

            if exclude is not None and exclude(point):
                continue

            value = function(
                point=point,
                **kwargs,
            )

            if hasattr(value, "value"):
                value = value.value

            samples.append(
                (
                    point,
                    float(value),
                )
            )

        return samples

    # ==========================================================
    # Vector field
    # ==========================================================

    def sample_vector_field(
        self,
        function: Callable[..., Any],
        *,
        exclude: Callable[[Point3D], bool] | None = None,
        **kwargs,
    ) -> list[tuple[Point3D, Any]]:
        """
        Sample a vector-valued function on the grid.
        """

        samples = []

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

        return samples