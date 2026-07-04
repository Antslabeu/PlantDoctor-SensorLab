from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from sensorlab.physics.boundary_conditions import (
    DirichletBoundaryConditions,
)

from sensorlab.physics.fields import PotentialField
from sensorlab.physics.geometry import Grid2D

from numba import njit

@njit(
    cache=True,
    fastmath=True,
)
def _iterate_kernel(
    values: np.ndarray,
    fixed: np.ndarray,
    relaxation: float,
    tolerance: float,
    max_iterations: int,
) -> int:
    """
    Numerical Gauss-Seidel / SOR kernel.

    Returns
    -------
    int
        Number of iterations required for convergence.
    """

    ny, nx = values.shape

    for iteration in range(max_iterations):

        max_delta = 0.0

        for iy in range(1, ny - 1):
            for ix in range(1, nx - 1):

                if fixed[iy, ix]:
                    continue

                old = values[iy, ix]

                average = (
                    values[iy - 1, ix]
                    + values[iy + 1, ix]
                    + values[iy, ix - 1]
                    + values[iy, ix + 1]
                ) * 0.25

                updated = old + relaxation * (average - old)

                values[iy, ix] = updated

                delta = abs(updated - old)

                if delta > max_delta:
                    max_delta = delta

        if max_delta < tolerance:
            return iteration + 1

    raise RuntimeError(
        "Laplace solver did not converge."
    )


@dataclass
class LaplaceSolver:
    """
    Finite-difference solver for the 2D Laplace equation.

    The solver computes the electric potential inside a
    rectangular domain with Dirichlet boundary conditions.
    """

    tolerance: float = 1e-6
    max_iterations: int = 10_000
    relaxation: float = 1.8

    # ==========================================================
    # Public API
    # ==========================================================

    def __post_init__(self):
        self.iterations: int = 0

    def solve(
        self,
        grid: Grid2D,
        boundary_conditions: DirichletBoundaryConditions,
        relaxation: float | None = None,
    ) -> PotentialField:
        """
        Solve the Laplace equation.
        """

        self._potential = self._initialize(grid)

        if relaxation is None:
            n = max(grid.nx, grid.ny)
            relaxation = 2.0 / (1.0 + np.sin(np.pi / n))

        self.relaxation = relaxation

        print(f"Using relaxation factor: {self.relaxation}")

        self._apply_boundary_conditions(
            boundary_conditions,
        )

        self.iterations = 0
        self._iterate()

        return self._potential

    # ==========================================================
    # Internal pipeline
    # ==========================================================

    def _initialize(
        self,
        grid: Grid2D,
    ) -> PotentialField:
        """
        Create an empty potential field.
        """

        return PotentialField(
            grid=grid,
            values=np.zeros(
                (grid.ny, grid.nx),
                dtype=float,
            ),
        )

    def _apply_boundary_conditions(
        self,
        boundary_conditions: DirichletBoundaryConditions,
    ) -> None:
        """
        Apply fixed electrode potentials to the grid.
        """

        self._fixed_nodes = np.zeros(
            self._potential.shape,
            dtype=bool,
        )

        for electrode in boundary_conditions:
            for iy, ix, point in self._potential.grid.indices():
                if not electrode.contains(point):
                    continue

                self._potential.values[iy, ix] = (electrode.potential.value)
                self._fixed_nodes[iy, ix] = True

    def _iterate(self, ) -> None:
        """
        Solve the Laplace equation using the
        Gauss-Seidel iteration.
        """

        self.iterations = _iterate_kernel(
            values=self._potential.values,
            fixed=self._fixed_nodes,
            relaxation=self.relaxation,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )

    

    def _converged(
        self,
        previous_error: float,
    ) -> bool:
        """
        Check convergence criterion.
        """
        return previous_error < self.tolerance