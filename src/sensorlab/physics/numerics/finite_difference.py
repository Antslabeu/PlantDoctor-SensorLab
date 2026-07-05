from __future__ import annotations

import numpy as np
from numba import njit

from sensorlab.physics.geometry import Grid2D


@njit(cache=True)
def _central_difference_x_kernel(
    values: np.ndarray,
    dx: float,
) -> np.ndarray:

    ny, nx = values.shape

    derivative = np.zeros_like(values)

    inv_dx = 1.0 / dx
    inv_2dx = 0.5 * inv_dx

    # ---------------------------------------------------------
    # Interior (central difference)
    # ---------------------------------------------------------

    for iy in range(ny):
        for ix in range(1, nx - 1):

            derivative[iy, ix] = (
                values[iy, ix + 1]
                - values[iy, ix - 1]
            ) * inv_2dx

    # ---------------------------------------------------------
    # Left boundary (forward difference)
    # ---------------------------------------------------------

    for iy in range(ny):

        derivative[iy, 0] = (
            values[iy, 1]
            - values[iy, 0]
        ) * inv_dx

    # ---------------------------------------------------------
    # Right boundary (backward difference)
    # ---------------------------------------------------------

    for iy in range(ny):

        derivative[iy, nx - 1] = (
            values[iy, nx - 1]
            - values[iy, nx - 2]
        ) * inv_dx

    return derivative


def central_difference_x(
    values: np.ndarray,
    grid: Grid2D,
) -> np.ndarray:
    """
    Compute ∂f/∂x on a regular Grid2D.

    Parameters
    ----------
    values
        Scalar field values.
    grid
        Regular Cartesian grid.

    Returns
    -------
    np.ndarray
        First derivative with respect to x.
    """

    return _central_difference_x_kernel(
        values=values,
        dx=grid.dx,
    )


@njit(cache=True)
def _central_difference_y_kernel(
    values: np.ndarray,
    dy: float,
) -> np.ndarray:

    ny, nx = values.shape

    derivative = np.zeros_like(values)

    inv_dy = 1.0 / dy
    inv_2dy = 0.5 * inv_dy

    # ---------------------------------------------------------
    # Interior (central difference)
    # ---------------------------------------------------------

    for iy in range(1, ny - 1):
        for ix in range(nx):

            derivative[iy, ix] = (
                values[iy + 1, ix]
                - values[iy - 1, ix]
            ) * inv_2dy

    # ---------------------------------------------------------
    # Bottom boundary (forward difference)
    # ---------------------------------------------------------

    for ix in range(nx):

        derivative[0, ix] = (
            values[1, ix]
            - values[0, ix]
        ) * inv_dy

    # ---------------------------------------------------------
    # Top boundary (backward difference)
    # ---------------------------------------------------------

    for ix in range(nx):

        derivative[ny - 1, ix] = (
            values[ny - 1, ix]
            - values[ny - 2, ix]
        ) * inv_dy

    return derivative


def central_difference_y(
    values: np.ndarray,
    grid: Grid2D,
) -> np.ndarray:
    """
    Compute ∂f/∂y on a regular Grid2D.

    Parameters
    ----------
    values
        Scalar field values.

    grid
        Regular Cartesian grid.

    Returns
    -------
    np.ndarray
        First derivative with respect to y.
    """

    return _central_difference_y_kernel(
        values=values,
        dy=grid.dy,
    )


def gradient(
    values: np.ndarray,
    grid: Grid2D,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradient of a scalar field.

    Parameters
    ----------
    values
        Scalar field values.

    grid
        Regular Cartesian grid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (∂f/∂x, ∂f/∂y)
    """

    dx = central_difference_x(
        values,
        grid,
    )

    dy = central_difference_y(
        values,
        grid,
    )

    return dx, dy