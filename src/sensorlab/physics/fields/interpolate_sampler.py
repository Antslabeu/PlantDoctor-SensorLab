from __future__ import annotations
from sensorlab.physics.geometry import Vector3D, Point3D


def interpolate_sample(field, point, ):
    """
    Bilinearly interpolate a vector field
    at an arbitrary point.
    """

    grid = field.grid

    #
    # Local coordinates
    #

    fx = (point.x.value - grid.xmin) / grid.dx
    fy = ( point.y.value - grid.ymin) / grid.dy

    #
    # Cell indices
    #

    ix = int(fx)
    iy = int(fy)

    #
    # Clamp to valid cell
    #

    ix = max(0, min(ix, grid.nx - 2), )
    iy = max(0, min(iy, grid.ny - 2), )

    #
    # Local coordinates inside cell
    #

    tx = fx - ix
    ty = fy - iy

    #
    # X component
    #

    x00 = field.x[iy, ix]
    x10 = field.x[iy, ix + 1]
    x01 = field.x[iy + 1, ix]
    x11 = field.x[iy + 1, ix + 1]

    x0 = x00 + tx * (x10 - x00)
    x1 = x01 + tx * (x11 - x01)

    x = x0 + ty * (x1 - x0)

    #
    # Y component
    #

    y00 = field.y[iy, ix]
    y10 = field.y[iy, ix + 1]
    y01 = field.y[iy + 1, ix]
    y11 = field.y[iy + 1, ix + 1]

    y0 = y00 + tx * (y10 - y00)
    y1 = y01 + tx * (y11 - y01)

    y = y0 + ty * (y1 - y0)

    return Vector3D(x=x, y=y, z=0.0, )