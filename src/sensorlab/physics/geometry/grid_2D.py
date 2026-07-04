from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from sensorlab.physics.geometry.point import Point3D
from sensorlab.physics.quantities.geometry import Coordinate


@dataclass(frozen=True)
class Grid2D:
    """
    Uniform rectangular 2D grid.

    The grid is defined in world coordinates and generates Point3D
    objects on demand. It contains no physics and no visualization.
    """

    xmin: float
    xmax: float

    ymin: float
    ymax: float

    nx: int
    ny: int

    def __post_init__(self):

        if self.nx < 2:
            raise ValueError("nx must be at least 2.")

        if self.ny < 2:
            raise ValueError("ny must be at least 2.")

        if self.xmax <= self.xmin:
            raise ValueError("xmax must be greater than xmin.")

        if self.ymax <= self.ymin:
            raise ValueError("ymax must be greater than ymin.")

    # ==========================================================
    # Geometry
    # ==========================================================

    @property
    def dx(self) -> float:
        """Grid spacing along X."""
        return (self.xmax - self.xmin) / (self.nx - 1)

    @property
    def dy(self) -> float:
        """Grid spacing along Y."""
        return (self.ymax - self.ymin) / (self.ny - 1)

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape."""
        return self.ny, self.nx

    @property
    def x(self) -> tuple[float, ...]:
        """Grid X coordinates."""
        return tuple(
            self.xmin + i * self.dx
            for i in range(self.nx)
        )
        
    @property
    def y(self) -> tuple[float, ...]:
        """Grid Y coordinates."""
        return tuple(
            self.ymin + j * self.dy
            for j in range(self.ny)
        )

    @property
    def width(self):
        return self.xmax - self.xmin
    
    @property
    def extent(self):
        return (
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        )

    @property
    def center(self):
        """
        Return the center of the grid.
        """

        return Point3D(
            Coordinate((self.xmin + self.xmax) / 2),
            Coordinate((self.ymin + self.ymax) / 2),
            Coordinate(0),
        )

    @property
    def corners(self):
        """
        Return the corners of the grid.
        """

        return [
            Point3D(
                Coordinate(x),
                Coordinate(y),
                Coordinate(0),
            )
            for x in (self.xmin, self.xmax)
            for y in (self.ymin, self.ymax)
        ]



    # ==========================================================
    # Point access
    # ==========================================================

    def at(
        self,
        ix: int,
        iy: int,
    ) -> Point3D:
        """
        Return grid point at coordinate ix, iy.
        """

        if not (0 <= ix < self.nx):
            raise IndexError("Grid X index out of range.")

        if not (0 <= iy < self.ny):
            raise IndexError("Grid Y index out of range.")

        x = self.xmin + ix * self.dx
        y = self.ymin + iy * self.dy

        return Point3D(
            Coordinate(x),
            Coordinate(y),
            Coordinate(0),
        )

    def __getitem__(
        self,
        index: tuple[int, int],
    ) -> Point3D:
        """
        grid[ix, iy]
        """

        ix, iy = index
        return self.at(ix, iy)

    def indices(self):
        """
        Iterate over all grid nodes together with their indices.
        """

        for iy, y in enumerate(self.y):
            for ix, x in enumerate(self.x):
                yield (
                    iy,
                    ix,
                    Point3D(
                        Coordinate(x),
                        Coordinate(y),
                        Coordinate(0),
                    ),
                )

    def points(self):
        """
        Iterate over all grid points.
        """
        yield from self
    
    


    
    
    # ==========================================================
    # Iteration
    # ==========================================================

    def __iter__(self) -> Iterator[Point3D]:
        """
        Iterate over all grid points.

        Order:
            left -> right
            bottom -> top
        """

        for iy in range(self.ny):
            for ix in range(self.nx):
                yield self.at(ix, iy)

    def __len__(self) -> int:
        """
        Total number of grid points.
        """

        return self.nx * self.ny