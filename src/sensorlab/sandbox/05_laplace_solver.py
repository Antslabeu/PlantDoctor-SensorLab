from sensorlab.physics.boundary_conditions import (
    DirichletBoundaryConditions,
)

from sensorlab.physics.electrodes import (
    RectangleElectrode,
)

from sensorlab.physics.geometry import (
    Grid2D,
    Point3D,
    Rectangle,
)

from sensorlab.physics.materials import COPPER

from sensorlab.physics.quantities.electrical import Voltage
from sensorlab.physics.quantities.geometry import (
    Coordinate,
    Length,
)

from sensorlab.solvers import LaplaceSolver

from sensorlab.visualization.scene import Scene
from sensorlab.visualization.renderer_matplotlib import (
    MatplotlibRenderer,
)


# ==========================================================
# Helpers
# ==========================================================


def point(
    x: float,
    y: float,
) -> Point3D:

    return Point3D(
        Coordinate(x),
        Coordinate(y),
        Coordinate(0),
    )


def plate(
    *,
    x: float,
    voltage: float,
) -> RectangleElectrode:

    return RectangleElectrode(
        material=COPPER,
        potential=Voltage(voltage),
        rectangles=[
            Rectangle(
                center=point(x, 0),
                width=Length(0.05),
                height=Length(1.6),
            ),
        ],
    )


# ==========================================================
# Main
# ==========================================================


def main():

    # ======================================================
    # Grid
    # ======================================================

    grid = Grid2D(
        xmin=-1.0,
        xmax=1.0,
        ymin=-1.0,
        ymax=1.0,
        nx=151,
        ny=151,
    )

    # ======================================================
    # Electrodes
    # ======================================================

    left = plate(
        x=-0.4,
        voltage=1.0,
    )

    right = plate(
        x=0.4,
        voltage=0.0,
    )

    boundary_conditions = DirichletBoundaryConditions(
        electrodes=[
            left,
            right,
        ],
    )

    # ======================================================
    # Solve
    # ======================================================

    solver = LaplaceSolver()

    potential = solver.solve(
        grid=grid,
        boundary_conditions=boundary_conditions,
    )



    # ======================================================
    # Scene
    # ======================================================

    scene = Scene()

    scene.add_scalar_field(
        potential,
    )

    scene.add_electrode(left)
    scene.add_electrode(right)

    # ======================================================
    # Render
    # ======================================================

    renderer = MatplotlibRenderer(
        window_title=(
            f"09 Laplace Solver "
            f"({solver.iterations} iterations)"
        ),
    )

    renderer.render(scene)

    renderer.show()


if __name__ == "__main__":
    main()