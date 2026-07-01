from sensorlab.physics.electrostatics import (
    PointCharge,
    electric_field,
)
from sensorlab.physics.geometry import (
    Grid2D,
    Point3D,
)

from sensorlab.physics.quantities.electrical import Charge
from sensorlab.physics.quantities.geometry import Coordinate

from sensorlab.visualization.scene import Scene
from sensorlab.visualization.renderer_matplotlib import MatplotlibRenderer


def point(
    x: float,
    y: float,
) -> Point3D:

    return Point3D(
        Coordinate(x),
        Coordinate(y),
        Coordinate(0),
    )


def main():

    # ==========================================================
    # Physical model
    # ==========================================================

    charge = PointCharge(
        charge=Charge(1e-6),
        position=point(
            0,
            0,
        ),
    )

    # ==========================================================
    # Computational grid
    # ==========================================================

    grid = Grid2D(
        xmin=-2,
        xmax=2,
        ymin=-2,
        ymax=2,
        nx=21,
        ny=21,
    )

    # ==========================================================
    # Scene
    # ==========================================================

    scene = Scene()

    scene.draw_charge(
        charge,
        name="+1 µC",
    )

    # ==========================================================
    # Calculate field
    # ==========================================================

    for observation_point in grid:

        if (
            observation_point.x.value
            == charge.position.x.value
            and observation_point.y.value
            == charge.position.y.value
        ):
            continue

        field = electric_field(
            charge,
            observation_point,
        )

        scene.draw_vector(
            origin=observation_point,
            vector=field,
        )

    # ==========================================================
    # Render
    # ==========================================================

    renderer = MatplotlibRenderer(
        vector_mode="normalized",
        vector_scale=0.15,
    )

    renderer.render(scene)

    renderer.show()


if __name__ == "__main__":
    main()