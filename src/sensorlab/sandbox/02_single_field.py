from sensorlab.physics.electrostatics import (
    PointCharge,
    electric_field,
)
from sensorlab.physics.geometry import Point3D

from sensorlab.physics.quantities.electrical import Charge
from sensorlab.physics.quantities.geometry import Coordinate

from sensorlab.visualization.scene import Scene
from sensorlab.visualization.renderer_matplotlib import MatplotlibRenderer, VectorStyle, VectorMode


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
        position=point(0, 0),
    )

    # ==========================================================
    # Observation points
    # ==========================================================

    observation_points = [

        point(-2, -2),
        point(-1, -2),
        point(0, -2),
        point(1, -2),
        point(2, -2),

        point(-2, -1),
        point(-1, -1),
        point(0, -1),
        point(1, -1),
        point(2, -1),

        point(-2, 0),
        point(-1, 0),
        point(1, 0),
        point(2, 0),

        point(-2, 1),
        point(-1, 1),
        point(0, 1),
        point(1, 1),
        point(2, 1),

        point(-2, 2),
        point(-1, 2),
        point(0, 2),
        point(1, 2),
        point(2, 2),
    ]

    # ==========================================================
    # Scene
    # ==========================================================

    scene = Scene()

    scene.add_charge(
        charge,
        name="+1 µC",
    )

    # ==========================================================
    # Calculate field
    # ==========================================================

    for p in observation_points:

        field = electric_field(
            charge,
            p,
        )

        scene.add_point(p)

        scene.add_vector(
            origin=p,
            vector=field,
        )

    # ==========================================================
    # Render
    # ==========================================================

    renderer = MatplotlibRenderer(
        vector_style=VectorStyle.STREAMPLOT,
        window_title="02 Single Field",
    )

    renderer.render(scene)
    renderer.show()


if __name__ == "__main__":
    main()