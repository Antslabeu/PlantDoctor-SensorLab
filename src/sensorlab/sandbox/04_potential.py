from sensorlab.physics.electrostatics import (
    PointCharge,
    electric_field,
    electric_potential
)
from sensorlab.physics.geometry import (
    Grid2D,
    Point3D,
)

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
    # Dipole
    # ==========================================================

    charges = [
        PointCharge(
            charge=Charge(+1e-6),
            position=point(-0.5, 0),
        ),

        PointCharge(
            charge=Charge(-1e-6),
            position=point(+0.5, 0),
        ),

    ]

    # ==========================================================
    # Grid
    # ==========================================================

    grid = Grid2D(
        xmin=-2,
        xmax=2,
        ymin=-2,
        ymax=2,
        nx=25,
        ny=25,
    )

    # ==========================================================
    # Scene
    # ==========================================================

    scene = Scene()

    for charge in charges:
        scene.add_charge(charge)

    # ==========================================================
    # Field
    # ==========================================================

    scalar_field_values = []
    for observation_point in grid:
        field = electric_field(
            charges,
            observation_point,
        )

        scene.add_vector(
            origin=observation_point,
            vector=field,
        )
        scalar_field_values.append(
            (observation_point, electric_potential(charges, observation_point))
        )

        
    scene.add_scalar_field(
        samples=scalar_field_values,
        name="Electric Potential",
    )

    # ==========================================================
    # Render
    # ==========================================================

    renderer = MatplotlibRenderer(
        vector_style=VectorStyle.STREAMPLOT,
        vector_mode=VectorMode.LINEAR,
        window_title="04 Electric Potential",
    )

    renderer.render(scene)
    renderer.show()


if __name__ == "__main__":
    main()