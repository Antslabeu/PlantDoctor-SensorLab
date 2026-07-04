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

from sensorlab.sampling import FieldSampler2D


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
    # Sampler
    # ==========================================================

    sampler = FieldSampler2D(grid)

    # ==========================================================
    # Scene
    # ==========================================================

    scene = Scene()

    for charge in charges:
        scene.add_charge(charge)

    # ==========================================================
    # Field
    # ==========================================================

    potential = sampler.sample_scalar_field(
        electric_potential,
        source=charges,
    )

    scene.add_scalar_field(potential)

    # ==========================================================
    # Render
    # ==========================================================

    renderer = MatplotlibRenderer(
        vector_style=None,
        vector_mode=VectorMode.LINEAR,
        window_title="04 Electric Potential",
    )

    renderer.render(scene)
    renderer.show()


if __name__ == "__main__":
    main()