from sensorlab.physics.electrostatics import PointCharge
from sensorlab.physics.geometry import Point3D

from sensorlab.physics.quantities.electrical import Charge
from sensorlab.physics.quantities.geometry import Coordinate

from sensorlab.visualization.scene import Scene
from sensorlab.visualization.primitives import ChargePrimitive
from sensorlab.visualization.renderer_matplotlib import MatplotlibRenderer


def main():

    scene = Scene()

    charge = PointCharge(
        charge=Charge(1e-6),
        position=Point3D(
            Coordinate(0),
            Coordinate(0),
            Coordinate(0),
        ),
    )

    scene.draw_charge(
        charge,
        name="+1 µC",
    )

    renderer = MatplotlibRenderer()
    renderer.render(scene)
    renderer.show()


if __name__ == "__main__":
    main()