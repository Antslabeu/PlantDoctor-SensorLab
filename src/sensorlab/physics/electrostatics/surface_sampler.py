from __future__ import annotations

import math


from sensorlab.physics.electrodes import RectangleElectrode
from sensorlab.physics.geometry import Point3D, Vector3D, SurfaceSample
from sensorlab.physics.quantities.geometry import Coordinate


class SurfaceSampler:
    """
    Samples the boundary of a rectangular electrode.

    Each sample stores:

    • point
    • outward normal
    • represented boundary length
    """

    @staticmethod
    def sample(
        electrode: RectangleElectrode,
        spacing: float,
    ) -> list[SurfaceSample]:

        samples: list[SurfaceSample] = []

        for rectangle in electrode.rectangles:

            cx = rectangle.center.x.value
            cy = rectangle.center.y.value

            width = rectangle.width.value
            height = rectangle.height.value

            xmin = cx - width / 2
            xmax = cx + width / 2

            ymin = cy - height / 2
            ymax = cy + height / 2

            print("Height: ")
            print(height)
            print("spacing: ")
            print(spacing)


            count_lr = math.floor(height / spacing)
            count_tb = math.floor(width / spacing)


            #
            # Left edge
            #

            for i in range(count_lr):
                y = ymin + (i + 0.5) * spacing
                samples.append(
                    SurfaceSample(
                        point=Point3D(
                            Coordinate(xmin - spacing / 2),
                            Coordinate(y),
                            Coordinate(0),
                        ),
                        normal=Vector3D(-1.0, 0.0, 0.0),
                        length=spacing,
                    )
                )

            #
            # Right edge
            #

            for i in range(count_lr):
                y = ymin + (i + 0.5) * spacing

                samples.append(
                    SurfaceSample(
                        point=Point3D(
                            Coordinate(xmax + spacing / 2),
                            Coordinate(y),
                            Coordinate(0),
                        ),
                        normal=Vector3D(-1.0, 0.0, 0.0),
                        length=spacing,
                    )
                )

            #
            # Bottom edge
            #

            for i in range(count_tb):
                x = xmin + (i + 0.5) * spacing

                samples.append(
                    SurfaceSample(
                        point=Point3D(
                            Coordinate(x),
                            Coordinate(ymin - spacing / 2),
                            Coordinate(0),
                        ),
                        normal=Vector3D(-1.0, 0.0, 0.0),
                        length=spacing,
                    )
                )

            #
            # Top edge
            #

            for i in range(count_tb):
                x = xmin + (i + 0.5) * spacing

                samples.append(
                    SurfaceSample(
                        point=Point3D(
                            Coordinate(x),
                            Coordinate(ymax + spacing / 2),
                            Coordinate(0),
                        ),
                        normal=Vector3D(-1.0, 0.0, 0.0),
                        length=spacing,
                    )
                )

        return samples