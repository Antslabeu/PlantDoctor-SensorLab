from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensorlab.physics.electrostatics import SurfaceSampler
from sensorlab.physics.electrodes import RectangleElectrode
from sensorlab.physics.fields import ElectricDisplacementField
from sensorlab.physics.geometry import Point3D, SurfaceSample
from sensorlab.physics.quantities import Length


@dataclass(frozen=True)
class SurfaceChargeDensity:
    """
    Surface charge density evaluated along the boundary
    of a single electrode.
    """

    samples: list[SurfaceSample]
    sigma: np.ndarray

    @property
    def points(self) -> list[Point3D]:
        return [
            sample.point
            for sample in self.samples
        ]
    @property
    def values(self) -> np.ndarray:
        return self.sigma

    @classmethod
    def from_displacement_field(
        cls,
        field: ElectricDisplacementField,
        electrode: RectangleElectrode,
        spacing: float,
    ) -> SurfaceChargeDensity:

        print("spacing before SurfaceSampler:", spacing, type(spacing))
        
        samples = SurfaceSampler.sample(
            electrode=electrode,
            spacing=spacing,
        )

        sigma = np.empty(
            len(samples),
            dtype=float,
        )

        for i, sample in enumerate(samples):
            D = field.sample(sample.point)
            sigma[i] = (D.x * sample.normal.x + D.y * sample.normal.y)

        return cls(
            samples=samples,
            sigma=sigma,
        )