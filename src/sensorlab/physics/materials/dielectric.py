from dataclasses import dataclass

from .base import Material

from sensorlab.physics.quantities.electrical import RelativePermittivity


@dataclass(frozen=True)
class Dielectric(Material):
    relative_permittivity: RelativePermittivity
    conductivity: float = 0.0

    def __post_init__(self):
        if self.conductivity < 0:
            raise ValueError("Conductivity cannot be negative.")
