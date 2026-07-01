from dataclasses import dataclass

from .base import Material


@dataclass(frozen=True)
class Conductor(Material):
    conductivity: float

    def __post_init__(self):
        if self.conductivity < 0:
            raise ValueError("Conductivity cannot be negative.")