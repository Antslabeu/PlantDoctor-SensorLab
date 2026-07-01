from dataclasses import dataclass

from .base import Quantity


@dataclass(frozen=True)
class Frequency(Quantity):
    unit = "Hz"


@dataclass(frozen=True)
class AngularFrequency(Quantity):
    unit = "rad/s"