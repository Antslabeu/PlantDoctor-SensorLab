from dataclasses import dataclass

from .base import Quantity


@dataclass(frozen=True)
class Charge(Quantity):
    """Electric charge."""
    unit = "C"
    allow_negative = True


@dataclass(frozen=True)
class Potential(Quantity):
    """Electric potential."""
    unit = "V"
    allow_negative = True


@dataclass(frozen=True)
class Voltage(Quantity):
    """Electric potential difference."""
    unit = "V"
    allow_negative = True


@dataclass(frozen=True)
class ElectricField(Quantity):
    """Electric field strength."""
    unit = "V/m"
    allow_negative = True


@dataclass(frozen=True)
class Capacitance(Quantity):
    """Electrical capacitance."""
    unit = "F"

@dataclass(frozen=True)
class RelativePermittivity(Quantity):
    """Relative permittivity of a dielectric material."""
    unit = ""
    def __post_init__(self):
        super().__post_init__()
        if self.value < 1:
            raise ValueError("Relative permittivity cannot be below vacuum.")