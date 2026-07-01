from dataclasses import dataclass
from .base import Quantity


@dataclass(frozen=True)
class Force(Quantity):
    """Force."""

    unit = "N"
    allow_negative = True

