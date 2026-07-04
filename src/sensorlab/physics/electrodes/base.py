from __future__ import annotations

from dataclasses import dataclass

from sensorlab.physics.materials import Conductor
from sensorlab.physics.quantities.electrical import Voltage


@dataclass(frozen=True)
class Electrode:
    """
    Base class for every electrode.

    An electrode is a conductor kept at a fixed electrical
    potential (Dirichlet boundary condition).
    """

    material: Conductor
    potential: Voltage