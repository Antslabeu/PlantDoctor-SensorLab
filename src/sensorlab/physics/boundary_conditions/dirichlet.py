from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sensorlab.physics.electrodes import RectangleElectrode
from sensorlab.physics.quantities.electrical import Voltage


@dataclass(frozen=True)
class DirichletBoundaryConditions:
    """
    Collection of Dirichlet boundary conditions.

    Every electrode defines a fixed electric potential.

    This class does not solve the electrostatic problem.
    It only describes the boundary conditions that will
    later be consumed by an electrostatic solver.
    """

    electrodes: list[RectangleElectrode]

    def __post_init__(self) -> None:

        if not self.electrodes:
            raise ValueError("At least one electrode must be provided.")

        if len(set(id(e) for e in self.electrodes)) != len(self.electrodes):
            raise ValueError("The same electrode cannot appear more than once.")

    # ======================================================
    # Convenience
    # ======================================================

    def __iter__(self):
        yield from self.electrodes

    def __len__(self) -> int:
        return len(self.electrodes)

    def __getitem__(self, index: int) -> RectangleElectrode:
        return self.electrodes[index]

    # ======================================================
    # Helpers
    # ======================================================

    @property
    def potentials(self) -> list[Voltage]:
        """
        Potentials of all electrodes.
        """
        return [
            electrode.potential
            for electrode in self.electrodes
        ]

    def potential_of(
        self,
        electrode: RectangleElectrode,
    ) -> Voltage:
        """
        Return potential assigned to the given electrode.
        """

        if electrode not in self.electrodes:
            raise ValueError("Electrode does not belong to these boundary conditions.")

        return electrode.potential

    @property
    def voltage_span(self) -> Voltage:
        """
        Maximum potential difference between all electrodes.
        """

        potentials = self.potentials
        return max(potentials) - min(potentials)