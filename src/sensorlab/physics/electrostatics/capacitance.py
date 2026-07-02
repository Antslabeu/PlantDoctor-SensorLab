"""
Relationships involving electrical capacitance.

This module provides analytical relationships
between charge, potential and capacitance.

No specific capacitor geometry is assumed.
"""


from sensorlab.physics.quantities.electrical import (
    Capacitance,
    Charge,
    Potential,
)


def from_charge_and_potential(
    charge: Charge,
    potential: Potential,
) -> Capacitance:
    """
    Compute capacitance from charge and potential.

    Parameters
    ----------
    charge
        Electrical charge.

    potential
        Electric potential relative to reference.

    Returns
    -------
    Capacitance
    """

    if potential.value == 0:
        raise ZeroDivisionError(...)

    capacitance = charge.value / potential.value

    if capacitance < 0:
        raise ValueError("Capacitance cannot be negative.")

    return Capacitance(capacitance)