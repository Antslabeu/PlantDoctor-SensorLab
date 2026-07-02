import math

import pytest

from sensorlab.physics.electrostatics.capacitance import (
    from_charge_and_potential,
)

from sensorlab.physics.quantities.electrical import (
    Capacitance,
    Charge,
    Potential,
)


# ==========================================================
# Basic definition
# ==========================================================


def test_one_coulomb_one_volt():
    """
    1 C / 1 V = 1 F
    """

    capacitance = from_charge_and_potential(
        Charge(1.0),
        Potential(1.0),
    )

    assert isinstance(
        capacitance,
        Capacitance,
    )

    assert math.isclose(
        capacitance.value,
        1.0,
    )


def test_nanocoulomb_one_volt():
    """
    1 nC / 1 V = 1 nF
    """

    capacitance = from_charge_and_potential(
        Charge(1e-9),
        Potential(1.0),
    )

    assert math.isclose(
        capacitance.value,
        1e-9,
    )


def test_one_coulomb_two_volts():
    """
    1 C / 2 V = 0.5 F
    """

    capacitance = from_charge_and_potential(
        Charge(1.0),
        Potential(2.0),
    )

    assert math.isclose(
        capacitance.value,
        0.5,
    )


# ==========================================================
# Scaling laws
# ==========================================================


def test_double_charge_doubles_capacitance():
    """
    For constant voltage:
    C ∝ Q
    """

    c1 = from_charge_and_potential(
        Charge(1.0),
        Potential(1.0),
    )

    c2 = from_charge_and_potential(
        Charge(2.0),
        Potential(1.0),
    )

    assert math.isclose(
        c2.value,
        2 * c1.value,
    )


def test_double_voltage_halves_capacitance():
    """
    For constant charge:
    C ∝ 1 / V
    """

    c1 = from_charge_and_potential(
        Charge(1.0),
        Potential(1.0),
    )

    c2 = from_charge_and_potential(
        Charge(1.0),
        Potential(2.0),
    )

    assert math.isclose(
        c2.value,
        c1.value / 2,
    )


# ==========================================================
# Sign convention
# ==========================================================


def test_negative_charge_and_negative_voltage():
    """
    (-Q)/(-V) must still produce positive capacitance.
    """

    capacitance = from_charge_and_potential(
        Charge(-1.0),
        Potential(-2.0),
    )

    assert math.isclose(
        capacitance.value,
        0.5,
    )


def test_positive_charge_negative_voltage_raises():
    """
    Opposite signs correspond to a negative capacitance,
    which is not physically allowed for passive capacitors.
    """

    with pytest.raises(ValueError):
        from_charge_and_potential(
            Charge(1.0),
            Potential(-2.0),
        )

def test_negative_charge_positive_voltage_raises():
    """
    Opposite signs correspond to a negative capacitance,
    which is not physically allowed for passive capacitors.
    """

    with pytest.raises(ValueError):
        from_charge_and_potential(
            Charge(-1.0),
            Potential(2.0),
        )


# ==========================================================
# Error handling
# ==========================================================


def test_zero_voltage_raises():
    """
    Division by zero is undefined.
    """

    with pytest.raises(
        ZeroDivisionError,
    ):
        from_charge_and_potential(
            Charge(1.0),
            Potential(0.0),
        )


# ==========================================================
# Return type
# ==========================================================


def test_returns_capacitance():
    capacitance = from_charge_and_potential(
        Charge(123.0),
        Potential(456.0),
    )

    assert isinstance(
        capacitance,
        Capacitance,
    )