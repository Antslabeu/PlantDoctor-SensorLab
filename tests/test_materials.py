import pytest

from sensorlab.physics.materials import (
    AIR,
    VACUUM,
    WATER,
    PTFE,
    FR4,
    SOLDERMASK,
    COPPER,
    ALUMINUM,
    Dielectric,
    Conductor,
)

from sensorlab.physics.quantities.electrical import RelativePermittivity


# ==========================================================
# Types
# ==========================================================

def test_air_is_dielectric():
    assert isinstance(AIR, Dielectric)


def test_copper_is_conductor():
    assert isinstance(COPPER, Conductor)


# ==========================================================
# Physical relationships
# ==========================================================

def test_vacuum_has_relative_permittivity_equal_one():
    assert VACUUM.relative_permittivity.value == 1.0


def test_air_has_higher_permittivity_than_vacuum():
    assert AIR.relative_permittivity.value > VACUUM.relative_permittivity.value


def test_water_has_high_relative_permittivity():
    assert WATER.relative_permittivity.value > 50


def test_fr4_has_higher_permittivity_than_ptfe():
    assert FR4.relative_permittivity.value > PTFE.relative_permittivity.value


def test_soldermask_is_similar_to_fr4():
    assert 2.5 < SOLDERMASK.relative_permittivity.value < 5.0


# ==========================================================
# Conductivity
# ==========================================================

def test_copper_is_better_conductor_than_aluminum():
    assert COPPER.conductivity > ALUMINUM.conductivity


def test_conductor_has_positive_conductivity():
    assert COPPER.conductivity > 0


# ==========================================================
# Validation
# ==========================================================

def test_negative_relative_permittivity_is_rejected():
    with pytest.raises(ValueError):
        RelativePermittivity(-1)


def test_relative_permittivity_below_vacuum_is_rejected():
    with pytest.raises(ValueError):
        RelativePermittivity(0.9)


def test_negative_conductivity_is_rejected():
    with pytest.raises(ValueError):
        Conductor(
            name="Broken",
            conductivity=-1,
        )


def test_negative_dielectric_conductivity_is_rejected():
    with pytest.raises(ValueError):
        Dielectric(
            name="Broken",
            relative_permittivity=RelativePermittivity(2.0),
            conductivity=-1,
        )