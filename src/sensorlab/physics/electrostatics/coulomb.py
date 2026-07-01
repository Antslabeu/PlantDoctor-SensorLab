from sensorlab.constants import COULOMB_CONSTANT
from sensorlab.physics.geometry import ForceVector
from .point_charge import PointCharge


def coulomb_force(
    source: PointCharge,
    target: PointCharge,
) -> ForceVector:
    """
    Electrostatic force exerted by source on target.
    """

    direction, distance = source.position.direction_to(
        target.position
    )

    magnitude = (
        COULOMB_CONSTANT
        * abs(source.charge.value)
        * abs(target.charge.value)
        / distance.value**2
    )

    vector = direction * magnitude

    if source.charge.value * target.charge.value < 0:
        vector = -vector

    return vector.cast(ForceVector)