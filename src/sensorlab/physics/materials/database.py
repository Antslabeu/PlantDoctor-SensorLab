from .dielectric import Dielectric
from .conductor import Conductor
from sensorlab.physics.quantities.electrical import RelativePermittivity

# ---------------------
# Dielectrics
# ---------------------
AIR = Dielectric(
    name="Air",
    relative_permittivity=RelativePermittivity(1.0006),
)

VACUUM = Dielectric(
    name="Vacuum",
    relative_permittivity=RelativePermittivity(1.0),
)

FR4 = Dielectric(
    name="FR4",
    relative_permittivity=RelativePermittivity(4.4),
)

PTFE = Dielectric(
    name="PTFE",
    relative_permittivity=RelativePermittivity(2.1),
)

WATER = Dielectric(
    name="Water",
    relative_permittivity=RelativePermittivity(80.0),
)

SOLDERMASK = Dielectric(
    name="Soldermask",
    relative_permittivity=RelativePermittivity(3.3),
)


# ---------------------
# Conductors
# ---------------------
COPPER = Conductor(
    name="Copper",
    conductivity=5.96e7
)

ALUMINUM = Conductor(
    name="Aluminum",
    conductivity=3.5e7
)