from .database import *

from .base import Material
from .dielectric import Dielectric
from .conductor import Conductor

__all__ = [
    "Material",
    "Dielectric",
    "Conductor",
    "VACUUM",
    "AIR",
    "WATER",
    "FR4",
    "PTFE",
    "SOLDERMASK",
    "COPPER",
    "ALUMINUM",
]