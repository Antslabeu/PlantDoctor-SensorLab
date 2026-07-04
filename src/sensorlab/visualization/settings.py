from enum import Enum

class VectorMode(Enum):
    NORMALIZED = 1
    LINEAR = 2

class VectorStyle(Enum):
    QUIVER = 1
    STREAMPLOT = 2

class ScalarColormap(Enum):
    POTENTIAL = 1
    PERMITTIVITY = 2
    ENERGY = 3
    MOISTURE = 4