"""
Physical constants used throughout SensorLab.

All values originate from scipy.constants.
"""

from scipy.constants import epsilon_0
from scipy.constants import pi

VACUUM_PERMITTIVITY = epsilon_0      # F/m
PI = pi

COULOMB_CONSTANT = 1 / (4 * pi * epsilon_0)