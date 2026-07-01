from dataclasses import dataclass
from .vector import Vector3D


@dataclass(frozen=True)
class ForceVector(Vector3D):
    """
    Force vector.

    Components are expressed in newtons (N).
    """