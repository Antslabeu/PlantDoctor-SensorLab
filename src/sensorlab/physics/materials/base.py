from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """
    Base class for every material.
    """

    name: str