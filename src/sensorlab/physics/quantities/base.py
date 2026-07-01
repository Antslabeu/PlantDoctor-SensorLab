from dataclasses import dataclass


@dataclass(frozen=True)
class Quantity:
    """
    Base class for all physical quantities.
    """

    value: float

    # class attributes (nadpisywane w klasach potomnych)
    unit = ""
    allow_negative = False

    def __post_init__(self):
        if not self.allow_negative and self.value < 0:
            raise ValueError(
                f"{self.__class__.__name__} cannot be negative."
            )

    def __float__(self):
        return self.value

    def as_float(self):
        return self.value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value} {self.unit})"