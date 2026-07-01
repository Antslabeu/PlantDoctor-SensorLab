from dataclasses import dataclass


@dataclass(frozen=True)
class Colors:

    POSITIVE_CHARGE = "red"
    NEGATIVE_CHARGE = "blue"

    FIELD = "green"

    VECTOR = "black"

    POINT = "black"

    GRID = "#DDDDDD"

    POTENTIAL = "viridis"


    @staticmethod
    def charge(value: float) -> str:
        return (
            Colors.POSITIVE_CHARGE
            if value >= 0
            else Colors.NEGATIVE_CHARGE
        )