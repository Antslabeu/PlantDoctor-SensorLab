from dataclasses import dataclass
from sensorlab.visualization.settings import VectorMode, VectorStyle, ScalarColormap


@dataclass(frozen=True)
class Colors:

    POSITIVE_CHARGE = "red"
    NEGATIVE_CHARGE = "blue"



    SCALAR_COLORMAPS = {
        ScalarColormap.POTENTIAL: "viridis",
        ScalarColormap.PERMITTIVITY: "plasma",
        ScalarColormap.ENERGY: "inferno",
        ScalarColormap.MOISTURE: "Blues",
    }


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

    @staticmethod
    def electrode_color( self, voltage: float, ) -> str:
        if voltage > 0:
            return POSITIVE_CHARGE
        if voltage < 0:
            return NEGATIVE_CHARGE
        return "#808080"          # masa


    @staticmethod
    def scalar_colormap(colormap: ScalarColormap, ) -> str:
        return Colors.SCALAR_COLORMAPS[colormap]