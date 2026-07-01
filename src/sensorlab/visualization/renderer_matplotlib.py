from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from sensorlab.visualization.colors import Colors
from sensorlab.visualization.primitives import (
    ChargePrimitive,
    PointPrimitive,
    Primitive,
    VectorPrimitive,
)
from sensorlab.visualization.scene import Scene


class MatplotlibRenderer:
    """
    Renderer using Matplotlib.

    The renderer knows only about Scene and Primitive objects.
    It has no knowledge of the underlying physics implementation.
    """

    def __init__(
        self,
        *,
        vector_scale: float = 1,
        vector_mode: str = "normalized",
        figsize: tuple[int, int] = (8, 8),
        equal_axes: bool = True,
        grid: bool = True,
    ):

        self.fig, self.ax = plt.subplots(figsize=figsize)

        self.vector_scale = vector_scale
        self.vector_mode = vector_mode

        if equal_axes:
            self.ax.set_aspect("equal")

        if grid:
            self.ax.grid(True)

        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")

    def _scale_vector(self, vector: Vector3D, ) -> Vector3D:
        """
        Scale a vector for visualization.
        """

        if self.vector_mode == "linear":
            return vector * self.vector_scale

        if self.vector_mode == "normalized":
            length = vector.magnitude().value
            if length == 0:
                return vector

            return vector.normalize() * self.vector_scale

        raise ValueError(
            f"Unknown vector mode: {self.vector_mode}"
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def render(
        self,
        scene: Scene,
    ) -> None:
        """
        Render every object contained in the scene.
        """

        for primitive in scene:

            if isinstance(primitive, ChargePrimitive):
                self._render_charge(primitive)

            elif isinstance(primitive, PointPrimitive):
                self._render_point(primitive)

            elif isinstance(primitive, VectorPrimitive):
                self._render_vector(primitive)

            else:
                raise TypeError(
                    f"Unsupported primitive: {type(primitive).__name__}"
                )

    def clear(self) -> None:
        """
        Clear current axes.
        """

        self.ax.cla()

    def show(self) -> None:
        """
        Display the current figure.
        """

        plt.tight_layout()
        plt.show()

    def save(
        self,
        filename: str | Path,
        *,
        dpi: int = 300,
    ) -> None:
        """
        Save current figure.
        """

        plt.tight_layout()
        self.fig.savefig(
            filename,
            dpi=dpi,
        )

    # ==========================================================
    # Rendering methods
    # ==========================================================

    def _render_point(
        self,
        primitive: PointPrimitive,
    ) -> None:

        p = primitive.position

        self.ax.scatter(
            p.x.value,
            p.y.value,
            color=Colors.POINT,
            s=40,
        )

        if primitive.name:

            self.ax.text(
                p.x.value,
                p.y.value,
                primitive.name,
            )

    def _render_charge(
        self,
        primitive: ChargePrimitive,
    ) -> None:

        charge = primitive.charge

        p = charge.position

        color = Colors.charge(charge.charge.value)
        positive = charge.charge.value >= 0

        symbol = "+" if positive else "−"

        self.ax.scatter(
            p.x.value,
            p.y.value,
            color=color,
            s=250,
            edgecolors="black",
            zorder=3,
        )

        self.ax.text(
            p.x.value,
            p.y.value,
            symbol,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
        )

        if primitive.name:

            self.ax.text(
                p.x.value,
                p.y.value - 0.15,
                primitive.name,
                ha="center",
            )

    def _render_vector(
        self,
        primitive: VectorPrimitive,
    ) -> None:

        p = primitive.origin
        v = self._scale_vector(primitive.vector)

        self.ax.quiver(
            p.x.value,
            p.y.value,
            v.x,
            v.y,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=Colors.VECTOR,
        )

        if primitive.name:
            self.ax.text(
                p.x.value + v.x,
                p.y.value + v.y,
                primitive.name,
            )