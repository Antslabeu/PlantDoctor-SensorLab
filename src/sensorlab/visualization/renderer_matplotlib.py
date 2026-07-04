from __future__ import annotations

from enum import Enum

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sensorlab.visualization.colors import Colors
from sensorlab.visualization.primitives import (
    ChargePrimitive,
    PointPrimitive,
    Primitive,
    VectorPrimitive,
)
from sensorlab.visualization.scene import Scene

from sensorlab.visualization.settings import VectorMode, VectorStyle, ScalarColormap

import numpy as np




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
        window_title: str = "SensorLab",
        vector_mode: VectorMode = VectorMode.NORMALIZED,
        minimum_vector_length = 1e-1,
        vector_style: VectorStyle = VectorStyle.QUIVER,
        figsize: tuple[int, int] = (8, 8),
        equal_axes: bool = True,
        grid: bool = True,
    ):

        self.fig, self.ax = plt.subplots(figsize=figsize)

        self.vector_scale = vector_scale
        self.vector_mode = vector_mode
        self.vector_style = vector_style
        self.fig.canvas.manager.set_window_title(window_title)

        self.minimum_vector_length = minimum_vector_length


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

        if self.vector_mode == VectorMode.LINEAR:
            return vector * self.vector_scale

        if self.vector_mode == VectorMode.NORMALIZED:
            length = vector.magnitude().value
            if length < self.minimum_vector_length:
                return Vector3D(0, 0, 0)

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

        for charge in scene.charges():
            self._render_charge(charge)

        for point in scene.points():
            self._render_point(point)


        vectors = list(scene.vectors())
        self._render_vectors(vectors)


        scalar_field = list(scene.scalar_fields())
        self._render_scalar_field(scalar_field)

        for electrode in scene.electrodes():
            self._render_electrode(electrode)

        

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

    def _render_vectors(
        self,
        primitives: list[VectorPrimitive],
    ) -> None:

        match self.vector_style:
            case VectorStyle.QUIVER:
                for primitive in primitives:
                    p = primitive.origin
                    v = self._scale_vector(primitive.vector)
                    self._render_vector_quiver(p, v, primitive)

            case VectorStyle.STREAMPLOT:
                self._render_vectors_streamplot(primitives)

    def _render_scalar_field(
        self,
        primitives: list[ScalarFieldPrimitive],
        scalar_colormap: ScalarColormap = ScalarColormap.POTENTIAL,
    ) -> None:
        """
        Render scalar fields stored in the scene.
        """

        if not primitives:
            return

        for primitive in primitives:
            self._render_scalar_field_contour(primitive, scalar_colormap)


    def _render_vector_quiver(self, p: Point3D, v: Vector3D, primitive: VectorPrimitive) -> None:
        """
        Render a vector using quiver.
        """

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
    
    def _render_vectors_streamplot(self, primitives: list[VectorPrimitive], ) -> None:
        """
        Render vector field using matplotlib.streamplot().
        """

        print(f"Generated {len(primitives)} vector primitives")

        if not primitives:
            return

        # ----------------------------------------------------------
        # Collect unique coordinates
        # ----------------------------------------------------------

        xs = np.unique([
            primitive.origin.x.value
            for primitive in primitives
        ])

        ys = np.unique([
            primitive.origin.y.value
            for primitive in primitives
        ])

        nx = len(xs)
        ny = len(ys)

        U = np.zeros((ny, nx))
        V = np.zeros((ny, nx))

        x_index = {
            value: index
            for index, value in enumerate(xs)
        }

        y_index = {
            value: index
            for index, value in enumerate(ys)
        }

        # ----------------------------------------------------------
        # Fill vector field
        # ----------------------------------------------------------

        for primitive in primitives:

            p = primitive.origin
            v = self._scale_vector(
                primitive.vector
            )

            ix = x_index[p.x.value]
            iy = y_index[p.y.value]

            U[iy, ix] = v.x
            V[iy, ix] = v.y

        # ----------------------------------------------------------
        # Mesh
        # ----------------------------------------------------------

        X, Y = np.meshgrid(xs, ys)

        # ----------------------------------------------------------
        # Streamplot
        # ----------------------------------------------------------

        magnitude = np.sqrt(U**2 + V**2)
        linewidth = np.sqrt(magnitude)
        linewidth /= linewidth.max()
        linewidth = 0.5 + 2.5 * linewidth

        self.ax.streamplot(
            X,
            Y,
            U,
            V,
            density=2.4,
            linewidth=linewidth,
            arrowsize=1.2,
            color=Colors.VECTOR,
        )

    def _render_scalar_field_contour(self, primitive: ScalarFieldPrimitive, scalar_colormap: ScalarColormap, ) -> None:

        field = primitive.field
        grid = field.grid

        X, Y = np.meshgrid(
            grid.x,
            grid.y,
        )

        Z = field.values

        self.ax.contourf(
            X,
            Y,
            Z,
            levels=40,
            cmap=Colors.scalar_colormap(scalar_colormap,),
        )

        self.ax.contour(
            X,
            Y,
            Z,
            levels=20,
            colors="black",
            linewidths=0.4,
        )
        
    def _render_electrode(self, primitive: ElectrodePrimitive, ) -> None:
        """
        Render a single rectangular electrode.
        """

        x = (
            primitive.center.x.value
            - primitive.width.value / 2
        )

        y = (
            primitive.center.y.value
            - primitive.height.value / 2
        )

        rectangle = Rectangle(
            (x, y),
            primitive.width.value,
            primitive.height.value,
            facecolor="gold",
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )

        self.ax.add_patch(rectangle)

        label = f"{primitive.potential.value:.2f} V"

        if primitive.name:
            label = f"{primitive.name}\n{label}"

        self.ax.text(
            primitive.center.x.value,
            primitive.center.y.value,
            label,
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            color="black",
            zorder=11,
        )
