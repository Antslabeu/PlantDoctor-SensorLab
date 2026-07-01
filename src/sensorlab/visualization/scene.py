from __future__ import annotations

from typing import Iterator
from typing import Self, TypeVar

from sensorlab.physics.electrostatics import PointCharge
from sensorlab.physics.geometry import Point3D
from sensorlab.physics.geometry import Vector3D

from .primitives import (
    ChargePrimitive,
    PointPrimitive,
    VectorPrimitive,
)


class Scene:
    """
    Collection of objects forming a visualization scene.

    Scene is renderer-agnostic.
    It only stores objects in world coordinates.
    """

    def __init__(self):
        self._objects: list[object] = []


    def _add(self, obj: object) -> object:
        """
        Low-level method for adding an object to the scene.
        """
        self._objects.append(obj)
        return obj

    # ==========================================================
    # High-level API
    # ==========================================================



    def draw_charge(self, charge: PointCharge, *, name: str | None = None, ) -> ChargePrimitive:
        """
        Add a point charge to the scene.
        """

        primitive = ChargePrimitive(
            charge=charge,
            name=name,
        )

        self._add(primitive)

        return primitive

    def draw_point(self, point: Point3D, *, name: str | None = None, ) -> PointPrimitive:
        """
        Add a point to the scene.
        """

        primitive = PointPrimitive(
            position=point,
            name=name,
        )

        self._add(primitive)

        return primitive

    def draw_vector(self, origin: Point3D, vector: Vector3D, *, name: str | None = None, ) -> VectorPrimitive:
        """
        Add a vector to the scene.
        """

        primitive = VectorPrimitive(
            origin=origin,
            vector=vector,
            name=name,
        )

        self._add(primitive)

        return primitive

    def clear(self) -> None:
        """
        Remove every object from the scene.
        """

        self._objects.clear()

    def remove(self, obj: object, ) -> None:
        """
        Remove an object from the scene.
        """

        self._objects.remove(obj)

    def __iter__(self) -> Iterator[object]:
        return iter(self._objects)

    def __len__(self) -> int:
        return len(self._objects)

    def __getitem__(self, index: int, ) -> object:
        return self._objects[index]

    def copy(self) -> Scene:
        scene = Scene()
        scene.extend(self._objects)

        return scene