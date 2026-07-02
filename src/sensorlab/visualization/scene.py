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
    ScalarFieldPrimitive,
)


class Scene:
    """
    Collection of objects forming a visualization scene.

    Scene is renderer-agnostic.
    It only stores objects in world coordinates.
    """

    def __init__(self):
        self._charges: list[ChargePrimitive] = []
        self._points: list[PointPrimitive] = []
        self._vectors: list[VectorPrimitive] = []
        self._scalar_fields: list[ScalarFieldPrimitive] = []


    def _add(self, obj: object) -> object:
        """
        Low-level method for adding an object to the scene.
        """

        match obj:
            case ChargePrimitive():
                self._charges.append(obj)

            case PointPrimitive():
                self._points.append(obj)

            case VectorPrimitive():
                self._vectors.append(obj)

            case ScalarFieldPrimitive():
                self._scalar_fields.append(obj)

            case _:
                raise TypeError(f"Unsupported object type: {type(obj)}")

        return obj

    # ==========================================================
    # High-level API
    # ==========================================================



    def add_charge(self, charge: PointCharge, *, name: str | None = None, ) -> ChargePrimitive:
        """
        Add a point charge to the scene.
        """

        primitive = ChargePrimitive(
            charge=charge,
            name=name,
        )

        self._add(primitive)

        return primitive

    def add_point(self, point: Point3D, *, name: str | None = None, ) -> PointPrimitive:
        """
        Add a point to the scene.
        """

        primitive = PointPrimitive(
            position=point,
            name=name,
        )

        self._add(primitive)

        return primitive

    def add_vector(self, origin: Point3D, vector: Vector3D, *, name: str | None = None, ) -> VectorPrimitive:
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

    def add_scalar_field(self, samples: list[tuple[Point3D, float]], name: str | None = None, ) -> ScalarFieldPrimitive:
        """
        Add a scalar field to the scene.
        """
        self._scalar_fields.append(
            ScalarFieldPrimitive(
                samples=samples,
                name=name,
            )
        )
        return self._scalar_fields[-1]

    def charges(self) -> Iterator[ChargePrimitive]:
        """
        Iterate over all charges in the scene.
        """
        yield from self._charges
    
    def points(self) -> Iterator[PointPrimitive]:
        """
        Iterate over all points in the scene.
        """
        yield from self._points
    
    def vectors(self) -> Iterator[VectorPrimitive]:
        """
        Iterate over all vectors in the scene.
        """
        yield from self._vectors
    
    def scalar_fields(self) -> Iterator[ScalarFieldPrimitive]:
        """
        Iterate over all scalar fields in the scene.
        """
        yield from self._scalar_fields

    def clear(self) -> None:
        """
        Remove every object from the scene.
        """

        self._charges.clear()
        self._points.clear()
        self._vectors.clear()

    def remove(self, obj: object, ) -> None:
        """
        Remove an object from the scene.
        """
        if _charge.contains(obj):
            self._charges.remove(obj)
            return
        if _point.contains(obj):
            self._points.remove(obj)
            return
        if _vector.contains(obj):
            self._vectors.remove(obj)
            return

