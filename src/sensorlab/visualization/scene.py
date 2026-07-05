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
    VectorFieldPrimitive,
    ElectrodePrimitive
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
        self._vector_fields: list[VectorFieldPrimitive] = []
        self._scalar_fields: list[ScalarFieldPrimitive] = []
        self._electrodes: list[ElectrodePrimitive] = []


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
            
            case VectorFieldPrimitive():
                self._vector_fields.append(obj)
            
            case ElectrodePrimitive():
                self._electrodes.append(obj)

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

    def add_scalar_field(self, field: PotentialField, ) -> ScalarFieldPrimitive:
        """
        Add a scalar field to the scene.
        """
        self._scalar_fields.append(
            ScalarFieldPrimitive(field=field, )
        )
        return self._scalar_fields[-1]

    def add_vector_field(self, field: ElectricFieldGrid, ) -> VectorFieldPrimitive:
        """
        Add a vector field to the scene.
        """

        self._vector_fields.append(
            VectorFieldPrimitive(field=field, )
        )

        return self._vector_fields[-1]
    
    def add_electrode(self, electrode: RectangleElectrode, name: str | None = None) -> None:
        """
        Add an electrode to the scene.
        """
        for rectangle in electrode.rectangles:
            self._electrodes.append(
                ElectrodePrimitive(
                    center=rectangle.center,
                    width=rectangle.width,
                    height=rectangle.height,
                    potential=electrode.potential,
                    name=name,
                )
            )
        

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

    def vector_fields(self, ) -> Iterator[VectorFieldPrimitive]:
        yield from self._vector_fields

    def electrodes(self) -> Iterator[ElectrodePrimitive]:
        """
        Iterate over all electrodes in the scene.
        """
        yield from self._electrodes

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

