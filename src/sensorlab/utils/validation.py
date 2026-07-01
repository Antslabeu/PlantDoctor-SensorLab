from sensorlab.physics.materials import Material


def validate_material(material: Material):

    if material.relative_permittivity < 1:
        raise ValueError(
            "Relative permittivity cannot be below vacuum."
        )

    if material.conductivity < 0:
        raise ValueError(
            "Conductivity cannot be negative."
        )