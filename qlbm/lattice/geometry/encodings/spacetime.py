"""Geometrical data encodings specific to the :class:`STQLBM` algorithm."""

from typing import List, Tuple

from qlbm.lattice.spacetime.properties_base import SpaceTimeLatticeBuilder


class SpaceTimeReflectionData:
    r"""
    Class encoding the necessary information for the reflection of a particle in the :class:`.STQBLM` algorithm.

    ==================================== =======================================================================
    Attribute                            Summary
    ==================================== =======================================================================
    :attr:`gridpoint_encoded`            The gridpoint encoded in the data.
    :attr:`qubits_to_invert`             The grid qubit indices that have the value :math:`\ket{0}`.
    :attr:`velocity_index_to_reflect`    The index of the qubit encoding the discrete velocity that should be reflected.
    :attr:`distance_from_boundary_point` The distance from the gridpoints where reflection takes place.
    :attr:`lattice_properties`           The properties of the lattice in which reflection takes place.
    ==================================== =======================================================================
    """

    def __init__(
        self,
        gridpoint_encoded: Tuple[int, ...],
        qubits_to_invert: List[int],
        velocity_index_to_reflect: int,
        distance_from_boundary_point: Tuple[int, ...],
        lattice_properties: SpaceTimeLatticeBuilder,
    ) -> None:
        self.gridpoint_encoded = gridpoint_encoded
        self.qubits_to_invert = qubits_to_invert
        self.velocity_index_to_reflect = velocity_index_to_reflect
        self.distance_from_boundary_point = distance_from_boundary_point
        self.reversed_distance_from_boundary_point = tuple(
            -x for x in distance_from_boundary_point
        )
        self.lattice_properties = lattice_properties
        self.neighbor_velocity_pairs: Tuple[Tuple[int, int], Tuple[int, int]] = (
            self.__get_neighbor_velocity_pairs()
        )

    def __get_neighbor_velocity_pairs(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        reflected_velocity_index = (
            self.lattice_properties.get_reflected_index_of_velocity(
                self.velocity_index_to_reflect
            )
        )
        increment = self.lattice_properties.get_reflection_increments(
            reflected_velocity_index
        )

        neighbor_of_reflection = self.lattice_properties.get_index_of_neighbor(
            self.reversed_distance_from_boundary_point
        )

        neighbor_of_streamed_particle = self.lattice_properties.get_index_of_neighbor(
            tuple(
                a + b
                for a, b in zip(self.reversed_distance_from_boundary_point, increment)
            )
        )

        return (
            (neighbor_of_streamed_particle, reflected_velocity_index),
            (neighbor_of_reflection, self.velocity_index_to_reflect),
        )


class SpaceTimeVolumetricReflectionData:
    r"""
    Class encoding the necessary information for the reflection of a volumetric split of particles in the :class:`.STQBLM` algorithm.

    ========================================= =======================================================================
    Attribute                                 Summary
    ========================================= =======================================================================
    :attr:`fixed_dim`                         The physical dimension that the volume does not span.
    :attr:`ranged_dims`                       The physical dimension(s) that the volume does span.
    :attr:`range_dimension_bounds`            The bounds of the volume in each ranged dimension.
    :attr:`fixed_dimension_qubits_to_invert`  The grid qubit indices that have the value :math:`\ket{0}` for the fixed dimension.
    :attr:`fixed_gridpoint`                   The numerical value of the fixed gridpoint. Used for debugging purposes.
    :attr:`velocity_index_to_reflect`         The index of the qubit encoding the discrete velocity that should be reflected.
    :attr:`distance_from_boundary_wall`       The distance from the gridpoints where reflection takes place.
    :attr:`lattice_properties`                The properties of the lattice in which reflection takes place.
    ========================================= =======================================================================
    """

    def __init__(
        self,
        fixed_dim: int,
        ranged_dims: List[int],
        range_dimension_bounds: List[Tuple[int, int]],
        fixed_dimension_qubits_to_invert: List[int],
        fixed_gridpoint: int,
        velocity_index_to_reflect: int,
        distance_from_boundary_wall: Tuple[int, ...],
        lattice_properties: SpaceTimeLatticeBuilder,
    ) -> None:
        self.fixed_dim = fixed_dim
        self.ranged_dims = ranged_dims
        self.range_dimension_bounds = range_dimension_bounds
        self.fixed_dimension_qubits_to_invert = fixed_dimension_qubits_to_invert
        self.fixed_gridpoint = fixed_gridpoint
        self.velocity_index_to_reflect = velocity_index_to_reflect
        self.distance_from_boundary_point = distance_from_boundary_wall
        self.reversed_distance_from_boundary_point = tuple(
            -x for x in distance_from_boundary_wall
        )
        self.lattice_properties = lattice_properties
        self.neighbor_velocity_pairs: Tuple[Tuple[int, int], Tuple[int, int]] = (
            self.__get_neighbor_velocity_pairs()
        )

    def __get_neighbor_velocity_pairs(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        reflected_velocity_index = (
            self.lattice_properties.get_reflected_index_of_velocity(
                self.velocity_index_to_reflect
            )
        )
        increment = self.lattice_properties.get_reflection_increments(
            reflected_velocity_index
        )

        neighbor_of_reflection = self.lattice_properties.get_index_of_neighbor(
            self.reversed_distance_from_boundary_point
        )

        neighbor_of_streamed_particle = self.lattice_properties.get_index_of_neighbor(
            tuple(
                a + b
                for a, b in zip(self.reversed_distance_from_boundary_point, increment)
            )
        )

        return (
            (neighbor_of_streamed_particle, reflected_velocity_index),
            (neighbor_of_reflection, self.velocity_index_to_reflect),
        )
