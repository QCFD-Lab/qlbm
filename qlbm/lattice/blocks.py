from functools import cmp_to_key
from itertools import product
from json import dumps
from typing import Callable, Dict, List, Tuple

import numpy as np
from stl import mesh

from qlbm.lattice.spacetime.properties_base import SpaceTimeLatticeBuilder
from qlbm.tools.utils import bit_value, dimension_letter, flatten, get_qubits_to_invert


class SpaceTimeReflectionData:
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

    # def __get_neighbor_velocity_pairs_d1q2(
    #     self,
    # ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    #     return


class Block:
    """
    Contains information required for the generation of bounce-back and specular reflection boundary conditions for the :class:`.CQLBM` algorithm.
    A block can be constructed from minimal information, see the Table below.

    .. list-table:: Constructor parameters
        :widths: 25 50
        :header-rows: 1

        * - Parameter
          - Description
        * - :attr:`bounds`
          - A ``List[Tuple[int, int]]`` of lower and upper bounds in each dimension. For example, ``[(2, 5), (10, 12)]``; ``[(2, 5), (9, 12), (33, 70)]``.
        * - :attr:`num_qubits`
          - The number of grid qubits of the underlying lattice.
        * - :attr:`boundary_condition`
          - A string indicating the type of boundary condition of the block. Should be either ``"specular"`` or ``"bounceback"``.

    The :class:`.Block` constructor will parse this information and automatically infer all of the information
    required to perform all of the reflection edge cases.
    This data is split over several attributes, see the table below.

    .. list-table:: Class attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`bounds`
          - The ``List[Tuple[int, int]]`` of lower and upper bounds in each dimension. For example, ``[(2, 5), (10, 12)]``; ``[(2, 5), (9, 12), (33, 70)]``.
        * - :attr:`inside_points_data`
          - The ``List[Tuple[DimensionalReflectionData, ...]]`` data encoding the corner points on the `inside` of the obstacle. The outer list contains :math:`d` entries, one per dimension. Each entry is a tuple of :class:`DimensionalReflectionData` of the lower and upper bounds of that dimension, respectively.
        * - :attr:`outside_points_data`
          - The ``List[Tuple[DimensionalReflectionData, ...]]`` data encoding the corner points on the `outside` of the obstacle. The outer list contains :math:`d` entries, one per dimension. Each entry is a tuple of :class:`DimensionalReflectionData` of the lower and upper bounds of that dimension, respectively.
        * - :attr:`walls_inside`
          - The ``List[List[ReflectionWall, ...]]`` data encoding the walls of the `inside` of the obstacle. The outer list contains :math:`d` entries, one per dimension. Each inner list is a list of two :class:`ReflectionWall`\ s of the lower and upper bounds of that dimension, respectively.
        * - :attr:`walls_outside`
          - The ``List[List[ReflectionWall, ...]]`` data encoding the walls of the `outside` of the obstacle. The outer list contains :math:`d` entries, one per dimension. Each inner list is a list of two :class:`ReflectionWall`\ s of the lower and upper bounds of that dimension, respectively.
        * - :attr:`corners_inside`
          - The ``List[ReflectionPoint]`` data encoding the corner points on the `inside` of the obstacle. There are 4 inner corner :class:`ReflectionPoint`\ s per obstacle in 2D, and 8 in 3D.
        * - :attr:`corners_outside`
          - The ``List[ReflectionPoint]`` data encoding the corner points on the `outside` of the obstacle. There are 4 outer corner :class:`ReflectionPoint`\ s per obstacle in 2D, and 8 in 3D.
        * - :attr:`near_corner_points_2d`
          - The ``List[ReflectionPoint]`` data encoding points on the outside of the object that are adjacent to inner corner points. These points require additional logic in the quantum circuit for particles that have streamed without reflecting off the obstacle. Only applicable to 2D example, since 3D example use edges instead. There are 8 near-corner :class:`ReflectionPoint` \s per obstacle.
        * - :attr:`corner_edges_3d`
          - The ``List[ReflectionResetEdge]`` data encoding edges on the outside of the object that are adjacent to corners of the obstacle. Used to reset ancilla qubit states after reflection. There are 12 corner :class:`ReflectionResetEdge` \s per obstacle.
        * - :attr:`near_corner_edges_3d`
          - The ``List[ReflectionResetEdge]`` data encoding edges on the outside of the object that are adjacent either side of :attr:`corner_edges_3d`. These edges require additional logic in the quantum circuit for particles that have streamed without reflecting off the obstacle. There are 24 near-corner :class:`ReflectionResetEdge` \s per obstacle.
        * - :attr:`overlapping_near_corner_edge_points_3d`
          - The ``List[ReflectionPoint]`` data encoding the set of points at the intersections of :attr:`near_corner_edges_3d`. These points require additional logic in to account for the fact that the state of obstacle ancilla qubits was doubly reset (once by each edge). There are 24 such :class:`ReflectionPoint` \s per obstacle.
    """

    mesh_indices_list: List[np.ndarray] = [
        np.array([[0, 1, 2], [1, 2, 3]]),
        np.array(
            [
                [0, 1, 4],  # Face 1, facing +Y
                [1, 4, 5],
                [4, 5, 6],  # Face 2, right of Face 1
                [5, 6, 7],
                [2, 3, 6],  # Face 3, behind Face 1
                [3, 6, 7],
                [0, 1, 2],  # Face 4, left of Face 1
                [1, 2, 3],
                [1, 3, 5],  # Face 5, above
                [3, 5, 7],
                [0, 2, 4],  # Face 6, below
                [2, 4, 6],
            ]
        ),
    ]

    def __init__(
        self,
        bounds: List[Tuple[int, int]],
        num_qubits: List[int],
        boundary_condition: str,
    ) -> None:
        # TODO: check whether the number of dimensions is consistent
        self.bounds = bounds
        self.num_dims = len(bounds)
        self.boundary_condition = boundary_condition
        self.num_qubits = num_qubits
        if self.num_dims == 3:
            self.mesh_vertices = np.array(
                list(
                    product(*bounds)  # All combinations of bounds for 3D
                )
            )
        elif self.num_dims == 2:
            self.mesh_vertices = np.array(
                [
                    corner_point
                    + (
                        1,
                    )  # All combinations of bounds and (1) for the z dimension in 2D
                    for corner_point in list(product(*bounds))
                ]
            )
        else:  # 1D
            # self.mesh_vertices = np.array([(bounds[0][0], 0, 1), (bounds[0][1], 1, 1)])
            self.mesh_vertices = np.array(
                [
                    corner_point
                    + (
                        1,
                    )  # All combinations of bounds and (1) for the z dimension in 2D
                    for corner_point in list(product(*(bounds + [(0, 1)])))
                ]
            )

        self.mesh_indices = self.mesh_indices_list[max(0, self.num_dims - 2)]

        # The number of qubits used to offset "higher" dimensions
        self.previous_qubits: List[int] = [
            sum(num_qubits[previous_dim] for previous_dim in range(dim))
            for dim in range(self.num_dims)
        ]

        self.inside_points_data: List[Tuple[DimensionalReflectionData, ...]] = [
            tuple(
                DimensionalReflectionData(
                    [
                        self.previous_qubits[dim] + i
                        for i in range(num_qubits[dim])
                        if not bit_value(bounds[dim][bound_type], i)
                    ],
                    bound_type=bound_type,
                    is_outside_obstacle_bounds=False,
                    gridpoint_encoded=bounds[dim][bound_type],
                    dim=dim,
                    name=f"swap_{'l' if not bound_type else 'u'}{dimension_letter(dim)}_in",
                )
                for bound_type in [False, True]
            )
            for dim in range(self.num_dims)
        ]

        self.outside_points_data: List[Tuple[DimensionalReflectionData, ...]] = [
            tuple(
                DimensionalReflectionData(
                    [
                        self.previous_qubits[dim] + i
                        for i in range(num_qubits[dim])
                        if not bit_value(
                            bounds[dim][bound_type] + 2 * bound_type - 1, i
                        )  # 2 * bound - 1 results in +1 for upper bound (True) and -1 for lower bounds (False)
                    ],
                    bound_type=bound_type,
                    is_outside_obstacle_bounds=True,
                    gridpoint_encoded=bounds[dim][bound_type] + 2 * bound_type - 1,
                    dim=dim,
                    name=f"swap_{'l' if not bound_type else 'u'}{dimension_letter(dim)}_out",
                )
                for bound_type in [False, True]
            )
            for dim in range(self.num_dims)
        ]

        # List of inside walls ordered by dimension
        self.walls_inside: List[List[ReflectionWall]] = [
            [
                ReflectionWall(
                    dim,
                    [
                        bounds[wall_alignment_dim][0]
                        for wall_alignment_dim in wall_alignment_dims
                    ],  # The lower bounds of the obstacle in the alignment directions, i.e., ly and lz for x-walls
                    [
                        bounds[wall_alignment_dim][1]
                        for wall_alignment_dim in wall_alignment_dims
                    ],  # The upper bounds of the obstacle in the alignment directions, i.e., uy and uz for x-walls
                    self.inside_points_data[dim][
                        bound
                    ],  # The grid qubits to swap for the dimension and bound
                )
                for bound in [
                    False,
                    True,
                ]  # False pertains to lower bounds, True to upper bounds
            ]
            for dim, wall_alignment_dims in [
                (d, [dimension for dimension in range(self.num_dims) if dimension != d])
                for d in range(self.num_dims)
            ]  # `dim` is the dimension that the wall reflects (i.e., x-walls)
            # `wall_alignment_dims` are the dimensions that the wall spans
            # In 2D, an x-wall is aligned with dimension y=1
            # In 3D, an x-wall is aligned with dimensions y=1 and z=2
        ]

        # List of outside walls ordered by dimension
        # Identical to inside walls except of `outer_qubits_to_swap` is selected instead
        self.walls_outside: List[List[ReflectionWall]] = [
            [
                ReflectionWall(
                    dim,
                    [
                        bounds[wall_alignment_dim][0]
                        for wall_alignment_dim in wall_alignment_dims
                    ],
                    [
                        bounds[wall_alignment_dim][1]
                        for wall_alignment_dim in wall_alignment_dims
                    ],
                    self.outside_points_data[dim][bound],
                )
                for bound in [
                    False,
                    True,
                ]
            ]
            for dim, wall_alignment_dims in [
                (d, [dimension for dimension in range(self.num_dims) if dimension != d])
                for d in range(self.num_dims)
            ]
        ]

        self.corners_inside: List[ReflectionPoint] = [
            ReflectionPoint(
                list(inner_corner_point),
                dims_inside=list(range(self.num_dims)),
                dims_outside=[],
                inversion_function=lambda data: [data.invert_velocity for data in data],
            )
            for inner_corner_point in product(*self.inside_points_data)
        ]

        self.corners_outside: List[ReflectionPoint] = [
            ReflectionPoint(
                list(outer_corner_point),
                dims_inside=[],
                dims_outside=list(range(self.num_dims)),
                inversion_function=lambda data: [data.invert_velocity for data in data],
            )
            for outer_corner_point in product(*self.outside_points_data)
        ]

        self.near_corner_points_2d: List[ReflectionPoint] = flatten(
            [
                [
                    ReflectionPoint(
                        list(near_corner_point_dimensional_data),
                        dims_inside=[
                            dim_index
                            for dim_index, is_outside in enumerate(dim_data)
                            if not is_outside
                        ],
                        dims_outside=[
                            dim_index
                            for dim_index, is_outside in enumerate(dim_data)
                            if is_outside
                        ],
                        inversion_function=lambda data: [
                            (not data.bound_type)
                            if data.is_outside_obstacle_bounds
                            else (data.bound_type)
                            for data in data
                        ],
                    )
                    for near_corner_point_dimensional_data in product(
                        *(
                            self.outside_points_data[dim_index]
                            if is_outside
                            else self.inside_points_data[dim_index]
                            for dim_index, is_outside in enumerate(dim_data)
                        )
                    )
                ]
                for dim_data in [
                    [(dimension == d) for dimension in range(self.num_dims)]
                    for d in range(self.num_dims)
                ]
            ]
        )

        self.near_corner_edges_3d: List[ReflectionResetEdge] = (
            (self.__get_3d_near_corner_edges()) if self.num_dims > 2 else []
        )

        self.corner_edges_3d: List[ReflectionResetEdge] = (
            (self.__get_3d_corner_edges()) if self.num_dims > 2 else []
        )

        self.overlapping_near_corner_edge_points_3d: List[ReflectionPoint] = (
            self.__get_overlapping_near_corner_edge_points_3d()
            if self.num_dims > 2
            else []
        )

    def __get_overlapping_near_corner_edge_points_3d(self):
        # Whether lower (F) or upper (T) bound in a given dimension
        bound_type_wall = [False, True]
        bound_type_neighboring_walls = [False, True]

        # The two walls joining together
        dimension_outside = [
            dim_data
            for dim_data in [
                [(dimension == d) for dimension in range(3)] for d in range(3)
            ]
        ]

        points: List[ReflectionPoint] = []

        for point_data in product(
            dimension_outside,
            bound_type_wall,
            bound_type_neighboring_walls,
            bound_type_neighboring_walls,
        ):
            outside_dimension: int = [e[0] for e in enumerate(point_data[0]) if e[1]][0]
            neighboring_wall_dims: List[int] = [
                e[0] for e in enumerate(point_data[0]) if not e[1]
            ]

            point_outside = [self.outside_points_data[outside_dimension][point_data[1]]]
            points_inside = [
                self.inside_points_data[dim_and_bound[0]][dim_and_bound[1]]
                for dim_and_bound in zip(neighboring_wall_dims, point_data[-2:])
            ]

            sorted_point_data: List[DimensionalReflectionData] = sorted(
                point_outside + points_inside,
                key=cmp_to_key(lambda d1, d2: d1.dim - d2.dim),  # type: ignore
            )

            def overlapping_near_corner_edge_points_3d_inversion(
                sorted_data: List[DimensionalReflectionData],
            ) -> List[bool]:
                outside_dim_data = [
                    d for d in sorted_data if d.is_outside_obstacle_bounds
                ][0]

                return [
                    not d.bound_type if d == outside_dim_data else (d.bound_type)
                    for d in sorted_data
                ]

            points.append(
                ReflectionPoint(
                    data=sorted_point_data,
                    dims_inside=[d.dim for d in points_inside],
                    dims_outside=[point_outside[0].dim],
                    inversion_function=overlapping_near_corner_edge_points_3d_inversion,
                )
            )

        return points

    def __get_3d_corner_edges(self):
        # Whether lower (F) or upper (T) bound in a given dimension
        bound_types = [False, True]

        # The two walls joining together
        dimensions_joining = [
            dim_data
            for dim_data in [
                [(dimension != d) for dimension in range(3)] for d in [2, 1, 0]
            ]
        ]

        segments: List[ReflectionResetEdge] = []

        for edge_data in product(dimensions_joining, bound_types, bound_types):
            dim_index_in: int = [e[0] for e in enumerate(edge_data[0]) if not e[1]][0]
            dim_index_out = tuple(e[0] for e in enumerate(edge_data[0]) if e[1])
            walls_out: List[DimensionalReflectionData] = []

            for dim_bool, dim_index in enumerate(
                [e[0] for e in enumerate(edge_data[0]) if e[1]]
            ):
                walls_out.append(
                    self.outside_points_data[dim_index][edge_data[1 + dim_bool]]  # type: ignore
                )

            segments.append(
                ReflectionResetEdge(
                    walls_out,
                    dim_index_out,  # type: ignore
                    self.bounds[dim_index_in],
                    None,
                )
            )

        return segments

    def __get_3d_near_corner_edges(self):
        # Whether lower (F) or upper (T) bound in a given dimension
        bound_types = [False, True]

        # Whether the edge is orthogonal to the first (F) or second (T) dimension
        # For an x- and a z- wall, coming together, F
        # Would encode the edge directly adjacent (perpendicular to) the x-wall
        orthogonal_dim = [0, 1]

        # The two walls joining together
        dimensions_joining = [
            dim_data
            for dim_data in [
                [(dimension != d) for dimension in range(3)] for d in [2, 1, 0]
            ]
        ]

        segments: List[ReflectionResetEdge] = []

        for edge_data in product(
            dimensions_joining, bound_types, bound_types, orthogonal_dim
        ):
            dim_index_in: int = [e[0] for e in enumerate(edge_data[0]) if not e[1]][0]
            dim_index_out = tuple(e[0] for e in enumerate(edge_data[0]) if e[1])
            walls_joining: List[DimensionalReflectionData] = []

            for dim_bool, dim_index in enumerate(
                [
                    e[0] for e in enumerate(edge_data[0]) if e[1]
                ]  # Enumerate over the indices of the dimensions that join at this edge
            ):
                walls_joining.append(
                    self.outside_points_data[dim_index][edge_data[1 + dim_bool]]  # type: ignore
                    if dim_bool
                    == edge_data[3]  # If this is the dimension that is outside
                    else self.inside_points_data[dim_index][edge_data[1 + dim_bool]]  # type: ignore
                )

            segments.append(
                ReflectionResetEdge(
                    walls_joining,
                    dim_index_out,  # type: ignore
                    self.bounds[dim_index_in],
                    edge_data[3],  # type: ignore
                )
            )

        return segments

    def get_spacetime_reflection_data_d1q2(
        self,
        properties: SpaceTimeLatticeBuilder,
        num_steps: int | None = None,
    ) -> List[SpaceTimeReflectionData]:
        if num_steps is None:
            num_steps = properties.num_timesteps

        reflection_list = []
        for c, reflection_bound in enumerate(self.bounds[0]):
            for reflection_direction in [1 - c, c]:
                increment = properties.get_reflection_increments(reflection_direction)
                origin: Tuple[int, ...] = (reflection_bound, 0)
                for t in range(num_steps):
                    num_gps_in_dim = 2 ** self.num_qubits[0]
                    gridpoint_encoded = tuple(
                        a + (t + 1 if 1 - c == reflection_direction else t) * b
                        for a, b in zip(origin, increment)
                    )

                    # Add periodic boundary conditions
                    gridpoint_encoded = tuple(
                        x % num_gps_in_dim
                        if num_gps_in_dim
                        else (x + num_gps_in_dim if x < 0 else x)
                        for x in gridpoint_encoded
                    )
                    distance_from_origin = tuple((t + 1) * x for x in increment)
                    qubits_to_invert = get_qubits_to_invert(
                        gridpoint_encoded[0], self.num_qubits[0]
                    )

                    reflection_list.append(
                        SpaceTimeReflectionData(
                            gridpoint_encoded,
                            qubits_to_invert,
                            not bool(reflection_direction),
                            distance_from_origin,
                            properties,
                        )
                    )
        return reflection_list

    def get_spacetime_reflection_data_d2q4(
        self,
        properties: SpaceTimeLatticeBuilder,
        num_steps: int | None = None,
    ) -> List[SpaceTimeReflectionData]:
        if num_steps is None:
            num_steps = properties.num_timesteps

        reflection_list = []
        surfaces = self.get_d2q4_surfaces()

        # Surfaces are stored by dimension first
        for fixed_dim, surfaces_of_dim in enumerate(surfaces):
            # Each dimension has a lower bound and an upper bound surface
            for bound, wall_of_surface in enumerate(surfaces_of_dim):
                # fixed_dim_qubits_to_invert = get_qubits_to_invert(
                #     wall_of_surface[0][fixed_dim], self.num_qubits[fixed_dim]
                # )
                streaming_line_velocities = [0, 2] if fixed_dim == 0 else [1, 3]
                if not bound:
                    streaming_line_velocities = list(
                        reversed(streaming_line_velocities)
                    )
                symmetric_non_reflection_dimension_increment: List[
                    Tuple[int, Tuple[int, ...]]
                ] = [(0, (0, 0))] + flatten(
                    [
                        [
                            (
                                direction_factor * t,
                                tuple(
                                    a * direction_factor * t
                                    for a in properties.get_reflection_increments(
                                        1 if fixed_dim == 0 else 0
                                    )
                                ),
                            )
                            for t in range(1, num_steps)
                        ]
                        for direction_factor in [1, -1]
                    ]
                )

                # Each surface is made up of several gridpoints
                for gp_in_wall in wall_of_surface:
                    for reflection_direction in streaming_line_velocities:
                        increment = properties.get_reflection_increments(
                            reflection_direction
                        )

                        # Each gridpoint is part of the stencil of several others
                        for (
                            offset,
                            starting_increment,
                        ) in symmetric_non_reflection_dimension_increment:
                            origin: Tuple[int, ...] = tuple(
                                a + b for a, b in zip(gp_in_wall, starting_increment)
                            )
                            for t in range(num_steps - abs(offset)):
                                num_gps_in_dim = [2**n for n in self.num_qubits]
                                gridpoint_encoded = tuple(
                                    a
                                    + (
                                        t + 1
                                        if (
                                            streaming_line_velocities[0]
                                            == reflection_direction
                                        )
                                        else t
                                    )
                                    * b
                                    for a, b in zip(origin, increment)
                                )

                                # Add periodic boundary conditions
                                gridpoint_encoded = tuple(
                                    x + num_gps_in_dim if x < 0 else x % num_gps_in_dim
                                    for x, num_gps_in_dim in zip(
                                        gridpoint_encoded, num_gps_in_dim
                                    )
                                )
                                distance_from_origin = tuple(
                                    (t + 1) * x + y
                                    for x, y in zip(increment, starting_increment)
                                )

                                # The qubits to invert for this gridpoint
                                qubits_to_invert = flatten(
                                    [
                                        [
                                            self.previous_qubits[dim] + q
                                            for q in get_qubits_to_invert(
                                                gridpoint_encoded[dim],
                                                self.num_qubits[dim],
                                            )
                                        ]
                                        for dim in range(2)
                                    ]
                                )
                                opposite_reflection_direction = (
                                    streaming_line_velocities[1]
                                    if reflection_direction
                                    == streaming_line_velocities[0]
                                    else streaming_line_velocities[0]
                                )

                                reflection_list.append(
                                    SpaceTimeReflectionData(
                                        gridpoint_encoded,
                                        qubits_to_invert,
                                        opposite_reflection_direction,
                                        distance_from_origin,
                                        properties,
                                    )
                                )

        return reflection_list

    def get_d2q4_surfaces(self) -> List[List[List[Tuple[int, ...]]]]:
        surfaces: List[List[List[Tuple[int, ...]]]] = []

        for d, walls in enumerate(self.walls_inside):
            surfaces_of_dim: List[List[Tuple[int, ...]]] = []
            for wall in walls:
                surfaces_of_dim.append(
                    [
                        (wall.data.gridpoint_encoded, coordinate_across_wall)
                        if d == 0
                        else (coordinate_across_wall, wall.data.gridpoint_encoded)
                        for coordinate_across_wall in range(
                            wall.lower_bounds[0], wall.upper_bounds[0] + 1
                        )
                    ]
                )
            surfaces.append(surfaces_of_dim)

        return surfaces

    def contains_gridpoint(self, gridpoint: Tuple[int, ...]) -> bool:
        return all(
            coord >= bound[0] and coord <= bound[1]
            for coord, bound in zip(gridpoint, self.bounds)
        )

    def stl_mesh(self) -> mesh.Mesh:
        """
        Provides the ``stl`` representation of the block.

        Returns
        -------
        ``stl.mesh.Mesh``
            The mesh representing the block.
        """
        block = mesh.Mesh(np.zeros(self.mesh_indices.shape[0], dtype=mesh.Mesh.dtype))
        for i, triangle_face in enumerate(self.mesh_indices):
            for j in range(3):
                block.vectors[i][j] = self.mesh_vertices[triangle_face[j]]

        return block

    def to_json(self) -> str:
        """
        Serializes the block to JSON format.

        Returns
        -------
        str
            The JSON representation of the block.
        """
        return dumps(self.to_dict())

    def to_dict(self) -> Dict[str, List[int] | str]:
        """
        Produces a dictionary representation of the block.

        Returns
        -------
        Dict[str, List[int] | str]
            A dictionary representation of the bounds and boundary conditions of the block.
        """
        block_dict: Dict[str, List[int] | str] = {
            dimension_letter(numeric_dim_index): list(self.bounds[numeric_dim_index])
            for numeric_dim_index in range(self.num_dims)
        }
        block_dict["boundary"] = self.boundary_condition

        return block_dict


class DimensionalReflectionData:
    """
    Contains one-dimensional information about the position of a grid point relevant to the obstacle.
    Used for edge cases relating to either inside or outside corner points.

    .. list-table:: Class attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`qubits_to_invert`
          - The ``List[int]`` of qubit indices that should be inverted in order to convert the state of grid qubits encoding this dimension to :math:`\ket{1}^{\otimes n_{g_d}}`. See the example below.
        * - :attr:`bound_type`
          - The ``bool`` indicating the type of bound this point belongs to. ``False`` indicates a lower bound, and ``True`` indicates an upper bound.
        * - :attr:`is_outside_obstacle_bounds`
          - The ``bool`` indicating the whether the point belongs to the solid domain. ``False`` that the point is inside the solid domain, and ``True`` indicates the outside.
        * - :attr:`dim`
          - The ``int`` indicating which dimension this object refers to.
        * - :attr:`gridpoint_encoded`
          - The ``int`` indicating which grid point this object encodes. Used for debugging purposes.
        * - :attr:`name`
          - A string assigned to each dimensional data object in the :class:`Block` constructor. Used for debugging purposes.

    .. note::
       Consider for example encoding the grid point at location 2
       (encoded as :math:`\ket{010}`) on the :math:`x`-axis on an :math:`8\\times 8` 2D grid.

       The ``DimensionalReflectionData`` object encoding this information
       would have a ``qubits_to_invert`` value of ``[0, 2]``.
       This means that the :math:`0^{th}` and :math:`2^{nd}` qubits
       would have to be inverted to produce the :math:`\ket{111}` state.
       This information is passed on to the reflection operators,
       which place the :math:`X` gates at the appropriate positions in the register,
       and can then use the :math:`g_x` register to control reflection.

       If we wanted to encode point :math:`3` (:math:`\ket{011}`) on :math:`y`-axis on the same grid,
       this would result in ``qubits_to_invert = [5]``, since the most significant qubit (index 2 of the :math:`y`-axis)
       is encoded last in the register, and there are 3 qubits encoding the :math:`x`-axis "in front" of it.
    """

    def __init__(
        self,
        qubits_to_invert: List[int],
        bound_type: bool,
        is_outside_obstacle_bounds: bool,
        gridpoint_encoded: int,
        dim: int,
        name: str,
    ) -> None:
        self.qubits_to_invert = qubits_to_invert
        self.bound_type = bound_type
        self.is_outside_obstacle_bounds = is_outside_obstacle_bounds
        self.gridpoint_encoded = gridpoint_encoded
        self.dim = dim
        self.name = name
        # XOR between lower and outside gives the same truth table that decides whether to invert the velocities
        self.invert_velocity = self.bound_type ^ self.is_outside_obstacle_bounds


class ReflectionWall:
    """
    Encodes the information required to perform reflection on a wall.
    Each wall is encoded as fixed over one dimensions and spanning one or two `alignment` dimensions.
    This in turn models which qubits are used for the comparator operations of the reflection operators.
    The information required for the alignment dimensions only consists of bounds,
    while the fixed dimension uses its :class:`DimensionalReflectionData` representation.

    .. list-table:: Class attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`lower_bounds`
          - The ``List[int]`` of lower bounds for each dimension.
        * - :attr:`upper_bounds`
          - The ``List[int]`` of upper bounds for each dimension.
        * - :attr:`data`
          - The ``DimensionalReflectionData`` of the fixed dimension.
        * - :attr:`dim`
          - The ``int`` indicating the fixed dimension.
        * - :attr:`alignment_dims`
          - The ``List[int]`` indicating the one or two alignment dimensions.
        * - :attr:`bounceback_loose_bounds`
          - The ``List[List[bool]]`` indicating whether the comparators should span the dimensions using tight (i.e., :math:`<`) or loose (i.e., :math:`\leq`) bounds.
    """

    def __init__(
        self,
        dim: int,
        lower_bounds: List[int],
        upper_bounds: List[int],
        reflection_data: DimensionalReflectionData,
    ):
        self.num_dims = len(lower_bounds) + 1
        self.dim = dim
        self.alignment_dims: List[int] = [
            dimension for dimension in range(len(lower_bounds) + 1) if dimension != dim
        ]
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.data = reflection_data
        self.bounceback_loose_bounds: List[List[bool]] = (
            self.__get_bounceback_wall_loose_bounds()
        )

    def __get_bounceback_wall_loose_bounds(self):
        # Whether to use comparator bounds (LE, GE - True) or (LT, GT - False)
        # When performing BB reflection on the inside walls.
        # Organized as outer list -> dimension, inner list -> alignment dimensions
        # Many combinations are possible here
        if self.num_dims == 2:
            return [[True], [False]]
        else:
            return [[True, True], [False, False], [False, True]]


class ReflectionPoint:
    """
    Encodes the information required to perform reflection on a single point.
    A point is encoded as 2 or 3 fixed :class:`DimensionalReflectionData` objects, one per dimension.
    This classes processes the information encoded in the reflection data
    objects into boolean valued attributes that determine
    whether the directional velocity qubits should be inverted to perform reflection.

    .. list-table:: Class attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`data`
          - The ``List[DimensionalReflectionData]`` containing the point data for each dimension.
        * - :attr:`num_dims`
          - The ``int`` number of dimensions of this point (also of the corresponding lattice).
        * - :attr:`dims_inside`
          - The ``List[int]`` that specifies which of the ``data`` entries are `inside` obstacle bounds in their respective dimension.
        * - :attr:`dims_outside`
          - The ``List[int]`` that specifies which of the ``data`` entries are `outside` obstacle bounds in their respective dimension.
        * - :attr:`qubits_to_invert`
          -  The ``List[int]`` of qubit indices that should be inverted in order to convert the state of grid qubits to :math:`\ket{1}^{\otimes n_{g_d}}`.
        * - :attr:`inversion_function`
          - The ``Callable[[List[DimensionalReflectionData]], List[bool]]`` function that converts the input data into a list of booleans that determine whether the directional velocity qubits should be inverted, per dimensions.
        * - :attr:`invert_velocity_in_dimension`
          - The ``List[bool]`` obtained by calling the ``inversion_function`` on the ``data``, indicating whether the directional velocity qubits should be inverted, per dimensions.
        * - :attr:`is_near_corner_point`
          - The ``bool`` indicating whether the point is a near-corner point (used in 2D reflection).
    """

    def __init__(
        self,
        data: List[DimensionalReflectionData],
        dims_inside: List[int],
        dims_outside: List[int],
        inversion_function: Callable[[List[DimensionalReflectionData]], List[bool]],
    ):
        self.num_dims: int = len(data)
        self.qubits_to_invert: List[int] = flatten(
            [case.qubits_to_invert for case in data]
        )
        self.dims_inside = dims_inside
        self.dims_outside = dims_outside
        self.data = data
        # A point is near a corner if it is outside the obstacle in at least one bound, but not all
        self.is_near_corner_point = (len(dims_inside) != 0) and (len(dims_outside) != 0)
        self.invert_velocity_in_dimension: List[bool] = inversion_function(data)


class ReflectionResetEdge:
    """
    Encodes the information required to perform reflection on an edge in 3D.
    An edge is encoded as 2 fixed points as :class:`DimensionalReflectionData` objects, and one spanning dimension.
    This classes processes the information encoded in the reflection data
    object into boolean valued attributes that determine
    whether the directional velocity qubits should be inverted to perform reflection.

    .. list-table:: Class attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`walls_joining`
          - The ``List[DimensionalReflectionData]`` containing the point data for the two fixed dimensions, stored in ascending order (:math:`x < y < z`).
        * - :attr:`dims_of_edge`
          - The ``Tuple[int, int]`` indicating the fixed dimensions.
        * - :attr:`dim_disconnected`
          - The ``int`` indicating the dimension the edge spans.
        * - :attr:`bounds_disconnected_dim`
          - The ``Tuple[int, int]`` that specifies the bounds of the edge in the dimension that it spans.
        * - :attr:`reflected_velocities`
          - The ``List[int]`` that indicates to which dimensions the velocities that this edge affects belong to.
        * - :attr:`invert_velocity_in_dimension`
          - The ``List[bool]`` indicating whether the directional velocity qubits should be inverted, per dimensions.
        * - :attr:`is_corner_edge`
          - The ``bool`` indicating whether the edge is directly at the corner of the object.
    """

    def __init__(
        self,
        walls_joining: List[DimensionalReflectionData],
        dims_of_edge: Tuple[int, int],
        bounds_disconnected_dim: Tuple[int, int],
        dimension_outside: bool | None,
    ):
        self.walls_joining = walls_joining
        self.dims_of_edge = dims_of_edge
        self.dim_disconnected = [d for d in range(3) if d not in dims_of_edge][0]
        self.bounds_disconnected_dim = bounds_disconnected_dim
        self.dimension_outside = dimension_outside
        self.is_corner_edge = dimension_outside is None
        self.reflected_velocities: List[int] = (
            list(dims_of_edge)
            if self.is_corner_edge
            else [
                dims_of_edge[bool(dimension_outside)]  # type : ignore
            ]  # The velocity that the wall reflects is the one of the inside dimension
        )
        self.invert_velocity_in_dimension = (
            self.__get_corner_inversions(
                (walls_joining[0].bound_type, walls_joining[1].bound_type)
            )
            if self.is_corner_edge
            else self.__get_near_corner_inversions(
                (walls_joining[0].bound_type, walls_joining[1].bound_type),
                dimension_outside,  # type: ignore
            )
        )

    def __get_corner_inversions(
        self, dim_bounds: Tuple[bool, bool]
    ) -> Tuple[bool, bool]:
        return (not dim_bounds[0], not dim_bounds[1])

    def __get_near_corner_inversions(
        self, dim_bounds: Tuple[bool, bool], orthogonal: bool
    ) -> Tuple[bool, bool]:
        predicate: bool = dim_bounds[0] ^ dim_bounds[1]
        xor: bool = dim_bounds[0] ^ orthogonal

        if predicate:
            return (not xor, not xor)

        return (not xor, xor)
