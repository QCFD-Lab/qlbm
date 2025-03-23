"""Implementation of cuboid data structure."""

from functools import cmp_to_key
from itertools import product
from json import dumps
from typing import Dict, List, Tuple, cast, override

import numpy as np
from stl import mesh

from qlbm.lattice.geometry.encodings.collisionless import (
    DimensionalReflectionData,
    ReflectionPoint,
    ReflectionResetEdge,
    ReflectionWall,
)
from qlbm.lattice.geometry.encodings.spacetime import (
    SpaceTimePWReflectionData,
    SpaceTimeVolumetricReflectionData,
)
from qlbm.lattice.geometry.shapes.base import SpaceTimeShape
from qlbm.lattice.spacetime.properties_base import SpaceTimeLatticeBuilder
from qlbm.tools.utils import bit_value, dimension_letter, flatten, get_qubits_to_invert


class Block(SpaceTimeShape):
    r"""
    Contains information required for the generation of boundary conditions for an axis-parallel cuboid obstacle.

    Available for the specular reflection and bounce-back for the :class:`.CQLBM` algorithm and bounce-back for the :class:`.STQBM` algorithm.
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
        num_grid_qubits: List[int],
        boundary_condition: str,
    ) -> None:
        super().__init__(num_grid_qubits, boundary_condition)
        # TODO: check whether the number of dimensions is consistent
        self.bounds: List[Tuple[int, int]] = bounds
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

        self.inside_points_data: List[Tuple[DimensionalReflectionData, ...]] = [
            tuple(
                DimensionalReflectionData(
                    [
                        self.previous_qubits[dim] + i
                        for i in range(num_grid_qubits[dim])
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
                        for i in range(num_grid_qubits[dim])
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

    @override
    def get_spacetime_reflection_data_d1q2(
        self,
        properties: SpaceTimeLatticeBuilder,
        num_steps: int | None = None,
    ) -> List[SpaceTimePWReflectionData]:
        if num_steps is None:
            num_steps = properties.num_timesteps

        reflection_list = []
        for c, reflection_bound in enumerate(self.bounds[0]):
            for reflection_direction in [1 - c, c]:
                increment = properties.get_reflection_increments(reflection_direction)
                origin: Tuple[int, ...] = (reflection_bound, 0)
                for t in range(num_steps):
                    num_gps_in_dim = 2 ** self.num_grid_qubits[0]
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
                        gridpoint_encoded[0], self.num_grid_qubits[0]
                    )

                    reflection_list.append(
                        SpaceTimePWReflectionData(
                            gridpoint_encoded,
                            qubits_to_invert,
                            [not bool(reflection_direction)],
                            distance_from_origin,
                            properties,
                        )
                    )
        return reflection_list

    @override
    def get_spacetime_reflection_data_d2q4(
        self,
        properties: SpaceTimeLatticeBuilder,
        num_steps: int | None = None,
    ) -> List[SpaceTimePWReflectionData]:
        if num_steps is None:
            num_steps = properties.num_timesteps
        reflection_list: List[SpaceTimePWReflectionData] = []
        surfaces = self.get_d2q4_surfaces()

        # Surfaces are stored by dimension first
        for fixed_dim, surfaces_of_dim in enumerate(surfaces):
            # Each dimension has a lower bound and an upper bound surface
            for bound, wall_of_surface in enumerate(surfaces_of_dim):
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

                reflection_list.extend(
                    self.get_spacetime_reflection_data_d2q4_from_points(
                        properties,
                        wall_of_surface,
                        streaming_line_velocities,
                        symmetric_non_reflection_dimension_increment,
                        num_steps,
                    )
                )

        return reflection_list

    @override
    def get_d2q4_volumetric_reflection_data(
        self,
        properties: SpaceTimeLatticeBuilder,
        num_steps: int | None = None,
    ) -> List[SpaceTimeVolumetricReflectionData]:
        if num_steps is None:
            num_steps = properties.num_timesteps

        reflection_list = []
        num_gps_in_dim = [2**n for n in self.num_grid_qubits]
        for fixed_dim, bounds_of_fixed_dim in enumerate(self.bounds):
            ranged_dim = 1 - fixed_dim
            bounds_ranged_dim: Tuple[int, int] = self.bounds[ranged_dim]
            for bound in [False, True]:
                streaming_line_velocities = [0, 2] if fixed_dim == 0 else [1, 3]
                fixed_dimension_gp = bounds_of_fixed_dim[bound]
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

                for reflection_direction in streaming_line_velocities:
                    increment = properties.get_reflection_increments(
                        reflection_direction
                    )

                    opposite_reflection_direction = (
                        streaming_line_velocities[1]
                        if reflection_direction == streaming_line_velocities[0]
                        else streaming_line_velocities[0]
                    )
                    for (
                        offset,
                        starting_increment,
                    ) in symmetric_non_reflection_dimension_increment:
                        fixed_dimension_gp_adjusted = (
                            fixed_dimension_gp + starting_increment[fixed_dim]
                        )
                        bounds_ranged_dim_adjusted = [
                            b + starting_increment[ranged_dim]
                            for b in bounds_ranged_dim
                        ]
                        for t in range(num_steps - abs(offset)):
                            fixed_dim_reflection = (
                                fixed_dimension_gp_adjusted
                                + (
                                    t + 1
                                    if (
                                        streaming_line_velocities[0]
                                        == reflection_direction
                                    )
                                    else t
                                )
                                * increment[fixed_dim]
                            )
                            fixed_dim_reflection = (
                                fixed_dim_reflection + num_gps_in_dim[fixed_dim]
                                if fixed_dim_reflection < 0
                                else fixed_dim_reflection % num_gps_in_dim[fixed_dim]
                            )

                            # Ranged dimension bounds are left unadjusted
                            # As they are corrected in the quantum component
                            # Responsible for placing the comparators and controls
                            bounds_ranged_dim_reflection = cast(
                                Tuple[int, int],
                                tuple(
                                    b
                                    + (
                                        t + 1
                                        if (
                                            streaming_line_velocities[0]
                                            == reflection_direction
                                        )
                                        else t
                                    )
                                    * increment[ranged_dim]
                                    for b in bounds_ranged_dim_adjusted
                                ),
                            )
                            opposite_reflection_direction = (
                                streaming_line_velocities[1]
                                if reflection_direction == streaming_line_velocities[0]
                                else streaming_line_velocities[0]
                            )
                            reflection_list.append(
                                SpaceTimeVolumetricReflectionData(
                                    fixed_dim,
                                    [ranged_dim],
                                    [bounds_ranged_dim_reflection],
                                    get_qubits_to_invert(
                                        fixed_dim_reflection,
                                        self.num_grid_qubits[fixed_dim],
                                    ),
                                    fixed_dim_reflection,
                                    opposite_reflection_direction,
                                    tuple(
                                        (t + 1) * x + y
                                        for x, y in zip(increment, starting_increment)
                                    ),
                                    properties,
                                )
                            )
        return reflection_list

    def get_d2q4_surfaces(self) -> List[List[List[Tuple[int, ...]]]]:
        """
        Get all surfaces of the block in 2 dimensions.

        The information is formatted as ``List[List[List[Tuple[int, ...]]]]``.
        The outermost list is by dimension.
        The middle list contains two lists pertaining to the lower and upper bounds of the block in that dimenison.
        The innermost list contains the gridpoints that make up the surface encoded as tuples.

        Returns
        -------
        List[List[List[Tuple[int, ...]]]]
            The block surfaces in two dimensions.
        """
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

    @override
    def contains_gridpoint(self, gridpoint: Tuple[int, ...]) -> bool:
        return all(
            coord >= bound[0] and coord <= bound[1]
            for coord, bound in zip(gridpoint, self.bounds)
        )

    @override
    def stl_mesh(self) -> mesh.Mesh:
        block = mesh.Mesh(np.zeros(self.mesh_indices.shape[0], dtype=mesh.Mesh.dtype))
        for i, triangle_face in enumerate(self.mesh_indices):
            for j in range(3):
                block.vectors[i][j] = self.mesh_vertices[triangle_face[j]]

        return block

    @override
    def to_json(self) -> str:
        return dumps(self.to_dict())

    @override
    def name(self) -> str:
        return "cuboid"

    @override
    def to_dict(self) -> Dict[str, List[int] | str]:
        block_dict: Dict[str, List[int] | str] = {
            dimension_letter(numeric_dim_index): list(self.bounds[numeric_dim_index])
            for numeric_dim_index in range(self.num_dims)
        }
        block_dict["boundary"] = self.boundary_condition
        block_dict["shape"] = self.name()

        return block_dict
