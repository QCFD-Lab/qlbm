from functools import cmp_to_key
from itertools import product
from json import dumps
from typing import Callable, Dict, List, Tuple

import numpy as np
from stl import mesh

from qlbm.tools.utils import bit_value, dimension_letter, flatten


class Block:
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
    ) -> None:
        # TODO: check whether the number of dimensions is consistent
        self.bounds = bounds
        self.num_dims = len(bounds)
        self.mesh_vertices = np.array(
            list(
                product(*bounds)  # All combinations of bounds for 3D
            )
            if self.num_dims > 2
            else [
                corner_point
                + (1,)  # All combinations of bounds and (1) for the z dimension in 2D
                for corner_point in list(product(*bounds))
            ]
        )

        self.mesh_indices = self.mesh_indices_list[self.num_dims - 2]

        # The number of qubits used to offset "higher" dimensions
        previous_qubits: List[int] = [
            sum(num_qubits[previous_dim] for previous_dim in range(dim))
            for dim in range(self.num_dims)
        ]

        self.inside_points_data: List[Tuple[DimensionalReflectionData, ...]] = [
            tuple(
                DimensionalReflectionData(
                    [
                        previous_qubits[dim] + i
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
                        previous_qubits[dim] + i
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

    def stl_mesh(self):
        block = mesh.Mesh(np.zeros(self.mesh_indices.shape[0], dtype=mesh.Mesh.dtype))
        for i, triangle_face in enumerate(self.mesh_indices):
            for j in range(3):
                block.vectors[i][j] = self.mesh_vertices[triangle_face[j]]

        return block

    def to_json(self, boundary_type: str) -> str:
        return dumps(self.to_dict(boundary_type))

    def to_dict(self, boundary_type: str) -> Dict[str, List[int] | str]:
        block_dict: Dict[str, List[int] | str] = {
            dimension_letter(numeric_dim_index): list(self.bounds[numeric_dim_index])
            for numeric_dim_index in range(self.num_dims)
        }
        block_dict["boundary"] = boundary_type

        return block_dict


class DimensionalReflectionData:
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


class ReflectionPoint:
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
