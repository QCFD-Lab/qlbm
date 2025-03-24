"""Implementation of circle data structure."""

from json import dumps
from typing import Dict, List, Tuple, cast, override

import numpy as np
from stl import mesh

from qlbm.lattice.geometry.encodings.spacetime import (
    SpaceTimePWReflectionData,
)
from qlbm.lattice.geometry.shapes.base import SpaceTimeShape
from qlbm.lattice.spacetime.properties_base import SpaceTimeLatticeBuilder
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import flatten


class Circle(SpaceTimeShape):
    """
    Contains information required for the generation of bounce-back boundary conditions for the :class:`.STQBM` algorithm.

    A circle can be constructed from minimal information, see the Table below.

    .. list-table:: Constructor parameters
        :widths: 25 50
        :header-rows: 1

        * - Parameter
          - Description
        * - :attr:`center`
          - A ``Tuple[int, ...]`` specifying the center of the circle. For example, ``(2, 5)``.
        * - :attr:`radius`
          - An ``int`` specifying the radius of the circle. For example, ``3``.
        * - :attr:`num_grid_qubits`
          - The number of grid qubits of the underlying lattice.
        * - :attr:`boundary_condition`
          - A ``string`` indicating the type of boundary condition of the block. At the moment, only ``"bounceback"`` is supported.
        * - :attr:`num_mesh_segments`
          - An ``int`` that describes how fine the ``stl`` of the object is.

    The :class:`.Circle` constructor will parse this information and automatically infer all of the information
    required to perform all of the reflection edge cases. Class attributes are described in the table below.

    .. list-table:: Class attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`perimeter_points`
          - The ``List[Tuple[int, int]]`` of all gridpoints that lie on the perimeter of the circle, and are therefore relevant for boundary conditions.

    """

    def __init__(
        self,
        center: Tuple[int, ...],
        radius: int,
        num_grid_qubits: List[int],
        boundary_condition: str,
        num_mesh_segments: int = 50,
    ):
        super().__init__(num_grid_qubits, boundary_condition)
        self.center = center
        self.radius = radius
        self.num_mesh_segments = num_mesh_segments
        self.perimeter_points = self.get_circle_perimeter()

    def get_circle_perimeter(self) -> List[Tuple[int, int]]:
        """
        Uses Bresenham's circle drawing algorithm to specify all points along the perimeter of the circle.

        Returns
        -------
        List[Tuple[int, int]]
            All gridpoints that lie on the perimeter of the circle
        """
        points = set()
        x, y = self.radius, 0

        d = 1 - self.radius
        while x >= y:
            points.update(
                {
                    (self.center[0] + x, self.center[1] + y),
                    (self.center[0] - x, self.center[1] + y),
                    (self.center[0] + x, self.center[1] - y),
                    (self.center[0] - x, self.center[1] - y),
                    (self.center[0] + y, self.center[1] + x),
                    (self.center[0] - y, self.center[1] + x),
                    (self.center[0] + y, self.center[1] - x),
                    (self.center[0] - y, self.center[1] - x),
                }
            )
            y += 1
            if d <= 0:
                d += 2 * y + 1
            else:
                x -= 1
                d += 2 * (y - x) + 1

        return sorted(points)

    @override
    def stl_mesh(self):
        vertices = np.array([(x, y, 0) for x, y in self.perimeter_points])
        faces = []

        num_points = len(self.perimeter_points)
        for i in range(num_points):
            for j in range(i + 1, num_points):
                for k in range(j + 1, num_points):
                    faces.append([i, j, k])

        faces = np.array(faces)  # type: ignore

        surface = mesh.Mesh(
            np.zeros(
                faces.shape[0],  # type: ignore
                dtype=mesh.Mesh.dtype,
            )
        )
        for i, f in enumerate(faces):
            for j in range(3):
                surface.vectors[i][j] = vertices[f[j]]

        return surface

    def __stl_mesh_smooth(self):
        angles = np.linspace(0, 2 * np.pi, self.num_mesh_segments, endpoint=False)
        top_vertices = np.array(
            [
                [
                    self.center[0] + self.radius * np.cos(a),
                    self.center[1] + self.radius * np.sin(a),
                    0.5,
                ]
                for a in angles
            ]
        )
        bottom_vertices = np.array(
            [
                [
                    self.center[0] + self.radius * np.cos(a),
                    self.center[1] + self.radius * np.sin(a),
                    -0.5,
                ]
                for a in angles
            ]
        )

        top_center = np.array([self.center[0], self.center[1], 0.5])
        bottom_center = np.array([self.center[0], self.center[1], -0.5])

        faces = []
        for i in range(self.num_mesh_segments):
            next_i = (i + 1) % self.num_mesh_segments

            # Top face
            faces.append([top_center, top_vertices[i], top_vertices[next_i]])

            # Bottom face
            faces.append([bottom_center, bottom_vertices[next_i], bottom_vertices[i]])

            # Side faces
            faces.append([top_vertices[i], bottom_vertices[i], bottom_vertices[next_i]])
            faces.append(
                [top_vertices[i], bottom_vertices[next_i], top_vertices[next_i]]
            )

        # Convert to numpy array
        faces_np = np.array(faces)

        # Create the mesh
        circle_mesh = mesh.Mesh(np.zeros(faces_np.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces_np):
            circle_mesh.vectors[i] = f

        return circle_mesh

    def is_point_on_segment(
        self, gridpoint: Tuple[int, int], segment: List[Tuple[int, int]]
    ) -> bool:
        """
        Whether the point belongs to a given axis-aligned segment.

        Parameters
        ----------
        gridpoint : Tuple[int, int]
            The gridpoint to test for.
        segment : List[Tuple[int, int]]
            The segment to test for.

        Returns
        -------
        bool
            Whether the point belongs to a given axis-aligned segment.
        """
        return min(segment[0][0], segment[1][0]) <= gridpoint[0] <= max(
            segment[0][0], segment[1][0]
        ) and min(segment[0][1], segment[1][1]) <= gridpoint[1] <= max(
            segment[0][1], segment[1][1]
        )

    def is_point_on_any_segment(
        self,
        gridpoint: Tuple[int, int],
        segments: List[List[Tuple[int, int]]],
    ) -> bool:
        """
        Whether the point belongs to any axis-aligned segment in a given list.

        Parameters
        ----------
        gridpoint : Tuple[int, int]
            The gridpoint to test for.
        segments : List[List[Tuple[int, int]]]
            The segments to test for.

        Returns
        -------
        bool
            Whether the point lays on any of the given segments.
        """
        return any(self.is_point_on_segment(gridpoint, s) for s in segments)

    def split_perimeter_points(
        self, points: List[Tuple[int, int]]
    ) -> Tuple[
        List[List[Tuple[int, int]]], List[List[Tuple[int, int]]], List[Tuple[int, int]]
    ]:
        """
        Splits point on the perimeter of the circle into three categories.

        Points belong to either
        (1) axis-aligned segments
        (2) diagonal segments, or
        (3) individual points.

        Axis-aligned and diagonal segments are encoded as the two ends of the segment.
        Individual points are simply listed by their coordinates.
        The results are returned in the order listed previously.

        Parameters
        ----------
        points : List[Tuple[int, int]]
            The gridpoints on the perimeter of the circle.

        Returns
        -------
        Tuple[List[Tuple[int, int]], List[List[Tuple[int, int]]], List[Tuple[int, int]]]
            The points classified by which category they belong to.
        """
        axis_segments = []
        diagonal_segments: List[List[Tuple[int, int]]] = []
        row_groups: Dict[int, List[int]] = {}
        col_groups: Dict[int, List[int]] = {}
        primary_diag_groups: Dict[int, List[Tuple[int, int]]] = {}  # x - y constant
        secondary_diag_groups: Dict[int, List[Tuple[int, int]]] = {}  # x + y constant

        for x, y in points:
            if x not in col_groups:
                col_groups[x] = []
            if y not in row_groups:
                row_groups[y] = []

            primary_diag_key = x - y
            secondary_diag_key = x + y

            if primary_diag_key not in primary_diag_groups:
                primary_diag_groups[primary_diag_key] = []
            if secondary_diag_key not in secondary_diag_groups:
                secondary_diag_groups[secondary_diag_key] = []

            row_groups[y].append(x)
            col_groups[x].append(y)
            primary_diag_groups[primary_diag_key].append((x, y))
            secondary_diag_groups[secondary_diag_key].append((x, y))

        # Horizontal segments
        for y, x_vals in row_groups.items():
            x_vals.sort()
            segment_start = x_vals[0]
            for i in range(1, len(x_vals)):
                if x_vals[i] != x_vals[i - 1] + 1:
                    if segment_start + 1 < x_vals[i - 1]:  # Trim edges
                        axis_segments.append(
                            [(segment_start + 1, y), (x_vals[i - 1] - 1, y)]
                        )
                    segment_start = x_vals[i]
            if segment_start + 1 < x_vals[-1]:
                axis_segments.append([(segment_start + 1, y), (x_vals[-1] - 1, y)])

        # Vertical segments
        for x, y_vals in col_groups.items():
            y_vals.sort()
            segment_start = y_vals[0]
            for i in range(1, len(y_vals)):
                if y_vals[i] != y_vals[i - 1] + 1:
                    if segment_start + 1 < y_vals[i - 1]:  # Trim edges
                        axis_segments.append(
                            [(x, segment_start + 1), (x, y_vals[i - 1] - 1)]
                        )
                    segment_start = y_vals[i]
            if segment_start + 1 < y_vals[-1]:
                axis_segments.append([(x, segment_start + 1), (x, y_vals[-1] - 1)])

        # Primary diagonal segments (x - y constant)
        for _, diag_points in primary_diag_groups.items():
            diag_points.sort()
            segment_start = diag_points[0]  # type: ignore
            for i in range(1, len(diag_points)):
                if (
                    diag_points[i][0] != diag_points[i - 1][0] + 1
                    or diag_points[i][1] != diag_points[i - 1][1] + 1
                ):
                    diagonal_segments.append([segment_start, diag_points[i - 1]])  # type: ignore
                    segment_start = diag_points[i]  # type: ignore
            diagonal_segments.append([segment_start, diag_points[-1]])  # type: ignore

        # Secondary diagonal segments (x + y constant)
        for _, diag_points in secondary_diag_groups.items():
            diag_points.sort()
            segment_start = diag_points[0]  # type: ignore
            for i in range(1, len(diag_points)):
                if (
                    diag_points[i][0] != diag_points[i - 1][0] + 1
                    or diag_points[i][1] != diag_points[i - 1][1] - 1
                ):
                    diagonal_segments.append([segment_start, diag_points[i - 1]])  # type: ignore
                    segment_start = diag_points[i]  # type: ignore
            diagonal_segments.append([segment_start, diag_points[-1]])  # type: ignore

        # Remove zero-length segments
        axis_segments = [seg for seg in axis_segments if seg[0] != seg[1]]
        diagonal_segments = [seg for seg in diagonal_segments if seg[0] != seg[1]]

        # Identify individual points that are not part of any segment
        individual_points = [
            ip
            for ip in points
            if not self.is_point_on_any_segment(ip, axis_segments + diagonal_segments)
        ]

        return axis_segments, diagonal_segments, individual_points

    @override
    def to_json(self):
        return dumps(self.to_dict())

    @override
    def name(self):
        return "sphere"

    @override
    def to_dict(self):
        return {
            "shape": self.name(),
            "center": list(self.center),
            "radius": self.radius,
            "boundary": self.boundary_condition,
        }

    @override
    def contains_gridpoint(self, gridpoint: Tuple[int, ...]) -> bool:
        return (gridpoint in self.perimeter_points) or (
            gridpoint[0] - self.center[0]
        ) * (gridpoint[0] - self.center[0]) + (gridpoint[1] - self.center[1]) * (
            gridpoint[1] - self.center[1]
        ) <= self.radius * self.radius

    @override
    def get_spacetime_reflection_data_d1q2(self, properties, num_steps=None):
        raise LatticeException("Circles unsupported in D1Q2 lattice.")

    def __symmetric_increment(
        self, num_steps: int, fixed_dim: int, properties: SpaceTimeLatticeBuilder
    ) -> List[Tuple[int, Tuple[int, ...]]]:
        return [(0, (0, 0))] + flatten(
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

    @override
    def get_spacetime_reflection_data_d2q4(
        self, properties, num_steps=None
    ) -> List[SpaceTimePWReflectionData]:
        if num_steps is None:
            num_steps = properties.num_timesteps

        reflection_list = []
        axis_segments, diag_segments, individual_points = self.split_perimeter_points(
            self.perimeter_points
        )

        # First compressing and then expanding the gridpoints is extremely inefficient.
        for segment in axis_segments:
            expanded_gridpoints = Circle.expand_axis_segments([segment])
            fixed_dim = [d for d in [0, 1] if segment[0][d] == segment[1][d]][0]
            ranged_dim = 1 - fixed_dim
            streaming_line_velocities = [0, 2] if fixed_dim == 0 else [1, 3]

            # If larger than the center, then it is an upper bound
            bound = segment[0][ranged_dim] > self.center[ranged_dim]

            if not bound:
                streaming_line_velocities = list(reversed(streaming_line_velocities))

            reflection_list.extend(
                self.get_spacetime_reflection_data_d2q4_from_points(
                    properties,
                    expanded_gridpoints,
                    streaming_line_velocities,
                    self.__symmetric_increment(num_steps, fixed_dim, properties),
                    num_steps,
                )
            )

        for diag_segment in diag_segments:
            expanded_diag_gridpoints: List[Tuple[int, ...]] = (
                Circle.expand_diagonal_segments([diag_segment])
            )

            if (4, 1) in expanded_diag_gridpoints:
                print("ok!")
            for reflection_dim in [0, 1]:
                other_dim = 1 - reflection_dim
                streaming_line_velocities = [0, 2] if reflection_dim == 0 else [1, 3]
                # If larger than the center, then it is an upper bound
                bound = diag_segment[0][reflection_dim] > self.center[reflection_dim]

                if not bound:
                    streaming_line_velocities = list(
                        reversed(streaming_line_velocities)
                    )

                reflection_list.extend(
                    self.get_spacetime_reflection_data_d2q4_from_points(
                        properties,
                        expanded_diag_gridpoints,
                        streaming_line_velocities,
                        self.__symmetric_increment(
                            num_steps, reflection_dim, properties
                        ),
                        num_steps,
                    )
                )

        for gridpoint in individual_points:
            for reflection_dim in [0, 1]:
                other_dim = 1 - reflection_dim
                streaming_line_velocities = [0, 2] if reflection_dim == 0 else [1, 3]
                # If larger than the center, then it is an upper bound
                bound = gridpoint[other_dim] > self.center[other_dim]

                if not bound:
                    streaming_line_velocities = list(
                        reversed(streaming_line_velocities)
                    )

                reflection_list.extend(
                    self.get_spacetime_reflection_data_d2q4_from_points(
                        properties,
                        [gridpoint],
                        streaming_line_velocities,
                        self.__symmetric_increment(
                            num_steps, reflection_dim, properties
                        ),
                        num_steps,
                    )
                )
        return reflection_list

    @override
    def get_d2q4_volumetric_reflection_data(self, properties, num_steps=None):
        raise NotImplementedError

    @staticmethod
    def expand_axis_segments(
        axis_segments: List[List[Tuple[int, int]]],
    ) -> List[Tuple[int, ...]]:
        """
        Expands axis-aligned segments encoded as the two extremes of the segment into lists of all points contained within each segment.

        Parameters
        ----------
        axis_segments : List[List[Tuple[int, int]]]
            The segments to expand.

        Returns
        -------
        List[Tuple[int, int]]
            All points within each segment.
        """
        points_in_segments = []
        for segment in axis_segments:
            (x1, y1), (x2, y2) = segment
            if x1 == x2:  # Vertical line
                y_values = list(range(min(y1, y2), max(y1, y2) + 1))
                x_values = [x1] * len(y_values)
            else:  # Horizontal line
                x_values = list(range(min(x1, x2), max(x1, x2) + 1))
                y_values = [y1] * len(x_values)
            points_in_segments.extend(
                [
                    cast(Tuple[int, ...], (x_values[i], y_values[i]))
                    for i in range(len(x_values))
                ]
            )

        return points_in_segments

    @staticmethod
    def expand_diagonal_segments(
        diagonal_segments: List[List[Tuple[int, int]]],
    ) -> List[Tuple[int, ...]]:
        """
        Expands diagonal segments encoded as the two extremes of the segment into lists of all points contained within each segment.

        Parameters
        ----------
        diagonal_segments : List[List[Tuple[int, int]]]
            The diagonal segments to expand

        Returns
        -------
        List[Tuple[int, ...]]
            All points within each segment.
        """
        points_in_segments = []
        for segment in diagonal_segments:
            (x1, y1), (x2, y2) = segment
            step = (1 if x1 < x2 else -1, 1 if y1 < y2 else -1)

            x_values, y_values = [], []

            for offset_multiplier in range(abs(x2 - x1) + 1):
                x_values.append(offset_multiplier * step[0] + x1)
                y_values.append(offset_multiplier * step[1] + y1)

            points_in_segments.extend(
                [
                    cast(Tuple[int, ...], (x_values[i], y_values[i]))
                    for i in range(len(x_values))
                ]
            )
        return points_in_segments
