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
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import bit_value, dimension_letter, flatten, get_qubits_to_invert


class Circle(SpaceTimeShape):
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
        points = set()
        x, y = self.radius, 0

        # Using Bresenham's circle algorithm
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
        # Create vertices for top and bottom faces
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

        # Center points for top and bottom
        top_center = np.array([self.center[0], self.center[1], 0.5])
        bottom_center = np.array([self.center[0], self.center[1], -0.5])

        # Create triangle faces
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
        return any(self.is_point_on_segment(gridpoint, s) for s in segments)

    def split_perimeter_points(self, points: List[Tuple[int, int]]):
        axis_segments = []
        diagonal_segments = []
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
            segment_start = diag_points[0]
            for i in range(1, len(diag_points)):
                if (
                    diag_points[i][0] != diag_points[i - 1][0] + 1
                    or diag_points[i][1] != diag_points[i - 1][1] + 1
                ):
                    diagonal_segments.append([segment_start, diag_points[i - 1]])
                    segment_start = diag_points[i]
            diagonal_segments.append([segment_start, diag_points[-1]])

        # Secondary diagonal segments (x + y constant)
        for _, diag_points in secondary_diag_groups.items():
            diag_points.sort()
            segment_start = diag_points[0]
            for i in range(1, len(diag_points)):
                if (
                    diag_points[i][0] != diag_points[i - 1][0] + 1
                    or diag_points[i][1] != diag_points[i - 1][1] - 1
                ):
                    diagonal_segments.append([segment_start, diag_points[i - 1]])
                    segment_start = diag_points[i]
            diagonal_segments.append([segment_start, diag_points[-1]])

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
    def to_dict(self):
        return {
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

        for segment in diag_segments:
            expanded_gridpoints = Circle.expand_diagonal_segments([segment])
            for reflection_dim in [0, 1]:
                other_dim = 1 - reflection_dim
                streaming_line_velocities = [0, 2] if reflection_dim == 0 else [1, 3]
                # If larger than the center, then it is an upper bound
                bound = segment[0][other_dim] > self.center[other_dim]

                if not bound:
                    streaming_line_velocities = list(
                        reversed(streaming_line_velocities)
                    )

                reflection_list.extend(
                    self.get_spacetime_reflection_data_d2q4_from_points(
                        properties,
                        expanded_gridpoints,
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
    def expand_axis_segments(axis_segments):
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
                [(x_values[i], y_values[i]) for i in range(len(x_values))]
            )

        return points_in_segments

    @staticmethod
    def expand_diagonal_segments(diagonal_segments):
        points_in_segments = []
        for segment in diagonal_segments:
            (x1, y1), (x2, y2) = segment
            step = (1 if x1 < x2 else -1, 1 if y1 < y2 else -1)

            x_values, y_values = [], []

            for offset_multiplier in range(abs(x2 - x1) + 1):
                x_values.append(offset_multiplier * step[0] + x1)
                y_values.append(offset_multiplier * step[1] + y1)

            points_in_segments.extend(
                [(x_values[i], y_values[i]) for i in range(len(x_values))]
            )
        return points_in_segments

    # @staticmethod
    # def expand
