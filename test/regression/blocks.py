from typing import List

import numpy as np
from stl import mesh

from qlbm.tools.utils import bit_value


class Block2D:
    mesh_indices: np.ndarray = np.array([[1, 2, 3], [3, 1, 0]])

    def __init__(self, lx: int, ux: int, ly: int, uy: int, nq_x: int, nq_y: int):
        self.lx = lx
        self.ux = ux
        self.ly = ly
        self.uy = uy

        self.outside_corner_lx = lx - 1
        self.outside_corner_ly = ly - 1
        self.outside_corner_ux = ux + 1
        self.outside_corner_uy = uy + 1

        self.y_q = list(range(nq_x, nq_y))
        self.x_q = list(range(nq_x))
        self.all_q = self.x_q + self.y_q

        self.to_swap_lx = [i for i in range(nq_x) if not bit_value(lx, i)]
        self.to_swap_ux = [i for i in range(nq_x) if not bit_value(ux, i)]
        self.to_swap_ly = [i + nq_x for i in range(nq_y) if not bit_value(ly, i)]
        self.to_swap_uy = [i + nq_x for i in range(nq_y) if not bit_value(uy, i)]

        self.to_swap_corner_lx = [
            i for i in range(nq_x) if not bit_value(self.outside_corner_lx, i)
        ]
        self.to_swap_corner_ux = [
            i for i in range(nq_x) if not bit_value(self.outside_corner_ux, i)
        ]
        self.to_swap_corner_ly = [
            i + nq_x for i in range(nq_y) if not bit_value(self.outside_corner_ly, i)
        ]
        self.to_swap_corner_uy = [
            i + nq_x for i in range(nq_y) if not bit_value(self.outside_corner_uy, i)
        ]

        self.corner_points = [
            ReflectionPoints2D(self.to_swap_ux + self.to_swap_uy, False, False),
            ReflectionPoints2D(self.to_swap_ux + self.to_swap_ly, False, True),
            ReflectionPoints2D(self.to_swap_lx + self.to_swap_uy, True, False),
            ReflectionPoints2D(self.to_swap_lx + self.to_swap_ly, True, True),
        ]

        self.outside_corner_points = [
            ReflectionPoints2D(
                self.to_swap_corner_lx + self.to_swap_corner_ly, False, False
            ),
            ReflectionPoints2D(
                self.to_swap_corner_lx + self.to_swap_corner_uy, False, True
            ),
            ReflectionPoints2D(
                self.to_swap_corner_ux + self.to_swap_corner_ly, True, False
            ),
            ReflectionPoints2D(
                self.to_swap_corner_ux + self.to_swap_corner_uy, True, True
            ),
        ]

        self.xside = [
            ReflectionPoints2D(self.to_swap_corner_lx + self.to_swap_uy, False, False),
            ReflectionPoints2D(self.to_swap_corner_lx + self.to_swap_ly, False, True),
            ReflectionPoints2D(self.to_swap_corner_ux + self.to_swap_uy, True, False),
            ReflectionPoints2D(self.to_swap_corner_ux + self.to_swap_ly, True, True),
        ]

        self.yside = [
            ReflectionPoints2D(self.to_swap_ux + self.to_swap_corner_ly, False, False),
            ReflectionPoints2D(self.to_swap_ux + self.to_swap_corner_uy, False, True),
            ReflectionPoints2D(self.to_swap_lx + self.to_swap_corner_ly, True, False),
            ReflectionPoints2D(self.to_swap_lx + self.to_swap_corner_uy, True, True),
        ]

        self.x_walls = [
            ReflectionWall2D(uy, ly, self.to_swap_lx, True),
            ReflectionWall2D(uy, ly, self.to_swap_ux, False),
        ]
        self.outside_x = [
            ReflectionWall2D(uy, ly, self.to_swap_corner_lx, False),
            ReflectionWall2D(uy, ly, self.to_swap_corner_ux, True),
        ]
        self.y_walls = [
            ReflectionWall2D(ux, lx, self.to_swap_ly, True),
            ReflectionWall2D(ux, lx, self.to_swap_uy, False),
        ]
        self.outside_y = [
            ReflectionWall2D(ux, lx, self.to_swap_corner_ly, False),
            ReflectionWall2D(ux, lx, self.to_swap_corner_uy, True),
        ]

    def stl_mesh(self):
        mesh_vertices = np.array(
            [
                [self.lx, self.ly, 1],
                [self.ux, self.ly, 1],
                [self.ux, self.uy, 1],
                [self.lx, self.uy, 1],
            ]
        )

        block = mesh.Mesh(np.zeros(self.mesh_indices.shape[0], dtype=mesh.Mesh.dtype))
        for i, triangle_face in enumerate(self.mesh_indices):
            for j in range(3):
                block.vectors[i][j] = mesh_vertices[triangle_face[j]]

        return block


class ReflectionWall2D:
    def __init__(self, upper: int, lower: int, invert: List[int], direction: bool):
        self.upper = upper
        self.lower = lower
        self.invert = invert
        self.direction = direction

    def __str__(self) -> str:
        return f"Reflection off Wall between {self.lower} and {self.upper} in direction {'positive' if self.direction else 'negative'} in qubits={self.invert}"


class ReflectionPoints2D:
    def __init__(self, point_list: List[int], x_reflection: bool, y_reflection: bool):
        self.point_list = point_list
        self.x_reflection = x_reflection
        self.y_reflection = y_reflection

    def __str__(self) -> str:
        return f"Reflection of corner points at {self.point_list} with x={self.x_reflection} and y={self.y_reflection}"
