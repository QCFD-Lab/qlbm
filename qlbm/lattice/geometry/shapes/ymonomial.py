"""Base classes for geometrical shapes."""

from json import dumps
from typing import Dict, List, override

import numpy as np
from stl import mesh

from qlbm.tools.utils import ComparatorMode

from .base import Shape


class YMonomial(Shape):
    """Base class for all geometrical shapes."""

    def __init__(
        self,
        num_grid_qubits: List[int],
        boundary_condition: str,
        exponent: int,
        comparator_mode: ComparatorMode,
    ):
        super().__init__(num_grid_qubits, boundary_condition)

        self.num_grid_qubits = num_grid_qubits
        self.boundary_condition = boundary_condition
        self.num_dims = len(num_grid_qubits)
        # The number of qubits used to offset "higher" dimensions
        self.previous_qubits: List[int] = [
            sum(num_grid_qubits[previous_dim] for previous_dim in range(dim))
            for dim in range(self.num_dims)
        ]
        self.comparator_mode = comparator_mode

        self.exponent = exponent
        self.boundary_points: List[List[bool]] = [
            [
                ComparatorMode.to_operator(comparator_mode)(y, x**exponent)
                for x in range(2 ** num_grid_qubits[0])
            ]
            for y in range(2 ** num_grid_qubits[1])
        ]

    @override
    def stl_mesh(self) -> mesh.Mesh:
        """
        Provides the ``stl`` representation of the shape.

        Returns
        -------
        ``stl.mesh.Mesh``
            The mesh representing the shape.
        """
        triangles: List[np.ndarray] = []

        for y, row in enumerate(self.boundary_points):
            for x, is_inside in enumerate(row):
                if not is_inside:
                    continue

                v00 = np.array([x, y, 1.0])
                v10 = np.array([x + 1, y, 1.0])
                v01 = np.array([x, y + 1, 1.0])
                v11 = np.array([x + 1, y + 1, 1.0])

                triangles.append(np.array([v00, v10, v11]))
                triangles.append(np.array([v00, v11, v01]))

        ymonomial_mesh = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
        for i, triangle in enumerate(triangles):
            ymonomial_mesh.vectors[i] = triangle

        return ymonomial_mesh

    @override
    def to_json(self) -> str:
        return dumps(self.to_dict())

    @override
    def name(self) -> str:
        return "ymonomial"

    @override
    def to_dict(self) -> Dict[str, int | str]:
        return {
            "shape": self.name(),
            "exponent": self.exponent,
            "comparator": self.comparator_mode.to_string(),
            "boundary": self.boundary_condition,
        }
