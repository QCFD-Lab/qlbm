"""Base classes for geometrical shapes."""

from json import dumps
from typing import Dict, List, override

import numpy as np
from stl import mesh

from qlbm.tools import ComparatorMode

from .base import Shape


class YMonomial(Shape):
    r"""
    Shape representing boundaries shaped by comparisons to monomials.

    Represents a 2D discretized shape defined by a comparison between the :math:`y`
    coordinate and the :math:`x` coordinate raised to a given exponent (i.e. the set of
    points satisfying ``y [comparator] x**exp``). The comparison operator is
    provided via a ComparatorMode instance and the spatial resolution is determined
    by the number of grid qubits per dimension.

    .. important::

        The ``YMonomial`` implementation is a work in progress.
        At present, only the :math:`x^2` monomial case is supported.
        Arbitrary grid sizes are allowed provided
        :math:`n_x \leq n_y` (square and tall grids).

    .. list-table:: Class attributes
            :widths: 25 50
            :header-rows: 1

            * - Attribute
              - Description
            * - :attr:`comparator_mode`
              - The :class:`.ComparatorMode` used to compare :math:`y` and :math:`x^\mathrm{exp}`.
            * - :attr:`exponent`
              - The monomial exponent used to evaluate :math:`x^\mathrm{exp}`.
            * - :attr:`boundary_points`
              - A ``List[List[bool]]`` occupancy grid indicating whether each point satisfies the monomial comparator.
    """

    comparator_mode: ComparatorMode
    """Comparator mode defining the boundary predicate ``y [op] x**exponent``."""

    exponent: int
    """Exponent of the monomial evaluated on the :math:`x` coordinate register."""

    boundary_points: List[List[bool]]
    """Boolean occupancy grid indicating which lattice points satisfy the monomial boundary."""

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
        x_segments = 2 ** self.num_grid_qubits[0]
        y_segments = 2 ** self.num_grid_qubits[1]
        x_segment_width = (x_segments - 1) / x_segments
        y_segment_height = (y_segments - 1) / y_segments

        for y, row in enumerate(self.boundary_points):
            for x, is_inside in enumerate(row):
                if not is_inside:
                    continue

                x0 = x * x_segment_width
                x1 = (x + 1) * x_segment_width
                y0 = y * y_segment_height
                y1 = (y + 1) * y_segment_height

                v00 = np.array([x0, y0, 1.0])
                v10 = np.array([x1, y0, 1.0])
                v01 = np.array([x0, y1, 1.0])
                v11 = np.array([x1, y1, 1.0])

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
