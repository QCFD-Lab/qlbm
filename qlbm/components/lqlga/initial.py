"""Initial conditions for the :class:`.LQLGA` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, Tuple

from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.tools.utils import flatten


class LQGLAInitialConditions(LBMPrimitive):
    """
    Primitive for setting initial conditions in the :class:`.LQLGA` algorithm.

    This operator allows the construction of arbitrary deterministic initial conditions for the LQLGA algorithm.
    The number of gates required by this operator is equal to the number of enabled velocity qubits across all grid points.
    The depth of the circuit is 1, as all gates are applied in parallel at each grid point.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.lattice import LQLGALattice
        from qlbm.components.lqlga import LQGLAInitialConditions

        lattice = LQLGALattice(
            {
                "lattice": {
                    "dim": {"x": 4},
                    "velocities": "D1Q3",
                },
                "geometry": [],
            },
        )
        initial_conditions = LQGLAInitialConditions(lattice, [(tuple([2]), (True, True, True))])
        initial_conditions.draw("mpl")
    """

    grid_data: List[Tuple[Tuple[int, ...], Tuple[bool, ...]]]
    """
    Grid data for the initial conditions, where each tuple contains:
    #. A tuple of grid point indices (e.g., `(x, y, z)`).
    #. A tuple of booleans indicating which velocity qubits are enabled at that grid point.
    """

    def __init__(
        self,
        lattice: LQLGALattice,
        grid_data: List[Tuple[Tuple[int, ...], Tuple[bool, ...]]],
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)

        self.lattice = lattice
        self.grid_data = grid_data

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self):
        circuit = self.lattice.circuit.copy()

        for tup in self.grid_data:
            gp, vel_profile = tup[0], tup[1]

            if not any(vel_profile):
                continue

            circuit.x(
                self.lattice.gridpoint_index_tuple(gp)
                * self.lattice.num_velocities_per_point
                + v
                for v, is_enabled in enumerate(vel_profile)
                if is_enabled
            )

        if self.lattice.has_multiple_geometries():
            circuit.h(self.lattice.marker_index())

        return circuit

    @override
    def __str__(self):
        return f"[Primitive LQGLAInitialConditions on lattice={self.lattice}, grid_data={self.grid_data})]"


class LQGLAAveragedInitialConditions(LBMPrimitive):
    """TODO."""

    gridpoints: List[int]
    """TODO."""

    def __init__(
        self,
        lattice: LQLGALattice,
        gridpoints: List[int],
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)

        self.lattice = lattice
        self.gridpoints = gridpoints

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self):
        circuit = self.lattice.circuit.copy()

        circuit.h(
            flatten(
                [
                    list(range(gp, gp + self.lattice.num_velocities_per_point))
                    for gp in self.gridpoints
                ]
            )
        )

        return circuit

    @override
    def __str__(self):
        return f"[Primitive LQGLAAveragedInitialConditions on lattice={self.lattice}, gps={self.gridpoints})]"
