"""Primitives for the implementation of the Collisionless Quantum Lattice Boltzmann Method introduced in :cite:t:`collisionless`."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import ClassicalRegister, QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.components.common.comparators import SingleRegisterComparator
from qlbm.lattice import MSLattice
from qlbm.lattice.geometry.encodings.ms import ReflectionResetEdge
from qlbm.tools import flatten
from qlbm.tools.utils import ComparatorMode


class GridMeasurement(LBMPrimitive):
    """A primitive that implements a measurement operation on the grid qubits.

    Used at the end of the time step circuit to extract information from the quantum state.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import GridMeasurement
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice({
            "lattice": {
                "dim": {
                        "x": 8,
                        "y": 8
                    },
                    "velocities": {
                        "x": 4,
                        "y": 4
                }
            },
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [5, 6],
                    "y": [1, 2],
                    "boundary": "specular"
                }
            ]
        })

        # Draw the measurement circuit
        GridMeasurement(lattice).draw("mpl")
    """

    def __init__(
        self,
        lattice: MSLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)
        all_grid_qubits: List[int] = flatten(
            [self.lattice.grid_index(dim) for dim in range(self.lattice.num_dims)]
        )
        circuit.add_register(ClassicalRegister(self.lattice.num_grid_qubits))

        circuit.measure(
            all_grid_qubits,
            list(range(self.lattice.num_grid_qubits)),
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive DVGridMeasurement with lattice {self.lattice}]"


class MSInitialConditions(LBMPrimitive):
    """A primitive that creates the quantum circuit to prepare the flow field in its initial conditions for the :class:`.MSLattice`.

    The initial conditions create a quantum state spanning half the grid
    in the x-axis, and the entirety of the y (and z)-axes (if 3D).
    All velocities are pointing in the positive direction.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import MSInitialConditions
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice({
            "lattice": {
                "dim": {
                        "x": 8,
                        "y": 8
                    },
                    "velocities": {
                        "x": 4,
                        "y": 4
                }
            },
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [5, 6],
                    "y": [1, 2],
                    "boundary": "specular"
                }
            ]
        })

        # Draw the initial conditions circuit
        MSInitialConditions(lattice).draw("mpl")
    """

    def __init__(
        self,
        lattice: MSLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        for dim in range(self.lattice.num_dims):
            circuit.x(self.lattice.velocity_dir_index(dim)[0])

        for x in self.lattice.grid_index(0)[:-1]:
            circuit.h(x)

        if self.lattice.num_dims > 1:
            circuit.h(self.lattice.grid_index(1))

        if self.lattice.num_dims > 2:
            circuit.h(self.lattice.grid_index(2))

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive InitialConditions with lattice {self.lattice}]"


class MSInitialConditions3DSlim(LBMPrimitive):
    r"""
    A primitive that creates the quantum circuit to prepare the flow field in its initial conditions for 3 dimensions.

    The initial conditions create the quantum state
    :math:`\Sigma_{j}\ket{0}^{\otimes n_{g_x}}\ket{0}^{\otimes n_{g_y}}\ket{j}` over the grid qubits,
    that is, spanning the z-axis at the bottom of the x- and y-axes.
    This is helpful for debugging edge cases around the corners of 3D obstacles.
    All velocities are pointing in the positive direction.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import MSInitialConditions3DSlim
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice({
            "lattice": {
                "dim": {
                "x": 8,
                "y": 8,
                "z": 8
                },
                "velocities": {
                "x": 4,
                "y": 4,
                "z": 4
                }
            },
            "geometry": []
        })

        # Draw the initial conditions circuit
        MSInitialConditions3DSlim(lattice).draw("mpl")
    """

    def __init__(
        self,
        lattice: MSLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        # for dim in range(self.lattice.num_dimensions):

        circuit.x(self.lattice.velocity_dir_index())

        # for x in self.lattice.grid_index(0)[:-1]:
        # circuit.h(self.lattice.grid_index(0)[0])

        # if self.lattice.num_dimensions > 1:
        #     circuit.h(self.lattice.grid_index(1))

        if self.lattice.num_dims > 2:
            circuit.h(self.lattice.grid_index(2))

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive InitialConditions with lattice {self.lattice}]"


class EdgeComparator(LBMPrimitive):
    """
    A primitive used in the 3D collisionless :class:`SpecularReflectionOperator` and :class:`BounceBackReflectionOperator` described in :cite:t:`collisionless`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    :attr:`edge`              The coordinates of the edge within the grid.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import EdgeComparator
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice(
            {
                "lattice": {
                    "dim": {"x": 8, "y": 8, "z": 8},
                    "velocities": {"x": 4, "y": 4, "z": 4},
                },
                "geometry": [{"shape":"cuboid", "x": [2, 5], "y": [2, 5], "z": [2, 5], "boundary": "specular"}],
            }
        )

        # Draw the edge comparator circuit for one specific corner edge
        EdgeComparator(lattice, lattice.shape_list[0].corner_edges_3d[0]).draw("mpl")
    """

    def __init__(
        self,
        lattice: MSLattice,
        edge: ReflectionResetEdge,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice
        self.edge = edge

        self.circuit = self.create_circuit()

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        lb_comparator = SingleRegisterComparator(
            self.lattice.num_gridpoints[self.edge.dim_disconnected].bit_length() + 1,
            self.edge.bounds_disconnected_dim[0],
            ComparatorMode.GE,
            logger=self.logger,
        ).circuit
        ub_comparator = SingleRegisterComparator(
            self.lattice.num_gridpoints[self.edge.dim_disconnected].bit_length() + 1,
            self.edge.bounds_disconnected_dim[1],
            ComparatorMode.LE,
            logger=self.logger,
        ).circuit

        # for c, wall_alignment_dim in enumerate(self.wall.alignment_dims):
        circuit.compose(
            lb_comparator,
            qubits=self.lattice.grid_index(self.edge.dim_disconnected)
            + self.lattice.ancillae_comparator_index(0)[
                :-1  # :-1 Effectively selects only the first (lb) qubit
            ],  # There are two comparator ancillae, for each relevant dimension, one for l and one for u
            inplace=True,
        )

        circuit.compose(
            ub_comparator,
            qubits=self.lattice.grid_index(self.edge.dim_disconnected)
            + self.lattice.ancillae_comparator_index(0)[
                1:  # 1: Effectively selects only the last (ub) qubit
            ],  # There are two comparator ancillae, for each relevant dimension, one for l and one for u.
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive SpecularEdgeComparator on edge={self.edge}]"
