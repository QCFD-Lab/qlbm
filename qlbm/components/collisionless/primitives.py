"""Primitives for the implementation of the Collisionless Quantum Lattice Boltzmann Method introduced in :cite:t:`collisionless`."""

from enum import Enum
from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import QFT
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.components.collisionless.streaming import SpeedSensitivePhaseShift
from qlbm.lattice import CollisionlessLattice
from qlbm.lattice.geometry.encodings.collisionless import ReflectionResetEdge
from qlbm.tools import flatten


class GridMeasurement(LBMPrimitive):
    """A primitive that implements a measurement operation on the grid qubits.

    Used at the end of the time step circuit to extract information from the quantum state.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.collisionless import GridMeasurement
        from qlbm.lattice import CollisionlessLattice

        # Build an example lattice
        lattice = CollisionlessLattice({
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
        lattice: CollisionlessLattice,
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
        return f"[Primitive InitialConditions with lattice {self.lattice}]"


class CollisionlessInitialConditions(LBMPrimitive):
    """A primitive that creates the quantum circuit to prepare the flow field in its initial conditions.

    The initial conditions create a quantum state spanning half the grid
    in the x-axis, and the entirety of the y (and z)-axes (if 3D).
    All velocities are pointing in the positive direction.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.collisionless import CollisionlessInitialConditions
        from qlbm.lattice import CollisionlessLattice

        # Build an example lattice
        lattice = CollisionlessLattice({
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
        CollisionlessInitialConditions(lattice).draw("mpl")
    """

    def __init__(
        self,
        lattice: CollisionlessLattice,
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


class CollisionlessInitialConditions3DSlim(LBMPrimitive):
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
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.collisionless import CollisionlessInitialConditions3DSlim
        from qlbm.lattice import CollisionlessLattice

        # Build an example lattice
        lattice = CollisionlessLattice({
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
        CollisionlessInitialConditions3DSlim(lattice).draw("mpl")
    """

    def __init__(
        self,
        lattice: CollisionlessLattice,
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


class ComparatorMode(Enum):
    r"""Enumerator for the modes of quantum comparator circuits.

    The modes are as follows:

    * (1, ``ComparatorMode.LT``, :math:`<`);
    * (2, ``ComparatorMode.LE``, :math:`\leq`);
    * (3, ``ComparatorMode.GT``, :math:`>`);
    * (4, ``ComparatorMode.GE``, :math:`\geq`).
    """

    LT = (1,)
    LE = (2,)
    GT = (3,)
    GE = (4,)


class SpeedSensitiveAdder(LBMPrimitive):
    r"""A QFT-based incrementer used to perform streaming in the CQLBM algorithm.

    Incrementation and decerementation are performed as rotations on grid qubits
    that have been previously mapped to the Fourier basis.
    This happens by nesting a :class:`.SpeedSensitivePhaseShift` primitive
    between regular and inverse :math:`QFT`\ s.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`num_qubits`        Number of qubits of the circuit.
    :attr:`speed`             The index of the speed to increment.
    :attr:`positive`          Whether to increment the particles traveling at this speed in the positive (T) or negative (F) direction.
    :attr:`logger`            The performance logger, by default getLogger("qlbm")
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.collisionless import SpeedSensitiveAdder

        SpeedSensitiveAdder(4, 1, True).draw("mpl")
    """

    def __init__(
        self,
        num_qubits: int,
        speed: int,
        positive: bool,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.num_qubits = num_qubits
        self.speed = speed
        self.positive = positive

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)

        circuit.compose(QFT(self.num_qubits), inplace=True)
        circuit.compose(
            SpeedSensitivePhaseShift(
                self.num_qubits,
                self.speed,
                self.positive,
                logger=self.logger,
            ).circuit,
            inplace=True,
        )
        circuit.compose(QFT(self.num_qubits, inverse=True), inplace=True)

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive SimpleAdder on {self.num_qubits} qubits, on velocity {self.speed}, in direction {self.positive}]"


class Comparator(LBMPrimitive):
    """
    Quantum comparator primitive that compares two a quantum state of ``num_qubits`` qubits and an integer ``num_to_compare`` with respect to a :class:`.ComparatorMode`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`num_qubits`        Number of qubits encoding the integer to compare.
    :attr:`num_to_compare`    The integer to compare against.
    :attr:`mode`              The :class:`.ComparatorMode` used to compare the two numbers.
    :attr:`logger`            The performance logger, by default getLogger("qlbm")
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.collisionless import Comparator, ComparatorMode

        # On a 5 qubit register, compare the number 3
        Comparator(num_qubits=5,
                   num_to_compare=3,
                   mode=ComparatorMode.LT).draw("mpl")
    """

    def __init__(
        self,
        num_qubits: int,
        num_to_compare: int,
        mode: ComparatorMode,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.num_qubits = num_qubits
        self.num_to_compare = num_to_compare
        self.mode = mode

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        return self.__create_circuit(self.num_qubits, self.num_to_compare, self.mode)

    def __create_circuit(
        self, num_qubits: int, num_to_compare: int, mode: ComparatorMode
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(num_qubits)

        match mode:
            case ComparatorMode.LT:
                circuit.compose(
                    SpeedSensitiveAdder(
                        num_qubits, num_to_compare, positive=False, logger=self.logger
                    ).circuit,
                    inplace=True,
                )
                circuit.compose(
                    SpeedSensitiveAdder(
                        num_qubits - 1,
                        num_to_compare,
                        positive=True,
                        logger=self.logger,
                    ).circuit,
                    inplace=True,
                    qubits=range(num_qubits - 1),
                )
                return circuit
            case ComparatorMode.LE:
                if num_to_compare == 2 ** (num_qubits - 1) - 1:
                    return self.__create_circuit(num_qubits, 0, ComparatorMode.GE)

                return self.__create_circuit(
                    num_qubits, num_to_compare + 1, ComparatorMode.LT
                )
            case ComparatorMode.GT:
                if num_to_compare == 2 ** (num_qubits - 1) - 1:
                    return circuit
                else:
                    return self.__create_circuit(
                        num_qubits, num_to_compare + 1, ComparatorMode.GE
                    )
            case ComparatorMode.GE:
                circuit = self.__create_circuit(
                    num_qubits, num_to_compare, ComparatorMode.LT
                )
                circuit.x(num_qubits - 1)
                return circuit
            case _:
                raise ValueError("Invalid Comparator Mode")

    @override
    def __str__(self) -> str:
        return f"[Primitive Comparator of {self.num_qubits} and {self.num_to_compare}, mode={self.mode}]"


class EdgeComparator(LBMPrimitive):
    """
    A primitive used in the 3D collisionless :class:`SpecularReflectionOperator` and :class:`BounceBackReflectionOperator` described in :cite:t:`collisionless`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    :attr:`edge`              The coordinates of the edge within the grid.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.collisionless import EdgeComparator
        from qlbm.lattice import CollisionlessLattice

        # Build an example lattice
        lattice = CollisionlessLattice(
            {
                "lattice": {
                    "dim": {"x": 8, "y": 8, "z": 8},
                    "velocities": {"x": 4, "y": 4, "z": 4},
                },
                "geometry": [{"shape":"cuboid", "x": [2, 5], "y": [2, 5], "z": [2, 5], "boundary": "specular"}],
            }
        )

        # Draw the edge comparator circuit for one specific corner edge
        EdgeComparator(lattice, lattice.block_list[0].corner_edges_3d[0]).draw("mpl")
    """

    def __init__(
        self,
        lattice: CollisionlessLattice,
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
        lb_comparator = Comparator(
            self.lattice.num_gridpoints[self.edge.dim_disconnected].bit_length() + 1,
            self.edge.bounds_disconnected_dim[0],
            ComparatorMode.GE,
            logger=self.logger,
        ).circuit
        ub_comparator = Comparator(
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
