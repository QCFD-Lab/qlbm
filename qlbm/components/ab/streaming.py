"""Quantum circuits used for streaming in the :class:`ABQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import QuantumCircuit
from qiskit.synthesis import synth_qft_full as QFT
from typing_extensions import override

from qlbm.components.base import LBMOperator
from qlbm.components.ms.streaming import PhaseShift
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import get_qubits_to_invert


class ABStreamingOperator(LBMOperator):
    """
    Streaming operator for the :class:`ABQLBM` algorithm.

    Uses a variant of the Draper adder described in :cite:`collisionless`.
    The operator works by applying QFTs in parallel to each dimension of the grid,
    followed by phase gates that perform incrementation in the Fourier basis, and, finally,
    by applying an inverse QFT mapping the qubits back to the computational basis.

    Populations are streamed one after the other in Fourier space
    by controlling phase gates on the state of the velocity qubits.
    Additional controls qubits can be specified to restrict this operation.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABStreamingOperator
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABStreamingOperator(lattice).draw("mpl")

    """

    lattice: ABLattice
    """The lattice to construct the component for."""

    additional_control_qubit_indices: List[int]
    """The qubits (if any) that streaming should be controlled over.
    This makes the operator useful for the application of boundary conditions.
    Controls need only be applied to the phase gates and not the QFT blocks."""

    def __init__(
        self,
        lattice: ABLattice,
        additional_control_qubit_indices: List[int] = [],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)

        self.additional_control_qubit_indices = additional_control_qubit_indices

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        if self.lattice.discretization == LatticeDiscretization.D1Q3:
            return self.__create_circuit_d1q3()

        if self.lattice.discretization == LatticeDiscretization.D2Q9:
            return self.__create_circuit_d2q9()

        raise LatticeException("ABE only currently supported in D1Q3 and D2Q9")

    def __create_circuit_d1q3(self):
        # TODO Remove?
        # TODO add geometry
        circuit = self.lattice.circuit.copy()

        circuit.compose(
            QFT(self.lattice.num_grid_qubits),
            qubits=self.lattice.grid_index(),
            inplace=True,
        )

        # 01 streaming in the positive direction
        circuit.x(self.lattice.velocity_index()[0])

        # Controlled Phase Gates for the positive direction
        circuit.compose(
            PhaseShift(
                num_qubits=len(self.lattice.grid_index()),
                positive=True,
                logger=self.logger,
            )
            .circuit.control(2)
            .decompose(),
            qubits=self.lattice.velocity_index() + self.lattice.grid_index(),
            inplace=True,
        )

        # 10 Streaming in the negative direction (and resetting the previous state prep)
        circuit.x(self.lattice.velocity_index())

        circuit.compose(
            PhaseShift(
                num_qubits=len(self.lattice.grid_index()),
                positive=False,  # Negative this time
                logger=self.logger,
            )
            .circuit.control(2)
            .decompose(),
            qubits=self.lattice.velocity_index() + self.lattice.grid_index(),
            inplace=True,
        )

        # Undo the second state prep
        circuit.x(self.lattice.velocity_index()[1])

        # Inverse QFT to return the grid to the computational basis
        circuit.compose(
            QFT(self.lattice.num_grid_qubits, inverse=True),
            qubits=self.lattice.grid_index(),
            inplace=True,
        )

        return circuit

    def __create_circuit_d2q9(self):
        circuit = self.lattice.circuit.copy()

        dim_indices = [
            [
                [1, 5, 8],  # f1, f5, f8 x <- x + 1
                [3, 6, 7],  # f3, f6, f7 x <- x - 1
            ],
            [
                [2, 5, 6],  # f2, f5, f6 y <- y + 1
                [4, 7, 8],  # f4, f7, f8 y <- y + 1
            ],
        ]

        for dim, dim_population_to_update in enumerate(dim_indices):
            circuit.compose(
                QFT(len(self.lattice.grid_index(dim))),
                qubits=self.lattice.grid_index(dim),
                inplace=True,
            )

            for direction, indices in enumerate(dim_population_to_update):
                positive = bool(1 - direction)

                for index in indices:
                    velocity_inversion_qubits = [
                        self.lattice.num_grid_qubits + q
                        for q in get_qubits_to_invert(
                            index, self.lattice.num_velocity_qubits
                        )
                    ]
                    if velocity_inversion_qubits:
                        circuit.x(velocity_inversion_qubits)

                    circuit.compose(
                        PhaseShift(
                            num_qubits=len(self.lattice.grid_index(dim)),
                            positive=positive,
                            logger=self.logger,
                        )
                        .circuit.control(
                            self.lattice.num_velocity_qubits
                            + len(self.additional_control_qubit_indices)
                        )
                        .decompose(),
                        qubits=self.additional_control_qubit_indices
                        + self.lattice.velocity_index()
                        + self.lattice.grid_index(dim),
                        inplace=True,
                    )

                    if velocity_inversion_qubits:
                        circuit.x(velocity_inversion_qubits)

            circuit.compose(
                QFT(len(self.lattice.grid_index(dim)), inverse=True),
                qubits=self.lattice.grid_index(dim),
                inplace=True,
            )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Operator ABStreaming with lattice {self.lattice}]"
