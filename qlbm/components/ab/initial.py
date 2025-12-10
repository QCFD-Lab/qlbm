"""Quantum circuits used for setting the initial state in the :class:`ABQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.ab.utils import BinaryToOHPermutation
from qlbm.components.base import LBMPrimitive
from qlbm.components.common.primitives import AdditionConversion, TruncatedQFT
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import dimension_letter


class ABInitialConditions(LBMPrimitive):
    """
    Initial conditions for the :class:`ABQLBM` algorithm.

    This component creates an equal magnitude superposition of all velocity
    basis states at position ``(0, 0)`` using the :class:`TruncatedQFT`.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABInitialConditions
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABInitialConditions(lattice).draw("mpl")

    You can also get the low-level decomposition of the circuit as:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABInitialConditions
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABInitialConditions(lattice).circuit.decompose(reps=2).draw("mpl")
    """

    lattice: ABLattice

    def __init__(
        self,
        lattice: ABLattice,
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

        nq = int(np.ceil(np.log2(self.lattice.num_velocities_per_point)))
        circuit.compose(
            TruncatedQFT(
                nq,
                self.lattice.num_velocity_qubits,
                self.logger,
            ).circuit,
            qubits=self.lattice.velocity_index()[:nq],
            inplace=True,
        )

        match self.lattice.get_encoding():
            case ABEncodingType.AB:
                circuit.h(self.lattice.grid_index(1))
            case ABEncodingType.OH:
                circuit.compose(
                    BinaryToOHPermutation(self.lattice, self.logger).circuit,
                    qubits=self.lattice.velocity_index(),
                    inplace=True,
                )
            case _:
                raise LatticeException(
                    f"Encoding {self.lattice.get_encoding()} not supported."
                )

        if self.lattice.has_multiple_geometries():
            circuit.h(self.lattice.marker_index())

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABEInitialConditions with lattice {self.lattice}]"


class DiscreteUniformVelocityABInitialConditions(LBMPrimitive):
    velocity_indices: List[int]

    grid_qubits_to_superpose: Tuple[List[int], ...]

    lattice: ABLattice

    def __init__(
        self,
        lattice: ABLattice,
        velocity_indices: List[int],
        grid_qubits_to_superpose: Tuple[List[int], ...],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.lattice = lattice

        if any(
            map(
                lambda x: x
                not in list(range(0, self.lattice.num_velocities_per_point)),
                velocity_indices,
            )
        ):
            raise LatticeException(
                f"Velocity indices should be in the interval 0..{self.lattice.num_velocities_per_point}"
            )

        if len(grid_qubits_to_superpose) != self.lattice.num_dims:
            raise LatticeException(
                f"Lattice has {self.lattice.num_dims} dimensions, but provided grid qubit information has {len(grid_qubits_to_superpose)} entries."
            )

        for dim in range(self.lattice.num_dims):
            if any(
                map(
                    lambda x: x
                    not in list(range(self.lattice.num_gridpoints[dim].bit_length())),
                    grid_qubits_to_superpose[dim],
                ),
            ):
                raise LatticeException(
                    f"Grid qubit specification in dimension {dimension_letter(dim)} out of range."
                )

        self.velocity_indices = sorted(velocity_indices)
        self.grid_qubits_to_superpose = grid_qubits_to_superpose

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        nq = int(np.ceil(np.log2(len(self.velocity_indices))))

        circuit.compose(
            TruncatedQFT(
                nq,
                len(self.velocity_indices),
                self.logger,
            ).circuit,
            qubits=self.lattice.velocity_index()[:nq],
            inplace=True,
        )

        states_from = list(range(len(self.velocity_indices)))
        states_to = self.velocity_indices.copy()

        # Remove indices that are already in place
        for v in self.velocity_indices:
            if v < len(self.velocity_indices):
                states_from.remove(v)
                states_to.remove(v)

        for v_from, v_to in zip(states_from, states_to):
            circuit.compose(
                AdditionConversion(
                    self.lattice.num_velocity_qubits, v_from, v_to, self.logger
                ).circuit,
                qubits=self.lattice.velocity_index()[
                    : self.lattice.num_velocities_per_point
                ]  # Additional guard necessary of OH
                + self.lattice.ancillae_obstacle_index(0),
                inplace=True,
            )

        if self.lattice.get_encoding() == ABEncodingType.OH:
            circuit.compose(
                BinaryToOHPermutation(self.lattice, self.logger).circuit,
                qubits=self.lattice.velocity_index(),
                inplace=True,
            )

        for dim in range(self.lattice.num_dims):
            if self.grid_qubits_to_superpose[dim]:
                circuit.h(
                    [
                        self.lattice.grid_index(dim)[0] + q
                        for q in self.grid_qubits_to_superpose[dim]
                    ]
                )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive DiscreteUniformVelocityABInitialConditions with lattice {self.lattice}, v={self.velocity_indices}, g={self.grid_qubits_to_superpose}]"
