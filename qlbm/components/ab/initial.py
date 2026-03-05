"""Quantum circuits used for setting the initial state in the :class:`ABQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, MCMTGate
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.ab.utils import BinaryToOHPermutation
from qlbm.components.base import LBMPrimitive
from qlbm.components.common.primitives import (
    AdditionConversion,
    StateSetter,
    UniformStatePrep,
)
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.tools.exceptions import CircuitException, LatticeException
from qlbm.tools.utils import dimension_letter


class ABInitialConditions(LBMPrimitive):
    """
    Initial conditions for the :class:`ABQLBM` algorithm.

    This component creates an equal magnitude superposition of all velocity
    basis states at position ``(0, 0)`` using the :class:`.UniformStatePrep`.

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

        circuit.compose(
            UniformStatePrep(
                self.lattice.num_velocity_qubits,
                self.lattice.num_velocities_per_point,
                logger=self.logger,
            ).circuit,
            qubits=self.lattice.velocity_index()[: self.lattice.num_velocity_qubits],
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


class ABDiscreteUniformInitialConditions(LBMPrimitive):
    """
    Initial conditions for the :class:`ABQLBM` algorithm.

    This component creates an equal magnitude superposition of a configurable set of velocity and grid indices.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABDiscreteUniformInitialConditions
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
            }
        )

        ABDiscreteUniformInitialConditions(lattice, [1, 3, 4], ([], [])).draw("mpl")

    The primitive can also applied to the :class:`.OHLattice`:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABDiscreteUniformInitialConditions
        from qlbm.lattice import OHLattice

        lattice = OHLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
            }
        )

        ABDiscreteUniformInitialConditions(lattice, [0, 1], ([0, 1], [0])).draw("mpl")
    """

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
            UniformStatePrep(
                nq,
                len(self.velocity_indices),
                logger=self.logger,
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
                    self.lattice.num_velocity_qubits, v_from, v_to, logger=self.logger
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
        return f"[Primitive ABDiscreteUniformInitialConditions with lattice {self.lattice}, v={self.velocity_indices}, g={self.grid_qubits_to_superpose}]"


class ABParallelDiscreteUniformInitialConditions(LBMPrimitive):
    """
    Marker-sensitive initial conditions for the :class:`ABQLBM` algorithm.

    This component creates an equal magnitude superposition of a configurable set of velocity and grid indices,
    entangled with the state of the marker register.
    Used in parallel realizations of configurations.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABParallelDiscreteUniformInitialConditions
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
            }
        )

        lattice.set_num_marker_qubits(2)

        ABParallelDiscreteUniformInitialConditions(
            lattice,
            [[0, 1], [0, 3], [0], [0, 5]],
            [([0], [0])] * 4,
        ).draw("mpl")

    """

    velocity_indices: List[List[int]]

    grid_qubits_to_superpose: List[Tuple[List[int], ...]]

    lattice: ABLattice

    def __init__(
        self,
        lattice: ABLattice,
        velocity_indices_list: List[List[int]],
        grid_qubits_to_superpose_list: List[Tuple[List[int], ...]],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.lattice = lattice

        if self.lattice.get_encoding() == ABEncodingType.OH:
            raise LatticeException(
                "OHLattice does not currently support parallel initial conditions."
            )

        if len(velocity_indices_list) != len(grid_qubits_to_superpose_list):
            raise CircuitException("Input lists have mismatched lengths.")

        if len(velocity_indices_list) > 2**self.lattice.num_marker_qubits:
            raise LatticeException(
                f"{self.lattice.num_marker_qubits} cannot encode {len(velocity_indices_list)} configurations."
            )

        for velocity_indices, grid_qubits_to_superpose in zip(
            velocity_indices_list, grid_qubits_to_superpose_list
        ):
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
                        not in list(
                            range(self.lattice.num_gridpoints[dim].bit_length())
                        ),
                        grid_qubits_to_superpose[dim],
                    ),
                ):
                    raise LatticeException(
                        f"Grid qubit specification in dimension {dimension_letter(dim)} out of range."
                    )

        self.velocity_indices_list = [
            sorted(velocity_indices) for velocity_indices in velocity_indices_list
        ]
        self.grid_qubits_to_superpose_list = grid_qubits_to_superpose_list

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        # Uniform superposition over the marker index
        circuit.compose(
            UniformStatePrep(
                self.lattice.num_marker_qubits,
                len(self.velocity_indices_list),
                logger=self.logger,
            ).circuit,
            qubits=self.lattice.marker_index(),
            inplace=True,
        )

        for marker_index, velocity_indices, grid_qubits_to_superpose in zip(
            list(range(len(self.velocity_indices_list))),
            self.velocity_indices_list,
            self.grid_qubits_to_superpose_list,
        ):
            nq = int(np.ceil(np.log2(len(velocity_indices))))

            state_setter_circ = StateSetter(
                self.lattice.num_marker_qubits, marker_index, self.logger
            ).circuit

            circuit.compose(
                state_setter_circ, qubits=self.lattice.marker_index(), inplace=True
            )

            circuit.compose(
                UniformStatePrep(
                    nq,
                    len(velocity_indices),
                    num_ctrl_qubits=self.lattice.num_marker_qubits,
                    logger=self.logger,
                ).circuit,
                qubits=self.lattice.velocity_index()[:nq] + self.lattice.marker_index(),
                inplace=True,
            )

            states_from: List[int] = list(range(len(velocity_indices)))
            states_to: List[int] = velocity_indices.copy()

            # Remove indices that are already in place
            for v in velocity_indices:
                if v < len(velocity_indices):
                    states_from.remove(v)
                    states_to.remove(v)

            for v_from, v_to in zip(states_from, states_to):
                circuit.compose(
                    AdditionConversion(
                        self.lattice.num_velocity_qubits,
                        v_from,
                        v_to,
                        num_ctrl_qubits=self.lattice.num_marker_qubits,
                        logger=self.logger,
                    ).circuit,
                    qubits=self.lattice.velocity_index()[
                        : self.lattice.num_velocities_per_point
                    ]  # Additional guard necessary of OH
                    + self.lattice.ancillae_obstacle_index(0)
                    + self.lattice.marker_index(),
                    inplace=True,
                )

            for dim in range(self.lattice.num_dims):
                if grid_qubits_to_superpose[dim]:
                    qs_to_superpose = [
                        self.lattice.grid_index(dim)[0] + q
                        for q in grid_qubits_to_superpose[dim]
                    ]
                    circuit.compose(
                        MCMTGate(
                            HGate(),
                            self.lattice.num_marker_qubits,
                            len(qs_to_superpose),
                        ),
                        qubits=self.lattice.marker_index() + qs_to_superpose,
                        inplace=True,
                    )

            circuit.compose(
                state_setter_circ, qubits=self.lattice.marker_index(), inplace=True
            )

        return circuit.decompose(reps=2)

    @override
    def __str__(self) -> str:
        return f"[Primitive ABParallelDiscreteUniformInitialConditions with lattice {self.lattice}, v={self.velocity_indices_list}, g={self.grid_qubits_to_superpose_list}]"
