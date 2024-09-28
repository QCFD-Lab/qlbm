from logging import Logger, getLogger
from math import pi
from time import perf_counter_ns
from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, QFT, XGate

from qlbm.components.base import CQLBMOperator, LBMPrimitive
from qlbm.lattice import CollisionlessLattice
from qlbm.tools import CircuitException, bit_value


class StreamingAncillaPreparation(LBMPrimitive):
    """
    A primitive used in :class:`.CollisionlessStreamingOperator` that implements the preparatory step of
    streaming necessary for the :class:`.CQLBM` method. Specifically
    this operator sets the ancilla qubits to :math:`\ket{1}` for the velocities that
    will be streamed in the next CFL time step.

    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`velocities`        The velocities that need to be streamed within the next time step.
    :attr:`dim`               The dimension to which the velocities correspond.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================
    """

    def __init__(
        self,
        lattice: CollisionlessLattice,
        velocities: List[int],
        dim: int,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice
        self.velocities = velocities
        self.dim = dim

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        # Ignore the directional qubit
        num_velocity_qubits = self.lattice.num_velocities[self.dim].bit_length() - 1

        for velocity in self.velocities:
            # The indices of the velocity qubits encoding
            # The velocities to be streamed.
            velocity_qubit_indices_to_invert = [
                self.lattice.velocity_index(self.dim)[velocity_qubit]
                for velocity_qubit in range(num_velocity_qubits)
                if (bit_value(velocity, velocity_qubit)) == 0
            ]

            if velocity_qubit_indices_to_invert:
                # Inverting the qubits that are 0 turns the
                # Velocity state in this dimension to |11...1>
                # Which in turn allows us to control on this one velocity
                circuit.x(velocity_qubit_indices_to_invert)

            circuit.compose(
                MCMT(XGate(), num_velocity_qubits, 1),
                qubits=self.lattice.velocity_index(self.dim)
                + self.lattice.ancillae_velocity_index(self.dim),
                inplace=True,
            )

            if velocity_qubit_indices_to_invert:
                # Applying the exact same inversion returns
                # The velocity qubits to their state before the operation
                circuit.x(velocity_qubit_indices_to_invert)

        return circuit

    def __str__(self) -> str:
        return f"[Primitive StreamingAncillaPreparation on dimension {self.dim}, for velocities {self.velocities}]"


class PhaseShift(LBMPrimitive):
    """
    Applies the phase-shift required for controlled
    incrementation as part of the streaming operator
    of the CQBM algorithm :cite:p:`collisionless`.
    Used in the :class`.ControlledIncrementer`.
    """

    def __init__(
        self,
        num_qubits: int,
        positive: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        """
        Parameters
        ----------
        num_qubits : int
            The number of qubits of the circuit.
        positive : bool, optional
            Whether incrementation should be performed in the positive direction, by default False.
        logger : Logger, optional
            The performance logger, by default getLogger("qlbm")
        """
        super().__init__(logger)

        self.num_qubits = num_qubits
        self.positive = positive

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)

        for c, qubit_index in enumerate(range(self.num_qubits)):
            # (2 * positive - 1) will flip the sign if positive is False
            # This effectively inverts the circuit
            phase = (2 * self.positive - 1) * pi / (2 ** (self.num_qubits - 1 - c))
            circuit.p(phase, qubit_index)

        return circuit

    def __str__(self) -> str:
        return f"[Primitive PhaseShift of {self.num_qubits} qubits, in direction {self.positive}]"


class ControlledIncrementer(LBMPrimitive):
    """
    A primitive used in :class:`.CollisionlessStreamingOperator` that implements the streaming operation
    on the states for which the ancilla qubits are in the state :math:`\ket{1}`. This primitive is applied
    after the primitive :class:`.StreamingAncillaPreparation` to compose the streaming operator.

    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`reflection`        The reflection attribute decides the type of reflection that will take place. This should
                              be either "specular", "bounceback", or ``None``, and defaults to None. This parameter
                              governs which qubits are used as controls for the Fourier space phase shifts.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================
    """

    supported_reflection: List[str] = ["specular", "bounceback"]

    def __init__(
        self,
        lattice: CollisionlessLattice,
        reflection: str | None = None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice
        self.reflection = reflection

        if reflection:
            if reflection not in self.supported_reflection:
                raise CircuitException(
                    f'Controlled Incrementer does not support reflection type "{reflection}". Supported types are {self.supported_reflection}'
                )

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        for dim in range(self.lattice.num_dims):
            num_qubits_dim = self.lattice.num_gridpoints[dim].bit_length()
            grid_index = self.lattice.grid_index(dim)

            circuit.compose(QFT(num_qubits_dim), inplace=True, qubits=grid_index)

            # The gate is controlled by the corresponding velocity direction and velocity ancilla qubits
            if self.reflection == "specular":
                control_qubits = self.lattice.ancillae_obstacle_index(
                    dim
                ) + self.lattice.velocity_dir_index(dim)
            elif self.reflection == "bounceback":
                control_qubits = self.lattice.ancillae_obstacle_index(
                    0
                ) + self.lattice.velocity_dir_index(dim)
            else:
                control_qubits = self.lattice.ancillae_velocity_index(
                    dim
                ) + self.lattice.velocity_dir_index(dim)

            # Add the UP+ controlled rotation block
            circuit.compose(
                PhaseShift(
                    num_qubits=len(grid_index),
                    positive=True,
                    logger=self.logger,
                )
                .circuit.control(2)
                .decompose(),
                qubits=control_qubits + grid_index,
                inplace=True,
            )

            circuit.x(self.lattice.velocity_dir_index(dim))

            # Add the UP- controlled rotation block circuit
            circuit.compose(
                PhaseShift(
                    num_qubits=len(grid_index),
                    positive=False,
                    logger=self.logger,
                )
                .circuit.control(2)
                .decompose(),
                qubits=control_qubits + grid_index,
                inplace=True,
            )

            # Reset the state of the velocity direction qubit
            circuit.x(self.lattice.velocity_dir_index(dim))

            circuit.compose(
                QFT(num_qubits_dim, inverse=True), inplace=True, qubits=grid_index
            )

        return circuit

    def __str__(self) -> str:
        return f"[Primitive ControlledIncrementer with reflection {self.reflection}]"


class CollisionlessStreamingOperator(CQLBMOperator):
    circuit: QuantumCircuit

    def __init__(
        self,
        lattice: CollisionlessLattice,
        velocities: List[int],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.velocities_to_stream = velocities

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self):
        circuit = self.lattice.circuit.copy()

        for dim in range(self.lattice.num_dims):
            circuit.compose(
                StreamingAncillaPreparation(
                    self.lattice,
                    self.velocities_to_stream,
                    dim,
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )

        circuit.compose(
            ControlledIncrementer(
                self.lattice,
                logger=self.logger,
            ).circuit,
            inplace=True,
        )

        return circuit

    def __str__(self) -> str:
        return (
            f"[Operator StreamingOperator for velocities {self.velocities_to_stream}]"
        )


class SpeedSensitivePhaseShift(LBMPrimitive):
    def __init__(
        self,
        num_qubits: int,
        speed: int,
        positive: bool = False,
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

    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)
        angles = np.zeros(self.num_qubits)

        for qubit_index in range(self.num_qubits):
            dig = bit_value(self.speed, qubit_index)
            for i in range(self.num_qubits - qubit_index):
                # (2 * positive - 1) will flip the sign if positive is False
                # This effectively inverts the circuit
                angles[i] += (
                    (2 * self.positive - 1)
                    * dig
                    * pi
                    / (2 ** (self.num_qubits - qubit_index - i - 1))
                )

        for qubit_index in range(self.num_qubits):
            circuit.p(angles[qubit_index], qubit_index)

        return circuit

    def __str__(self) -> str:
        return f"[Primitive SpeedSensitivePhaseShift of {self.num_qubits} qubits, speed {self.speed}, in direction {self.positive}]"
