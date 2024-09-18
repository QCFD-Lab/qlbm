from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import MCMT, QFT, XGate

from qlbm.components.base import LBMPrimitive
from qlbm.components.common import PhaseShift
from qlbm.lattice import CollisionlessLattice
from qlbm.tools import CircuitException, bit_value, flatten


class StreamingAncillaPreparation(LBMPrimitive):
    """
    A primitive used in :class:`.CollisionlessStreamingOperator` that implements the preparatory step of 
    streaming necessary for the collisionless QLBM method. Specifically
    this operator sets the ancilla qubits to 1 for the velocities that 
    will be streamed in the next time step.

    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    :attr:`velocities`        The velocities that need to be streamed within the next time step. 
    :attr:`dim`               The dimension to which the velocities correspond.
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


class ControlledIncrementer(LBMPrimitive):
    """
    A primitive used in :class:`.CollisionlessStreamingOperator` that implements the streaming operation 
    on the states for which the ancilla qubits are in the state 1. This primitive is applied 
    after the primitive :class:`.StreamingAncillaPreparation`.

    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    :attr:`reflection`        The reflection attribute decides the type of reflection that will take place. This should
                              be either "specular" or "bounceback".
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

        for dim in range(self.lattice.num_dimensions):
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


class GridMeasurement(LBMPrimitive):
    """
    A primitive that implements a measurement operation on the qubits expressing the grid. 

    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================
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

    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)
        all_grid_qubits: List[int] = flatten(
            [self.lattice.grid_index(dim) for dim in range(self.lattice.num_dimensions)]
        )
        circuit.add_register(ClassicalRegister(self.lattice.num_grid_qubits))

        circuit.measure(
            all_grid_qubits,
            list(range(self.lattice.num_grid_qubits)),
        )

        return circuit

    def __str__(self) -> str:
        return f"[Primitive InitialConditions with lattice {self.lattice}]"


class CollisionlessInitialConditions(LBMPrimitive):
    """
    A primitive that creates the quantum circuit to prepare the flow field in its initial conditions. 

    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================
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

    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        for dim in range(self.lattice.num_dimensions):
            circuit.x(self.lattice.velocity_dir_index(dim)[0])

        for x in self.lattice.grid_index(0)[:-1]:
            circuit.h(x)

        if self.lattice.num_dimensions > 1:
            circuit.h(self.lattice.grid_index(1))

        if self.lattice.num_dimensions > 2:
            circuit.h(self.lattice.grid_index(2))

        return circuit

    def __str__(self) -> str:
        return f"[Primitive InitialConditions with lattice {self.lattice}]"


class CollisionlessInitialConditions3DSlim(LBMPrimitive):
    """
    A primitive that creates the quantum circuit to prepare the flow field in its initial conditions for 3 dimensions. 

    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================
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

    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        # for dim in range(self.lattice.num_dimensions):

        circuit.x(self.lattice.velocity_dir_index())

        # for x in self.lattice.grid_index(0)[:-1]:
        # circuit.h(self.lattice.grid_index(0)[0])

        # if self.lattice.num_dimensions > 1:
        #     circuit.h(self.lattice.grid_index(1))

        if self.lattice.num_dimensions > 2:
            circuit.h(self.lattice.grid_index(2))

        return circuit

    def __str__(self) -> str:
        return f"[Primitive InitialConditions with lattice {self.lattice}]"
