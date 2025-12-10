"""Quantum circuits for the implementation of QFT-based streaming as described in :cite:t:`collisionless`."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit.synthesis import synth_qft_full as QFT
from typing_extensions import override

from qlbm.components.base import LBMPrimitive, MSOperator
from qlbm.components.common.adders import PhaseShift
from qlbm.lattice import MSLattice
from qlbm.tools import CircuitException, bit_value


class StreamingAncillaPreparation(LBMPrimitive):
    r"""
    A primitive used in :class:`.MSStreamingOperator` that implements the preparatory step of streaming necessary for the :class:`.MSQLBM` method.

    This operator sets the ancilla qubits to :math:`\ket{1}` for the velocities that
    will be streamed in the next CFL time step.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`velocities`        The velocities that need to be streamed within the next time step.
    :attr:`dim`               The dimension to which the velocities correspond.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import StreamingAncillaPreparation
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
                "geometry": [],
            }
        )

        # Streaming velocities indexed 2 in the y (1) dimension
        StreamingAncillaPreparation(lattice=lattice, velocities=[2], dim=1).draw("mpl")
    """

    def __init__(
        self,
        lattice: MSLattice,
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

    @override
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
                MCXGate(num_velocity_qubits),
                qubits=self.lattice.velocity_index(self.dim)
                + self.lattice.ancillae_velocity_index(self.dim),
                inplace=True,
            )

            if velocity_qubit_indices_to_invert:
                # Applying the exact same inversion returns
                # The velocity qubits to their state before the operation
                circuit.x(velocity_qubit_indices_to_invert)

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive StreamingAncillaPreparation on dimension {self.dim}, for velocities {self.velocities}]"


class ControlledIncrementer(LBMPrimitive):
    r"""
    A primitive used in :class:`.MSStreamingOperator` that implements the streaming operation on the states for which the ancilla qubits are in the state :math:`\ket{1}`.

    This primitive is applied after the primitive :class:`.StreamingAncillaPreparation` to compose the streaming operator.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`reflection`        The reflection attribute decides the type of reflection that will take place. This should
                              be either "specular", "bounceback", or ``None``, and defaults to None. This parameter
                              governs which qubits are used as controls for the Fourier space phase shifts.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import ControlledIncrementer
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
                "geometry": [],
            }
        )

        # Streaming velocities indexed 2 in the y (1) dimension
        ControlledIncrementer(lattice=lattice).draw("mpl")
    """

    supported_reflection: List[str] = ["specular", "bounceback"]

    def __init__(
        self,
        lattice: MSLattice,
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

    @override
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

    @override
    def __str__(self) -> str:
        return f"[Primitive ControlledIncrementer with reflection {self.reflection}]"


class MSStreamingOperator(MSOperator):
    """An operator that performs streaming in Fourier space as part of the :class:`.MSQLBM` algorithm.

    Streaming is broken down into the following steps:

    #. A :class:`.StreamingAncillaPreparation` object prepares the ancilla velocity qubits for CFL time step. This happens independently for all dimensions, and it is assumed the velocity discretization is uniform across dimensions.
    #. A :class:`.ControlledIncrementer` performs incrementation or decrementation in the Fourier space, controlled on the ancilla qubits set in the previous steps.
    #. For efficiency reasons, the velocity qubits set in step 1 are **not** reset, as they will be re-used in the subsequent reflection step. Another instance of the :class:`.StreamingAncillaPreparation` would be required to consistently end the step.

    For an in-depth mathematical explanation of the procedure, consult Section 4 of :cite:t:`collisionless`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` based on which the properties of the operator are inferred.
    :attr:`velocities`        A list of velocities to increment. This is computed according to CFL counter.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import MSStreamingOperator
        from qlbm.lattice import MSLattice

        # Build an example lattice
        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
                "geometry": [],
            }
        )

        # Streaming the velocity with index 2
        MSStreamingOperator(lattice=lattice, velocities=[2]).draw("mpl")
    """

    circuit: QuantumCircuit

    def __init__(
        self,
        lattice: MSLattice,
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

    @override
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

    @override
    def __str__(self) -> str:
        return (
            f"[Operator StreamingOperator for velocities {self.velocities_to_stream}]"
        )
