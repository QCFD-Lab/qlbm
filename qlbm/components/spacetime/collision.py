"""Collision operators for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from logging import Logger, getLogger
from math import pi
from time import perf_counter_ns

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import MCMT, RYGate
from typing_extensions import override

from qlbm.components.base import SpaceTimeOperator
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice


class SpaceTimeCollisionOperator(SpaceTimeOperator):
    r"""An operator that performs collision part of the :class:`.SpaceTimeQLBM` algorithm.

    Collision is a local operation that is performed simultaneously on all velocity qubits corresponding to a grid location.
    In practice, this means the same circuit is repeated across all "local" qubit register chunks.
    Collision can be understood as follows:

    #. For each group of qubits, the states encoding velocities belonging to a particular equivalence class are first isolated with a series of :math:`X` and :math:`CX` gates. This leaves qubits not affected by the rotation in :math:`\ket{1}^{\otimes n_v-1}` state.
    #. A rotation gate is applied to the qubit(s) relevant to the equivalence class shift, controlled on the qubits set in the previous step.
    #. The operation performed in Step 1 is undone.

    The register setup of the :class:`.SpaceTimeLattice` is such that following each
    time step, an additional "layer" neighboring velocity qubits can be discarded,
    since the information they encode can never reach the relative origin in the remaining number of time steps.
    As such, the complexity of the collision operator decreases with the number of steps (still) to be simulated.
    For an in-depth mathematical explanation of the procedure, consult pages 11-15 of :cite:t:`spacetime`.


    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`timestep`          The time step for which to perform streaming.
    :attr:`gate_to_apply`     The gate to apply to the velocities matching equivalence classes. Defaults to :math:`R_y(\frac{\pi}{2})`.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================


    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.spacetime import SpaceTimeCollisionOperator
        from qlbm.lattice import SpaceTimeLattice

        # Build an example lattice
        lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
            },
        )

        # Draw the collision operator for 1 time step
        SpaceTimeCollisionOperator(lattice=lattice, timestep=1).draw("mpl")
    """

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        timestep: int,
        gate_to_apply: Gate = RYGate(pi / 2),
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice
        self.timestep = timestep
        self.gate_to_apply = gate_to_apply

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        collision_circuit = self.local_collision_circuit(reset_state=False)
        collision_circuit.compose(
            MCMT(
                self.gate_to_apply,
                self.lattice.properties.get_num_velocities_per_point() - 1,
                1,
            ),
            qubits=list(
                range(1, self.lattice.properties.get_num_velocities_per_point())
            )
            + [0],
            inplace=True,
        )

        # Create the local circuit once
        collision_circuit.compose(
            self.local_collision_circuit(reset_state=True), inplace=True
        )

        # Append the collision circuit at each step
        for velocity_qubit_indices in range(
            self.lattice.properties.get_num_grid_qubits(),
            self.lattice.properties.get_num_grid_qubits()
            + self.lattice.properties.get_num_velocity_qubits(self.timestep),
            self.lattice.properties.get_num_velocities_per_point(),
        ):
            circuit.compose(
                collision_circuit,
                inplace=True,
                qubits=range(velocity_qubit_indices, velocity_qubit_indices + 4),
            )
        return circuit

    def local_collision_circuit(self, reset_state: bool) -> QuantumCircuit:
        """
        Sets the state for the collision circuit one local gridpoint and its corresponding velocity qubits.

        Parameters
        ----------
        reset_state : bool
            Whether the circuit sets or re-sets the state past the application of the rotation operator.

        Returns
        -------
        QuantumCircuit
            The collision (re-)set circuit for a single gridpoint.
        """
        circuit = QuantumCircuit(self.lattice.properties.get_num_velocities_per_point())

        # circuit.cx(1, 2)
        # circuit.cx(0, 1)
        # circuit.cx(0, 3)

        # return circuit if not reset_state else circuit.inverse()

        if not reset_state:
            circuit.cx(control_qubit=0, target_qubit=2)
            circuit.x(0)
            circuit.cx(control_qubit=1, target_qubit=3)
            circuit.cx(control_qubit=0, target_qubit=1)
            circuit.x(list(range(circuit.num_qubits)))
        # Same circuit, but mirrored
        else:
            circuit.x(list(range(circuit.num_qubits)))
            circuit.cx(control_qubit=0, target_qubit=1)
            circuit.cx(control_qubit=1, target_qubit=3)
            circuit.x(0)
            circuit.cx(control_qubit=0, target_qubit=2)

        return circuit

    @override
    def __str__(self) -> str:
        # TODO: Implement
        return "Space Time Collision Operator"
