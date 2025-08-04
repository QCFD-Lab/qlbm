"""Collision operators for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import SpaceTimeOperator
from qlbm.components.common.cbse_collision.cbse_collision import EQCCollisionOperator
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice


class GenericSpaceTimeCollisionOperator(SpaceTimeOperator):
    """
    A generic space-time collision operator that can be used to apply any gate to the velocities of a grid location.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`gate`              The gate to apply to the velocities.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================
    """

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        timestep: int,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice
        self.timestep = timestep

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        local_collision_circuit = EQCCollisionOperator(
            self.lattice.properties.get_discretization()
        ).circuit
        circuit = self.lattice.circuit.copy()

        for velocity_qubit_indices in range(
            self.lattice.properties.get_num_grid_qubits(),
            self.lattice.properties.get_num_grid_qubits()
            + self.lattice.properties.get_num_velocity_qubits(self.timestep),
            self.lattice.properties.get_num_velocities_per_point(),
        ):
            circuit.compose(
                local_collision_circuit,
                inplace=True,
                qubits=range(
                    velocity_qubit_indices,
                    velocity_qubit_indices
                    + self.lattice.properties.get_num_velocities_per_point(),
                ),
            )
        return circuit

    @override
    def __str__(self) -> str:
        return f"[GenericSpaceTimeCollisionOperator for discretization {self.lattice.properties.get_discretization()}]"
