"""Reflection operators for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, XGate
from typing_extensions import override

from qlbm.components.base import SpaceTimeOperator
from qlbm.lattice.blocks import Block
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import CircuitException


class SpaceTimeReflectionOperator(SpaceTimeOperator):
    """Operator implementing reflection in the :class:`.SpaceTimeQLBM` algorithm.

    Work in progress.

    ============================ ======================================================================
    Attribute                     Summary
    ============================ ======================================================================
    :attr:`lattice`              The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`timestep`             The timestep for to which to perform reflection.
    :attr:`blocks`               A list of  :class:`.Block` objects for which to generate the BB boundary condition circuits.
    :attr:`filter_inside_blocks` A ``bool`` that, when enabled, disregards operations that would have happened inside blocks.
    :attr:`logger`               The performance logger, by default ``getLogger("qlbm")``.
    ============================ ======================================================================

    """

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        timestep: int,
        blocks: List[Block],
        filter_inside_blocks: bool = True,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.timestep = timestep

        if timestep < 1 or timestep > lattice.num_timesteps:
            raise CircuitException(
                f"Invalid time step {timestep}, select a value between 1 and {lattice.num_timesteps}"
            )

        self.blocks = blocks
        self.filter_inside_blocks = filter_inside_blocks

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        discretization = self.lattice.properties.get_discretization()
        if discretization == LatticeDiscretization.D1Q2:
            return self.__create_circuit_d1q2()
        if discretization == LatticeDiscretization.D2Q4:
            return self.__create_circuit_d2q4()

        raise CircuitException(f"Reflection Operator unsupported for {discretization}.")

    def __create_circuit_d1q2(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        for block in self.blocks:
            for reflection_data in block.get_spacetime_reflection_data_d1q2(
                self.lattice.properties, self.timestep
            ):
                grid_qubit_indices_to_invert = [
                    self.lattice.grid_index(0)[0] + qubit
                    for qubit in reflection_data.qubits_to_invert
                ]

                if grid_qubit_indices_to_invert:
                    # Inverting the qubits that are 0 turns the
                    # Dimensional grid qubit state encoding this wall to |11...1>
                    # Which in turn allows us to control on this row, in combination with the comparator
                    circuit.x(grid_qubit_indices_to_invert)

                control_qubits = self.lattice.grid_index()
                target_qubits = [
                    self.lattice.velocity_index(
                        neighbor_velocity_pair[0],
                        neighbor_velocity_pair[1],
                    )[0]
                    for neighbor_velocity_pair in reflection_data.neighbor_velocity_pairs
                ]

                # Controlled swap decompositions
                circuit.cx(target_qubits[1], target_qubits[0])
                circuit.compose(
                    MCMT(XGate(), len(control_qubits) + 1, len(target_qubits) - 1),
                    qubits=control_qubits + target_qubits,
                    inplace=True,
                )
                circuit.cx(target_qubits[1], target_qubits[0])

                if grid_qubit_indices_to_invert:
                    circuit.x(grid_qubit_indices_to_invert)
        return circuit

    def __create_circuit_d2q4(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        for block in self.blocks:
            reflection_data_points = block.get_spacetime_reflection_data_d2q4(
                self.lattice.properties, self.timestep
            )

            if self.filter_inside_blocks:
                reflection_data_points = [
                    rdp
                    for rdp in reflection_data_points
                    if not self.lattice.is_inside_an_obstacle(rdp.gridpoint_encoded)
                ]

            for reflection_data in reflection_data_points:
                grid_qubit_indices_to_invert = [
                    self.lattice.grid_index(0)[0] + qubit
                    for qubit in reflection_data.qubits_to_invert
                ]

                if grid_qubit_indices_to_invert:
                    # Inverting the qubits that are 0 turns the
                    # Dimensional grid qubit state encoding this wall to |11...1>
                    # Which in turn allows us to control on this row, in combination with the comparator
                    circuit.x(grid_qubit_indices_to_invert)

                control_qubits = self.lattice.grid_index()
                target_qubits = [
                    self.lattice.velocity_index(
                        neighbor_velocity_pair[0],
                        neighbor_velocity_pair[1],
                    )[0]
                    for neighbor_velocity_pair in reflection_data.neighbor_velocity_pairs
                ]

                # Controlled swap decompositions
                circuit.cx(target_qubits[1], target_qubits[0])
                circuit.compose(
                    MCMT(XGate(), len(control_qubits) + 1, len(target_qubits) - 1),
                    qubits=control_qubits + target_qubits,
                    inplace=True,
                )
                circuit.cx(target_qubits[1], target_qubits[0])

                if grid_qubit_indices_to_invert:
                    circuit.x(grid_qubit_indices_to_invert)
        return circuit

    @override
    def __str__(self) -> str:
        # TODO: Implement
        return "Space Time Reflection Operator"
