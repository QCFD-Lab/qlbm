"""Streaming operators for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import SpaceTimeOperator
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import CircuitException


class SpaceTimeStreamingOperator(SpaceTimeOperator):
    """An operator that performs streaming as a series of :math:`SWAP` gates as part of the :class:`.SpaceTimeQLBM` algorithm.

    The velocities corresponding to neighboring gridpoints are streamed "into" the gridpoint affected relative to the ``timestep``.
    The register setup of the :class:`.SpaceTimeLattice` is such that following each
    time step, an additional "layer" neighboring velocity qubits can be discarded,
    since the information they encode can never reach the relative origin in the remaining number of time steps.
    As such, the complexity of the streaming operator decreases with the number of steps (still) to be simulated.
    For an in-depth mathematical explanation of the procedure, consult pages 15-18 of :cite:t:`spacetime`.


    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`timestep`          The time step for which to perform streaming.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.spacetime import SpaceTimeStreamingOperator
        from qlbm.lattice import SpaceTimeLattice

        # Build an example lattice
        lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
            },
        )

        # Draw the streaming operator for 1 time step
        SpaceTimeStreamingOperator(lattice=lattice, timestep=1).draw("mpl")
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

        if timestep < 1 or timestep > lattice.num_timesteps:
            raise CircuitException(
                f"Invalid time step {timestep}, select a value between 1 and {lattice.num_timesteps}"
            )

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

        raise CircuitException(f"Streaming Operator unsupported for {discretization}.")

    def __create_circuit_d1q2(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        circuit = self.stream_lines(
            self.lattice.properties.get_streaming_lines(0, True, self.timestep),
            0,
            circuit,
        )
        circuit = self.stream_lines(
            self.lattice.properties.get_streaming_lines(0, False, self.timestep),
            1,
            circuit,
        )

        return circuit

    def __create_circuit_d2q4(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        circuit = self.stream_lines(
            self.lattice.properties.get_streaming_lines(0, True, self.timestep),
            0,
            circuit,
        )
        circuit = self.stream_lines(
            self.lattice.properties.get_streaming_lines(0, False, self.timestep),
            2,
            circuit,
        )
        circuit = self.stream_lines(
            self.lattice.properties.get_streaming_lines(1, True, self.timestep),
            1,
            circuit,
        )
        circuit = self.stream_lines(
            self.lattice.properties.get_streaming_lines(1, False, self.timestep),
            3,
            circuit,
        )

        return circuit

    def stream_lines(
        self,
        streaming_lines: List[List[int]],
        velocity_direction: int,
        circuit: QuantumCircuit,
    ) -> QuantumCircuit:
        """
        Apply the swap gates that move the velocity qubits to their neighboring gridpoints along a line.

        Parameters
        ----------
        streaming_lines : List[List[int]]
            The lines to stream, formatted as ordered lists of neighbor indices to swap.
        velocity_direction : int
            The number of the velocity qubit to swap for each neighbor.
        circuit : QuantumCircuit
            The circuit to extend.

        Returns
        -------
        QuantumCircuit
            The circuit containing the line streaming operations.
        """
        for streaming_line in streaming_lines:
            for c, neighbor in enumerate(streaming_line):
                if c == len(streaming_line) - 1:
                    break
                circuit.swap(
                    self.lattice.velocity_index(neighbor, velocity_direction),
                    self.lattice.velocity_index(
                        streaming_line[c + 1], velocity_direction
                    ),
                )

        return circuit

    @override
    def __str__(self) -> str:
        # TODO: Implement
        return "Space Time Streaming Operator"
