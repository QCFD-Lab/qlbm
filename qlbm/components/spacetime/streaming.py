from logging import Logger, getLogger
from time import perf_counter_ns

from qlbm.components.base import SpaceTimeOperator
from qlbm.lattice import SpaceTimeLattice
from qlbm.tools.exceptions import CircuitException


class SpaceTimeStreamingOperator(SpaceTimeOperator):
    """
    An operator that performs streaming as a series of :math:`SWAP` gates as part of the :class:`.SpaceTimeQLBM` algorithm.
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

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self):
        circuit = self.lattice.circuit.copy()

        if self.timestep == 1:
            # !!! TODO: Generalize
            for point_class, extreme_point in enumerate(
                self.lattice.extreme_point_indices[self.timestep]
            ):
                velocity_to_swap = extreme_point.velocity_index_to_swap(point_class, 1)
                circuit.swap(
                    self.lattice.velocity_index(
                        extreme_point.neighbor_index,
                        velocity_to_swap,
                    ),
                    self.lattice.velocity_index(0, velocity_to_swap),
                )
        else:
            raise CircuitException("Not implemented.")

        return circuit

    def __str__(self) -> str:
        # TODO: Implement
        return "Space Time Streaming Operator"
