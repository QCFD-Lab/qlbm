from logging import Logger, getLogger

from qiskit import ClassicalRegister

from qlbm.components.base import SpaceTimeOperator
from qlbm.lattice import SpaceTimeLattice


class SpaceTimeGridVelocityMeasurement(SpaceTimeOperator):
    """
    A primitive that implements a measurement operation on the grid and the local velocity qubits.
    Used at the end of the simulation to extract information from the quantum state.
    Together, the information from the local and grid qubits can be used for on-the-fly reinitialization.


    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================


    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.spacetime import SpaceTimeGridVelocityMeasurement
        from qlbm.lattice import SpaceTimeLattice

        # Build an example lattice
        lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
            },
        )

        # Draw the measurement circuit
        SpaceTimeGridVelocityMeasurement(lattice=lattice).draw("mpl")
    """
    def __init__(
        self,
        lattice: SpaceTimeLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice

        self.circuit = self.create_circuit()

    def create_circuit(self):
        circuit = self.lattice.circuit.copy()

        qubits_to_measure = self.lattice.grid_index() + self.lattice.velocity_index(0)
        circuit.add_register(
            ClassicalRegister(
                self.lattice.num_grid_qubits + self.lattice.num_velocities_per_point
            )
        )

        circuit.measure(
            qubits_to_measure,
            list(range(len(qubits_to_measure))),
        )

        return circuit

    def __str__(self) -> str:
        # TODO: Implement
        return "Space Gird Measurement"
