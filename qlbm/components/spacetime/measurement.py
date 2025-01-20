"""Measurement operator for the :class:`.SpaceTimeQLBM` algorithm :cite:`spacetime`."""

from logging import Logger, getLogger
from typing import Tuple

from qiskit import ClassicalRegister
from qiskit.circuit.library import MCMT, XGate
from typing_extensions import override

from qlbm.components.base import SpaceTimeOperator
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.utils import flatten, get_qubits_to_invert


class SpaceTimeGridVelocityMeasurement(SpaceTimeOperator):
    """A primitive that implements a measurement operation on the grid and the local velocity qubits.

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

    @override
    def create_circuit(self):
        circuit = self.lattice.circuit.copy()

        qubits_to_measure = self.lattice.grid_index() + self.lattice.velocity_index(0)
        circuit.add_register(
            ClassicalRegister(
                self.lattice.properties.get_num_grid_qubits()
                + self.lattice.properties.get_num_velocities_per_point()
            )
        )

        circuit.measure(
            qubits_to_measure,
            list(range(len(qubits_to_measure))),
        )

        return circuit

    @override
    def __str__(self) -> str:
        # TODO: Implement
        return "Space Gird Measurement"


class SpaceTimePointWiseMassMeasurement(SpaceTimeOperator):
    """
    A primitive that performs mass measurement.

    WIP.
    """

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        gridpoint: Tuple[int, int],
        velocity_index_to_measure: int,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        self.lattice = lattice
        self.gridpoint = gridpoint
        self.velocity_index_to_measure = velocity_index_to_measure
        self.logger = logger

        self.circuit = self.create_circuit()

    @override
    def create_circuit(self):
        circuit = self.lattice.circuit.copy()

        circuit.add_register(ClassicalRegister(1))

        qubits_to_invert = [
            get_qubits_to_invert(coord, self.lattice.num_gridpoints[dim].bit_length())
            for dim, coord in enumerate(self.gridpoint)
        ]

        for dim in range(self.lattice.num_dims):
            for i in range(len(qubits_to_invert[dim])):
                qubits_to_invert[dim][i] += (
                    self.lattice.properties.get_num_previous_grid_qubits(dim)
                )

        qubits_to_invert = flatten(qubits_to_invert)

        if qubits_to_invert:
            circuit.x(qubits_to_invert)

        control_qubits = self.lattice.grid_index() + self.lattice.velocity_index(
            0, self.velocity_index_to_measure
        )
        target_qubits = self.lattice.ancilla_mass_index()

        circuit.compose(
            MCMT(
                XGate(),
                len(control_qubits),
                len(target_qubits),
            ),
            qubits=control_qubits + target_qubits,
            inplace=True,
        )

        circuit.measure(
            target_qubits,
            [0],
        )

        return circuit

    @override
    def __str__(self) -> str:
        # TODO: Implement
        return "SpaceTimePointWiseMassMeasurement"
