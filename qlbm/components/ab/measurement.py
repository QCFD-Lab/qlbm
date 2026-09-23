"""Quantum circuits used for measurement in the :class:`ABQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import ClassicalRegister, QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.lattices.ab_lattice import ABLattice


class ABGridMeasurement(LBMPrimitive):
    """
    Grid measurement for the :class:`ABQLBM` algorithm.

    By default, this component measures only the grid register. Setting
    ``measure_velocity_qubits=True`` appends the velocity register to the same
    classical register. For a ``16 x 8`` D2Q9 lattice, this produces 7 grid bits
    by default, 11 total bits for an :class:`.ABLattice` with velocity
    measurement, or 16 total bits for an :class:`.OHLattice` with velocity
    measurement.

    With :class:`.ABLattice`, the measured velocity bits encode the binary D2Q9
    channel index. With :class:`.OHLattice`, they form a nine-bit one-hot value.

    .. warning::

        Qiskit displays classical bit strings from the highest classical-bit
        index on the left to index 0 on the right. This circuit maps x-coordinate
        bits first, followed by the remaining dimensions and then velocity bits.
        For the 2D example below, the displayed groups therefore appear in the
        reverse order: velocity, y, then x.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABGridMeasurement
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABGridMeasurement(lattice).draw("mpl")

        # Include the four binary velocity-index qubits as well.
        ABGridMeasurement(lattice, measure_velocity_qubits=True).draw("mpl")

    """

    def __init__(
        self,
        lattice: ABLattice,
        measure_velocity_qubits: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice
        self.measure_velocity_qubits = measure_velocity_qubits

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        circuit.add_register(
            ClassicalRegister(
                self.lattice.num_grid_qubits
                + (
                    self.lattice.num_velocity_qubits
                    if self.measure_velocity_qubits
                    else 0
                )
            )
        )

        circuit.measure(
            self.lattice.grid_index()
            + (self.lattice.velocity_index() if self.measure_velocity_qubits else []),
            list(
                range(
                    self.lattice.num_grid_qubits
                    + (
                        self.lattice.num_velocity_qubits
                        if self.measure_velocity_qubits
                        else 0
                    )
                )
            ),
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABEGridMeasurement with lattice {self.lattice}]"
