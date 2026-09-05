r"""Quantum circuits used for measurement in the :class:`.ABBGKQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import ClassicalRegister, QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.lattices.ab_bgk_lattice import ABBGKLattice


class ABBGKMeasurement(LBMPrimitive):
    r"""
    Measurement for the :class:`.ABBGKQLBM` algorithm.

    Reconstructing a flow field takes more than the grid marginal that
    :class:`.ABGridMeasurement` samples: the macroscopic moments are weighted sums over
    the velocity channels of each grid point, so the grid and velocity registers have to
    be sampled jointly. The marker qubit is measured alongside them, which lets
    :class:`.ABBGKResult` discard the auxiliary sector that the collision unitary
    populates.

    Qubits are measured in the order ``[grid, velocity, marker]``, so a Qiskit count
    string reversed into least-significant-bit-first order splits into the :math:`x`
    coordinate, the :math:`y` coordinate, the velocity index, and the marker.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab.collision import ABBGKMeasurement
        from qlbm.lattice import ABBGKLattice

        lattice = ABBGKLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABBGKMeasurement(lattice).draw("mpl")

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`lattice`
          - The :class:`.ABBGKLattice` based on which the properties of the component are inferred.
        * - :attr:`logger`
          - The performance logger, by default ``getLogger("qlbm")``.
    """

    lattice: ABBGKLattice
    """The lattice to construct the component for."""

    def __init__(
        self,
        lattice: ABBGKLattice,
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

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        measured_qubits = (
            self.lattice.grid_index()
            + self.lattice.velocity_index()
            + self.lattice.marker_index()
        )

        circuit.add_register(ClassicalRegister(len(measured_qubits)))
        circuit.measure(measured_qubits, list(range(len(measured_qubits))))

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABBGKMeasurement with lattice {self.lattice}]"
