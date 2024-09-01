from logging import Logger, getLogger
from typing import List, Tuple, cast

from qiskit import QuantumCircuit as QiskitQC
from qiskit.quantum_info import Statevector
from qiskit.result import Counts
from qiskit_aer.backends.aer_simulator import AerBackend
from qulacs import QuantumCircuit as QulacsQC

from qlbm.components.spacetime import SpaceTimeInitialConditions
from qlbm.infra.compiler import CircuitCompiler
from qlbm.lattice import SpaceTimeLattice

from .base import Reinitializer


class SpaceTimeReinitializer(Reinitializer):
    lattice: SpaceTimeLattice
    counts: Counts

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        compiler: CircuitCompiler,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(lattice, compiler, logger)
        self.lattice = lattice
        self.logger = logger
        self.x_grid_qubits = self.lattice.num_gridpoints[0].bit_length()
        self.y_grid_qubits = self.lattice.num_gridpoints[1].bit_length()

    def reinitialize(
        self,
        statevector: Statevector,
        counts: Counts,
        backend: AerBackend | None,
        optimization_level: int = 0,
    ) -> QiskitQC | QulacsQC:
        return self.compiler.compile(
            SpaceTimeInitialConditions(
                self.lattice, self.counts_to_velocity_pairs(counts)
            ),
            backend=backend,
            optimization_level=optimization_level,
        )

    def counts_to_velocity_pairs(
        self,
        counts: Counts,
    ) -> List[Tuple[Tuple[int, int], Tuple[bool, bool, bool, bool]]]:
        return [self.split_count(count) for count in counts if int(count[:4], 2) > 0]

    def split_count(
        self, count: str
    ) -> Tuple[Tuple[int, int], Tuple[bool, bool, bool, bool]]:
        inverse_count = count[::-1]
        return (
            (
                int(inverse_count[: self.x_grid_qubits][::-1], 2),
                int(
                    inverse_count[
                        self.x_grid_qubits : self.x_grid_qubits + self.y_grid_qubits
                    ][::-1],
                    2,
                ),
            ),
            cast(
                Tuple[bool, bool, bool, bool],
                tuple(
                    bool(int(x, 2))
                    for x in inverse_count[self.x_grid_qubits + self.y_grid_qubits :]
                ),
            ),
        )

    def requires_statevector(self) -> bool:
        return False
