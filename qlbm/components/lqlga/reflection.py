"""Reflection operator for the :class:`.LQLGA` algorithm :cite:`spacetime` that swaps particles one gridpoint at a time."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, cast

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.base import LQLGAOperator
from qlbm.lattice.geometry.shapes.base import LQLGAShape, Shape
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import CircuitException


class LQLGAReflectionOperator(LQLGAOperator):
    def __init__(
        self,
        lattice: LQLGALattice,
        shapes: List[Shape],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.shapes = cast(List[LQLGAShape], shapes)

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        discretization = self.lattice.discretization
        if discretization == LatticeDiscretization.D1Q2:
            return self.__create_circuit_d1q2()

        raise CircuitException(f"Reflection Operator unsupported for {discretization}.")

    def __create_circuit_d1q2(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        for shape in self.shapes:
            for reflection_data in shape.get_lqlga_reflection_data_d1q2():
                circuit.swap(
                    self.lattice.velocity_index_tuple(
                        reflection_data.gridpoints[0],
                        reflection_data.velocity_indices_to_swap[0],
                    ),
                    self.lattice.velocity_index_tuple(
                        reflection_data.gridpoints[1],
                        reflection_data.velocity_indices_to_swap[1],
                    ),
                )
        return circuit

    @override
    def __str__(self) -> str:
        return f"[PointWiseLQLGAReflectionOperator for lattice {self.lattice}, shapes {self.shapes}]"
