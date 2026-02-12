from qlbm.components.ab.reflection.standard_reflection import ABReflectionOperator
from qlbm.components.ab.streaming import ABStreamingOperator
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.lattices.base import AmplitudeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import flatten


from qiskit import QuantumCircuit


from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, cast
from typing_extensions import override


class ABZoneAgnosticReflectionOperator(ABReflectionOperator):
    lattice: AmplitudeLattice

    def __init__(
        self,
        lattice: ABLattice,
        blocks: List[Block] | None = None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, blocks, logger)

        self.blocks = (
            (
                cast(List[Block], flatten(list(self.lattice.geometries[0].values())))
                if not self.lattice.has_multiple_geometries()
                else [
                    gdict["bounceback"] + gdict["specular"]  # type: ignore
                    for gdict in self.lattice.geometries  # type: ignore
                ]
            )
            if blocks is None
            else blocks
        )

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        print("Ok!")

        if self.lattice.discretization not in [LatticeDiscretization.D2Q9]:
            raise LatticeException("AB reflection only currently supported in D2Q9")
        circuit = self.lattice.circuit.copy()

        # 2-3. oracle
        for block in self.blocks:
            circuit.compose(
                self.set_inside_wall_ancilla_state(
                    block, control_on_marker_state=False
                ),
                inplace=True,
            )

        circuit.compose(
            self.set_ancilla_of_point_state(
                flatten(
                    [[(p, None) for p in block.corners_inside] for block in self.blocks]
                ),
                ignore_velocity_data=True,
                control_on_marker_state=False,
            ),
            inplace=True,
        )

        # 3-4. controlled permutation and stream
        circuit.compose(self.permute_and_stream(), inplace=True)

        # 4-5. uncontrolled inverse stream
        circuit.compose(
            ABStreamingOperator(self.lattice, logger=self.logger).circuit.inverse(),
            inplace=True,
        )

        # 5-6. oracle
        for block in self.blocks:
            circuit.compose(
                self.set_inside_wall_ancilla_state(
                    block, control_on_marker_state=False
                ),
                inplace=True,
            )

        circuit.compose(
            self.set_ancilla_of_point_state(
                flatten(
                    [[(p, None) for p in block.corners_inside] for block in self.blocks]
                ),
                ignore_velocity_data=True,
                control_on_marker_state=False,
            ),
            inplace=True,
        )

        # 6-7. uncontrolled regular stream
        circuit.compose(
            ABStreamingOperator(self.lattice, logger=self.logger).circuit,
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Operator ABZoneAgnosticReflection with lattice {self.lattice}]"