"""Zone-agnostic reflection utilities for the :class:`.ABQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, cast

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.ab.reflection.standard_reflection import ABReflectionOperator
from qlbm.components.ab.streaming import ABStreamingOperator
from qlbm.components.base import LBMPrimitive
from qlbm.components.common.adders import ParameterizedDraperAdder
from qlbm.components.ms.primitives import Comparator, ComparatorMode
from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.geometry.shapes.circle import Circle
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.lattices.base import AmplitudeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import CircuitException, LatticeException
from qlbm.tools.utils import flatten


class ABZoneAgnosticReflectionOperator(ABReflectionOperator):
    """
    Implements bounceback reflection in the amplitude-based encoding of :class:`.ABQLBM` for :math:`D_dQ_q` discretizations.

    Uses a zone-agnostic approach that relies on the existence of an oracle that marks the
    basis states belonging to the inside of the solid geometry.
    For more details on the oracle, see :class:`.ABZoneAgnosticReflectionOracle`.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABZoneAgnosticReflectionOperator
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
                "geometry": [
                    {
                        "shape": "cuboid",
                        "x": [1, 3],
                        "y": [1, 3],
                        "boundary": "bounceback",
                    }
                ],
            }
        )

        ABZoneAgnosticReflectionOperator(lattice, blocks=lattice.shapes["bounceback"]).draw("mpl")

    """

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
        if self.lattice.discretization not in [LatticeDiscretization.D2Q9]:
            raise LatticeException("AB reflection only currently supported in D2Q9")
        circuit = self.lattice.circuit.copy()

        oracle = self.lattice.circuit.copy()
        # build the oracle once
        for block in self.blocks:
            oracle.compose(
                ABZoneAgnosticReflectionOracle(
                    self.lattice, block, logger=self.logger
                ).circuit,
                inplace=True,
            )

        # 2-3. oracle
        circuit.compose(oracle, inplace=True)

        # 3-4. controlled permutation and stream
        circuit.compose(self.permute_and_stream(), inplace=True)

        # 4-5. uncontrolled inverse stream
        circuit.compose(
            ABStreamingOperator(self.lattice, logger=self.logger).circuit.inverse(),
            inplace=True,
        )

        # 5-6. oracle
        circuit.compose(oracle, inplace=True)

        # 6-7. uncontrolled regular stream
        circuit.compose(
            ABStreamingOperator(self.lattice, logger=self.logger).circuit,
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Operator ABZoneAgnosticReflection with lattice {self.lattice}]"


class ABZoneAgnosticReflectionOracle(LBMPrimitive):
    r"""
    Implementation of the oracle required for :class:`.ABZoneAgnosticReflectionOperator`.

    An oracle is an operator :math:`U_{\omega}` for an obstacle's region
    :math:`\omega` such that, in the amplitude-based encoding,
    :math:`U_\omega\ket{x}\ket{v}\ket{0}_\mathbb{o} = \ket{x}\ket{v}\ket{x \in \omega}_\mathbb{o}`.
    Intuitively, the operator flips the object ancilla qubit if and only if the position :math:`x`
    falls within the bounds of the object.

    Currently, the only available implementation is for 2D axis-aligned objects.
    This is an improvement in asymptotic and practical complexity compared to
    the methods described in :cite:`collisionless`.
    This operation relies on basic arithmetic through the :class:`.ParameterizedDraperAdder` class
    and comparison operation through the :class:`Comparator` circuits.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABZoneAgnosticReflectionOperator
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
                "geometry": [
                    {
                        "shape": "cuboid",
                        "x": [1, 3],
                        "y": [1, 3],
                        "boundary": "bounceback",
                    }
                ],
            }
        )

        ABZoneAgnosticReflectionOperator(lattice, blocks=lattice.shapes["bounceback"]).draw("mpl")

    """

    lattice: AmplitudeLattice

    def __init__(
        self,
        lattice: AmplitudeLattice,
        shape: Shape,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.lattice = lattice
        self.shape = shape

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        if isinstance(self.shape, Block):
            return self.__create_circuit_block()
        elif isinstance(self.shape, Circle):
            return self.__create_circuit_circle()

    def __create_circuit_block(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        block: Block = cast(Block, self.shape)

        for dim in range(self.lattice.num_dims):
            circuit.compose(
                ParameterizedDraperAdder(
                    len(self.lattice.grid_index(dim)),
                    block.bounds[dim][0],
                    positive=False,
                ).circuit,
                qubits=self.lattice.grid_index(dim),
                inplace=True,
            )

            circuit.compose(
                Comparator(
                    num_qubits=len(self.lattice.grid_index(dim)) + 1,
                    num_to_compare=block.bounds[dim][1] - block.bounds[dim][0],
                    mode=ComparatorMode.LE,
                ).circuit,
                qubits=self.lattice.grid_index(dim)
                + [self.lattice.ancillae_comparator_index(0)[dim]],
                inplace=True,
            )

        circuit.mcx(
            self.lattice.ancillae_comparator_index(0)[: self.lattice.num_dims],
            self.lattice.ancillae_obstacle_index()[0],
        )

        for dim in range(self.lattice.num_dims):
            circuit.compose(
                Comparator(
                    num_qubits=len(self.lattice.grid_index(dim)) + 1,
                    num_to_compare=block.bounds[dim][1] - block.bounds[dim][0],
                    mode=ComparatorMode.LE,
                ).circuit,
                qubits=self.lattice.grid_index(dim)
                + [self.lattice.ancillae_comparator_index(0)[dim]],
                inplace=True,
            )

            circuit.compose(
                ParameterizedDraperAdder(
                    len(self.lattice.grid_index(dim)),
                    block.bounds[dim][0],
                    positive=True,
                ).circuit,
                qubits=self.lattice.grid_index(dim),
                inplace=True,
            )

        return circuit

    def __create_circuit_circle(self) -> QuantumCircuit:
        raise CircuitException("Not implemented")

    @override
    def __str__(self) -> str:
        return f"[Primitive ABZoneAgnosticReflectionOracle with lattice {self.lattice}, shape={self.shape}]"
