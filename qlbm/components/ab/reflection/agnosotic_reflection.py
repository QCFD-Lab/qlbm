"""Zone-agnostic reflection utilities for the :class:`.ABQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, cast

from qiskit import QuantumCircuit
from qiskit.circuit.library import RGQFTMultiplier
from typing_extensions import override

from qlbm.components.ab.reflection.standard_reflection import ABReflectionOperator
from qlbm.components.ab.streaming import ABStreamingOperator
from qlbm.components.base import LBMPrimitive
from qlbm.components.common.adders import ParameterizedDraperAdder
from qlbm.components.common.comparators import (
    SingleRegisterComparator,
    TwoRegisterComparator,
)
from qlbm.lattice.geometry.shapes import Block, Circle, YMonomial
from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.lattices.base import AmplitudeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import CircuitException, LatticeException
from qlbm.tools.utils import ComparatorMode, flatten


class ABZoneAgnosticReflectionOperator(ABReflectionOperator):
    """
    Implements bounceback reflection in the amplitude-based encoding of :class:`.ABQLBM` for :math:`D_dQ_q` discretizations.

    Uses a zone-agnostic approach that relies on the existence of an oracle that marks the
    basis states belonging to the inside of the solid geometry.
    For more details on the oracle, see :class:`.ABZoneAgnosticReflectionOracle`.

    Example usage:

    .. plot::
        :include-source:

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

        ABZoneAgnosticReflectionOperator(lattice, shapes=lattice.shapes["bounceback"]).draw("mpl")

    """

    lattice: AmplitudeLattice

    def __init__(
        self,
        lattice: ABLattice,
        shapes: List[Shape] | None = None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, [], logger)

        self.shapes = (
            (
                cast(List[Block], flatten(list(self.lattice.geometries[0].values())))
                if not self.lattice.has_multiple_geometries()
                else [
                    gdict["bounceback"] + gdict["specular"]  # type: ignore
                    for gdict in self.lattice.geometries  # type: ignore
                ]
            )
            if shapes is None
            else shapes
        )

        supported_shapes = ["cuboid", "ymonomial"]

        if any([x.name() not in supported_shapes for x in self.shapes]):  # type: ignore
            raise CircuitException(
                f"Agnostic reflection operator only supports the following shapes: {supported_shapes}."
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
        for shape in self.shapes:
            oracle.compose(
                ABZoneAgnosticReflectionOracle(
                    self.lattice, shape, logger=self.logger  # type: ignore
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

    An oracle is an operator :math:`U_{\Omega}` for an obstacle's region
    :math:`\Omega` such that, in the amplitude-based encoding,
    :math:`U_\Omega\ket{x}\ket{v}\ket{0}_\mathbb{o} = \ket{x}\ket{v}\ket{x \in \Omega}_\mathbb{o}`.
    Intuitively, the operator flips the object ancilla qubit if and only if the position :math:`x`
    falls within the bounds of the object.

    Currently, the only available implementation is for 2D axis-aligned :class:`.Block`
    and :class:`.YMonomial` objects.

    .. important::

        The ``YMonomial`` implementation is a work in progress.
        At present, only the :math:`x^2` monomial case is supported,
        and only when the monomial result register width matches the :math:`y`
        grid register width.

    This is an improvement in asymptotic and practical complexity compared to
    the methods described in :cite:`collisionless`.
    This operation relies on basic arithmetic through the :class:`.ParameterizedDraperAdder` class
    and comparison operation through the :class:`Comparator` circuits.

    Example usage for a cuboid :class:`.Block`:

    .. plot::
        :include-source:

        from qlbm.components.ab.reflection import ABZoneAgnosticReflectionOracle
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 16}, "velocities": "d2q9"},
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

        ABZoneAgnosticReflectionOracle(lattice, shape=lattice.shapes["bounceback"][0]).draw("mpl")

    And for a :class:`.YMonomial`:

    .. plot::
        :include-source:

        from qlbm.components.ab.reflection import ABZoneAgnosticReflectionOracle
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 16}, "velocities": "d2q9"},
                "geometry": [
                    {
                        "shape": "ymonomial",
                        "exponent": 2,
                        "comparator": "<",
                        "boundary": "bounceback",
                    }
                ],
            }
        )

        ABZoneAgnosticReflectionOracle(lattice, shape=lattice.shapes["bounceback"][0]).draw("mpl")


    """

    lattice: ABLattice

    def __init__(
        self,
        lattice: ABLattice,
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
        elif isinstance(self.shape, YMonomial):
            return self.__create_circuit_ymonomial()

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
                SingleRegisterComparator(
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
                SingleRegisterComparator(
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

    def __create_circuit_ymonomial(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        ym: YMonomial = cast(YMonomial, self.shape)

        if ym.exponent != 2:
            raise CircuitException(
                "YMonomial oracle is a work in progress: only exponent=2 (x^2) is currently supported."
            )

        # Qubits used in this oracle
        grid_x_qubits = self.lattice.grid_index(0)
        grid_y_qubits = self.lattice.grid_index(1)
        copy_qubits = self.lattice.ancillae_copy_index()
        result_qubits = self.lattice.ancillae_monomial_index()

        if len(result_qubits) != len(grid_y_qubits):
            raise CircuitException(
                "YMonomial oracle is a work in progress: only configurations with equal y and monomial result register sizes are currently supported."
            )

        # circuits used more than once
        multiplication_circuit = RGQFTMultiplier(
            num_state_qubits=len(grid_x_qubits),
            num_result_qubits=len(result_qubits),
        )

        comparator_circuit = TwoRegisterComparator(
            len(grid_y_qubits), ym.comparator_mode
        ).circuit

        # Copy x into the copy register
        for qc, qt in zip(grid_x_qubits, copy_qubits):
            circuit.cx(qc, qt)

        # Do the multiplication
        circuit.compose(
            multiplication_circuit,
            qubits=grid_x_qubits + copy_qubits + result_qubits,
            inplace=True,
        )

        # Comparator
        circuit.compose(
            comparator_circuit,
            qubits=grid_y_qubits
            + result_qubits
            + self.lattice.ancillae_obstacle_index(),
            inplace=True,
        )

        # Undo multiplication
        circuit.compose(
            multiplication_circuit.inverse(),
            qubits=grid_x_qubits + copy_qubits + result_qubits,
            inplace=True,
        )

        # Undo copy
        for qc, qt in zip(grid_x_qubits, copy_qubits):
            circuit.cx(qc, qt)

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABZoneAgnosticReflectionOracle with lattice {self.lattice}, shape={self.shape}]"
