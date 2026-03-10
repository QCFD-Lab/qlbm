"""Zone-agnostic reflection utilities for the :class:`.ABQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, cast

from qiskit import QuantumCircuit
from qiskit.circuit.library import RGQFTMultiplier
from typing_extensions import override

from qlbm.components.ab.reflection.common import ABReflectionPermutation
from qlbm.components.ab.streaming import ABStreamingOperator
from qlbm.components.base import LBMOperator, LBMPrimitive
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
from qlbm.tools.utils import ComparatorMode, flatten, get_qubits_to_invert


class ABZoneAgnosticReflectionOperator(LBMOperator):
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
        super().__init__(lattice, logger)

        self.shapes = (
            (
                flatten(list(self.lattice.geometries[0].values()))
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

        # For multi-geometry, self.shapes is a list of lists;
        # for single geometry, it is a flat list.
        all_shapes = (
            flatten(self.shapes)
            if self.lattice.has_multiple_geometries() and shapes is None
            else self.shapes
        )

        if any([x.name() not in supported_shapes for x in all_shapes]):  # type: ignore
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

        if not self.lattice.has_multiple_geometries():
            return self.__create_circuit_single_geometry()
        else:
            return self.__create_circuit_multi_geometry()

    def __create_circuit_single_geometry(self) -> QuantumCircuit:
        r"""Create the zone-agnostic reflection circuit for a single geometry.

        The circuit structure is:

        .. math::

            U = S \cdot O \cdot S^{-1} \cdot (Perm \cdot S)_{\text{ctrl}\ a_o} \cdot O

        where :math:`O` is the oracle, :math:`S` is the streaming operator,
        and :math:`Perm` is the velocity permutation.
        """
        circuit = self.lattice.circuit.copy()

        oracle = self.lattice.circuit.copy()
        for shape in self.shapes:
            oracle.compose(
                ABZoneAgnosticReflectionOracle(
                    self.lattice, shape, logger=self.logger  # type: ignore
                ).circuit,
                inplace=True,
            )

        circuit.compose(oracle, inplace=True)
        circuit.compose(self.permute_and_stream(), inplace=True)
        circuit.compose(
            ABStreamingOperator(self.lattice, logger=self.logger).circuit.inverse(),
            inplace=True,
        )
        circuit.compose(oracle, inplace=True)
        circuit.compose(
            ABStreamingOperator(self.lattice, logger=self.logger).circuit,
            inplace=True,
        )

        return circuit

    def __create_circuit_multi_geometry(self) -> QuantumCircuit:
        r"""Create the zone-agnostic reflection circuit for multiple geometries.

        For :math:`m` geometries, a combined oracle :math:`O_{\text{combined}}`
        is built by applying each geometry's oracle :math:`O_c` controlled on
        the marker register being in state :math:`\ket{c}`.
        Since different marker states occupy orthogonal subspaces, the oracles
        do not interfere and the obstacle ancilla is correctly set for each
        geometry independently.

        The circuit structure is:

        .. math::

            U = S \cdot O_\text{combined} \cdot S^{-1}
                \cdot (Perm \cdot S)_{\text{ctrl}\ a_o}
                \cdot O_\text{combined}

        Only the oracle is controlled on the marker state; the permutation,
        streaming, and inverse streaming are shared across all geometries.
        The permutation and streaming are implicitly geometry-specific because
        they are controlled on the obstacle ancilla, which the marker-controlled
        oracle has already set correctly.
        """
        circuit = self.lattice.circuit.copy()

        oracle = self.build_combined_oracle()

        circuit.compose(oracle, inplace=True)
        circuit.compose(self.permute_and_stream(), inplace=True)
        circuit.compose(
            ABStreamingOperator(self.lattice, logger=self.logger).circuit.inverse(),
            inplace=True,
        )
        circuit.compose(oracle, inplace=True)
        circuit.compose(
            ABStreamingOperator(self.lattice, logger=self.logger).circuit,
            inplace=True,
        )

        return circuit

    def build_combined_oracle(self) -> QuantumCircuit:
        r"""Build the combined oracle for all geometries.

        For each geometry index :math:`c`, the marker register qubits are
        flipped so that geometry :math:`c` maps to the all-ones state.
        The oracle for that geometry is then applied with its central MCX gate
        additionally controlled on the marker register.
        Finally, the marker qubits are unflipped to restore the original state.

        Returns
        -------
        QuantumCircuit
            The combined oracle circuit.
        """
        oracle = self.lattice.circuit.copy()

        for c, shapes_for_geometry in enumerate(self.shapes):
            qubits_to_invert = [
                q + self.lattice.marker_index()[0]
                for q in get_qubits_to_invert(c, self.lattice.num_marker_qubits)
            ]

            if qubits_to_invert:
                oracle.x(qubits_to_invert)

            for shape in shapes_for_geometry:
                oracle.compose(
                    ABZoneAgnosticReflectionOracle(
                        self.lattice,  # type: ignore[arg-type]
                        shape,
                        control_on_marker_state=True,
                        logger=self.logger,
                    ).circuit,
                    inplace=True,
                )

            if qubits_to_invert:
                oracle.x(qubits_to_invert)

        return oracle

    def permute_and_stream(self) -> QuantumCircuit:
        """
        Performs the permutation of basis states that implements bounceback reflection in the amplitude-based encoding.

        Returns
        -------
        QuantumCircuit
            The permutation acting on only the velocity register.
        """
        circuit = self.lattice.circuit.copy()

        # Permute the velocities according to reflection rules
        circuit.compose(
            ABReflectionPermutation(
                self.lattice.num_velocity_qubits,
                self.lattice.discretization,
                self.lattice.get_encoding(),
                self.logger,
            )
            .circuit.control(1)
            .decompose(),
            qubits=self.lattice.ancillae_obstacle_index()
            + self.lattice.velocity_index(),
            inplace=True,
        )

        circuit.compose(
            ABStreamingOperator(
                self.lattice, self.lattice.ancillae_obstacle_index(), self.logger
            ).circuit,
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

    When ``control_on_marker_state`` is ``True``, the oracle additionally conditions
    the obstacle ancilla flip on the marker register being in the all-ones state.
    This is used for parallel boundary conditions where multiple geometries
    are simulated on the same lattice, each identified by a marker state.
    Only the central MCX gate (for cuboids) is controlled on the marker,
    since the surrounding adder and comparator operations are self-inverse
    and their net effect on the grid register is zero.

    .. important::

        Marker-controlled oracles for :class:`.YMonomial` shapes are not yet supported.
        Passing ``control_on_marker_state=True`` with a ``YMonomial`` shape will raise
        a :class:`.CircuitException`.

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

    control_on_marker_state: bool
    """Whether the oracle is additionally controlled on the marker register."""

    def __init__(
        self,
        lattice: ABLattice,
        shape: Shape,
        control_on_marker_state: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.lattice = lattice
        self.shape = shape
        self.control_on_marker_state = control_on_marker_state

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
            if self.control_on_marker_state:
                raise CircuitException(
                    "Marker-controlled oracles for YMonomial shapes are not yet supported. "
                    "Parallel boundary conditions with YMonomial geometries require a future extension."
                )
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

        control_qubits = self.lattice.ancillae_comparator_index(0)[
            : self.lattice.num_dims
        ]

        if self.control_on_marker_state:
            control_qubits = control_qubits + self.lattice.marker_index()

        circuit.mcx(
            control_qubits,
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
