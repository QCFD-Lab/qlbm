"""Zone-agnostic reflection utilities for the :class:`.ABQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import Dict, List, Tuple, cast

from qiskit import QuantumCircuit
from qiskit.circuit.library import RGQFTMultiplier
from qiskit.synthesis import synth_qft_full as QFT
from typing_extensions import override

from qlbm.components.ab.reflection.common import (
    ABBounceBackReflectionPermutation,
    ABSpecularReflectionPermutation,
)
from qlbm.components.ab.streaming import ABStreamingOperator
from qlbm.components.base import LBMOperator, LBMPrimitive
from qlbm.components.common.adders import ParameterizedDraperAdder, PhaseShift
from qlbm.components.common.comparators import (
    SingleRegisterComparator,
    TwoRegisterComparator,
)
from qlbm.lattice.geometry.shapes import Block, Circle, YMonomial
from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.lattices.ab_lattice import ABLattice
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

        ABZoneAgnosticReflectionOperator(lattice).draw("mpl")

    """

    lattice: ABLattice

    shapes: Dict[str, List[Shape]]

    def __init__(
        self,
        lattice: ABLattice,
        shapes: Dict[str, List[Shape]] | None = None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)

        if shapes is None:
            if self.lattice.has_multiple_geometries():
                self.markered_shapes = self.lattice.geometries
                raise CircuitException(
                    "Multigeometry only currently supported for standard boundary condition imposition."
                )
            else:
                self.shapes = self.lattice.geometries[0]

        else:
            self.shapes = shapes

        supported_shapes = ["cuboid", "ymonomial"]

        all_shapes = (
            flatten(flatten(g.values() for g in self.markered_shapes))
            if self.lattice.has_multiple_geometries() and shapes is None
            else flatten(self.shapes.values())
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
        return self.__create_circuit_single_geometry_bounceback().compose(
            self.__create_circuit_single_geometry_sr()
        )

    def __create_circuit_single_geometry_bounceback(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        if ("bounceback" not in self.shapes) or (not self.shapes["bounceback"]):
            return circuit

        oracle = self.lattice.circuit.copy()
        for shape in self.shapes["bounceback"]:
            oracle.compose(
                ABZoneAgnosticReflectionOracle(
                    self.lattice, shape, logger=self.logger  # type: ignore
                ).circuit,
                inplace=True,
            )
        circuit.compose(oracle, inplace=True)
        circuit.compose(self.permute_and_stream_bounceback(), inplace=True)
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

    def __create_circuit_single_geometry_sr(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        if "specular" not in self.shapes or (len(self.shapes["specular"]) == 0):
            return circuit

        oracle = self.__build_oracle("specular")

        # Step 1: Oracle (sets a_o)
        circuit.compose(oracle, inplace=True)
        # Step 2: SR check (sets a_x, a_y for diagonals)
        circuit.compose(
            ABZoneAgnosticSRCheck(
                self.lattice,
                self.lattice.discretization,
                self.shapes["specular"],
                check_negative_direction=True,
                logger=self.logger,
            ).circuit,
            inplace=True,
        )
        # Step 3: Permutations (velocity changes, position unchanged)
        circuit.compose(self.__apply_permutations_sr(), inplace=True)
        # Step 4: Dim-selective stream for diagonals (ctrl a_o AND a_{d+1})
        circuit.compose(self.__dim_selective_stream(), inplace=True)

        # Step 5: Inverse stream
        circuit.compose(
            ABStreamingOperator(self.lattice, logger=self.logger).circuit.inverse(),
            inplace=True,
        )

        # Step 6: SR check in the positive direction
        circuit.compose(
            ABZoneAgnosticSRCheck(
                self.lattice,
                self.lattice.discretization,
                self.shapes["specular"],
                check_negative_direction=False,
                logger=self.logger,
            ).circuit,
            inplace=True,
        )
        # Step 7: Oracle (unitarily uncomputes a_o)
        circuit.compose(oracle, inplace=True)
        # Step 8: Stream
        circuit.compose(
            ABStreamingOperator(self.lattice, logger=self.logger).circuit,
            inplace=True,
        )

        return circuit

    def __build_oracle(self, boundary_condition) -> QuantumCircuit:
        """Build the composite oracle for all shapes, targeting a_o (index 0)."""
        oracle = self.lattice.circuit.copy()
        for shape in self.shapes[boundary_condition]:
            oracle.compose(
                ABZoneAgnosticReflectionOracle(
                    self.lattice, shape, logger=self.logger  # type: ignore
                ).circuit,
                inplace=True,
            )
        return oracle

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

        oracle = self.build_combined_oracle("bounceback")

        circuit.compose(oracle, inplace=True)
        circuit.compose(self.permute_and_stream_bounceback(), inplace=True)
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

    def build_combined_oracle(self, boundary_condition: str) -> QuantumCircuit:
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

        for c, shapes_for_geometry in enumerate(self.markered_shapes):
            qubits_to_invert = [
                q + self.lattice.marker_index()[0]
                for q in get_qubits_to_invert(c, self.lattice.num_marker_qubits)
            ]

            if qubits_to_invert:
                oracle.x(qubits_to_invert)

            for shape in shapes_for_geometry[boundary_condition]:
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

    def __apply_permutations_sr(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        a_x = self.lattice.ancillae_obstacle_index(1)
        a_y = self.lattice.ancillae_obstacle_index(2)
        a_xy = self.lattice.ancillae_obstacle_index(3)
        all_obstacle = self.lattice.ancillae_obstacle_index()
        vel = self.lattice.velocity_index()

        def controlled_perm(perm_circuit: QuantumCircuit) -> None:
            circuit.compose(
                perm_circuit.control(self.lattice.num_obstacle_qubits).decompose(),
                qubits=all_obstacle + vel,
                inplace=True,
            )

        circuit.x(a_xy)

        circuit.x(a_y)
        controlled_perm(
            ABSpecularReflectionPermutation(
                self.lattice.num_velocity_qubits,
                self.lattice.discretization,
                self.lattice.get_encoding(),
                (True, False),
                self.logger,
            ).circuit
        )
        circuit.x(a_y)

        circuit.x(a_x)
        controlled_perm(
            ABSpecularReflectionPermutation(
                self.lattice.num_velocity_qubits,
                self.lattice.discretization,
                self.lattice.get_encoding(),
                (False, True),
                self.logger,
            ).circuit
        )
        circuit.x(a_x)

        # Case (1,1,1): corner hit, reflect both
        controlled_perm(
            ABSpecularReflectionPermutation(
                self.lattice.num_velocity_qubits,
                self.lattice.discretization,
                self.lattice.get_encoding(),
                (True, True),
                self.logger,
            ).circuit
        )

        circuit.x(a_xy)

        # Case (1,0,0): cardinal velocity, reflect both (BB equivalent)
        # Need a_o=1, a_x=1, a_y=1, so flip both a_x and a_y
        circuit.x(a_x + a_y)
        controlled_perm(
            ABBounceBackReflectionPermutation(
                self.lattice.num_velocity_qubits,
                self.lattice.discretization,
                self.lattice.get_encoding(),
                self.logger,
            ).circuit
        )
        circuit.x(a_x + a_y)

        return circuit

    def __dim_selective_stream(self) -> QuantumCircuit:
        r"""Stream reflected particles back out, only in the dimension(s) that caused reflection.

        For each dimension :math:`d`, applies the streaming operator for
        that dimension only, controlled on :math:`a_o \wedge a_{d+1}`.
        This ensures a particle reflected off an x-wall is streamed back in x,
        a particle reflected off a y-wall is streamed back in y, and a corner-reflected
        particle is streamed back in both dimensions.

        Returns
        -------
        QuantumCircuit
            Circuit performing dimension-selective controlled streaming.
        """
        circuit = self.lattice.circuit.copy()

        dim_indices = [
            [
                [1, 5, 8],  # x <- x + 1
                [3, 6, 7],  # x <- x - 1
            ],
            [
                [2, 5, 6],  # y <- y + 1
                [4, 7, 8],  # y <- y - 1
            ],
        ]

        diagonal_velocities: Dict[int, Tuple[bool, ...]] = {
            5: tuple([True, True]),
            6: tuple([False, True]),
            7: tuple([False, False]),
            8: tuple([True, False]),
        }

        circuit.x(self.lattice.ancillae_obstacle_index()[-1])

        for dim in range(self.lattice.num_dims):
            control_qubits = (
                self.lattice.ancillae_obstacle_index(0)  # a_{o, 0}
                + self.lattice.ancillae_obstacle_index(dim + 1)  # a_{o, x/y}
                + [self.lattice.ancillae_obstacle_index()[-1]]  # a_{o, xy}
            )

            circuit.compose(
                QFT(len(self.lattice.grid_index(dim))),
                qubits=self.lattice.grid_index(dim),
                inplace=True,
            )

            for direction, indices in enumerate(dim_indices[dim]):
                positive = bool(1 - direction)
                for index in indices:
                    velocity_inversion_qubits = [
                        self.lattice.num_grid_qubits + q
                        for q in get_qubits_to_invert(
                            index, self.lattice.num_velocity_qubits
                        )
                    ]
                    if velocity_inversion_qubits:
                        circuit.x(velocity_inversion_qubits)

                    circuit.compose(
                        PhaseShift(
                            num_qubits=len(self.lattice.grid_index(dim)),
                            positive=positive,
                            logger=self.logger,
                        )
                        .circuit.control(
                            self.lattice.num_velocity_qubits + len(control_qubits)
                        )
                        .decompose(),
                        qubits=control_qubits
                        + self.lattice.velocity_index()
                        + self.lattice.grid_index(dim),
                        inplace=True,
                    )

                    if velocity_inversion_qubits:
                        circuit.x(velocity_inversion_qubits)

        # Now diagonal velocities that hit a concave corner
        circuit.x(self.lattice.ancillae_obstacle_index()[-1])
        # Now we control on |001>
        circuit.x(self.lattice.ancillae_obstacle_index()[1:-1])
        for diagonal_velocity in diagonal_velocities:
            control_qubits = self.lattice.ancillae_obstacle_index()
            for dim in range(self.lattice.num_dims):
                positive = diagonal_velocities[diagonal_velocity][dim]

                velocity_inversion_qubits = [
                    self.lattice.num_grid_qubits + q
                    for q in get_qubits_to_invert(
                        diagonal_velocity, self.lattice.num_velocity_qubits
                    )
                ]
                if velocity_inversion_qubits:
                    circuit.x(velocity_inversion_qubits)

                circuit.compose(
                    PhaseShift(
                        num_qubits=len(self.lattice.grid_index(dim)),
                        positive=positive,
                        logger=self.logger,
                    )
                    .circuit.control(
                        self.lattice.num_velocity_qubits + len(control_qubits)
                    )
                    .decompose(),
                    qubits=control_qubits
                    + self.lattice.velocity_index()
                    + self.lattice.grid_index(dim),
                    inplace=True,
                )

                if velocity_inversion_qubits:
                    circuit.x(velocity_inversion_qubits)

        circuit.x(self.lattice.ancillae_obstacle_index()[1:-1])

        for dim in range(self.lattice.num_dims):
            circuit.compose(
                QFT(len(self.lattice.grid_index(dim)), inverse=True),
                qubits=self.lattice.grid_index(dim),
                inplace=True,
            )

        return circuit

    def permute_and_stream_bounceback(self) -> QuantumCircuit:
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
            ABBounceBackReflectionPermutation(
                self.lattice.num_velocity_qubits,
                self.lattice.discretization,
                self.lattice.get_encoding(),
                self.logger,
            )
            .circuit.control(1)
            .decompose(),
            qubits=self.lattice.ancillae_obstacle_index(0)
            + self.lattice.velocity_index(),
            inplace=True,
        )

        circuit.compose(
            ABStreamingOperator(
                self.lattice, self.lattice.ancillae_obstacle_index(0), self.logger
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

    target_obstacle_index: int
    """Index within the obstacle ancilla register to target with the oracle flip."""

    def __init__(
        self,
        lattice: ABLattice,
        shape: Shape,
        control_on_marker_state: bool = False,
        additional_control_qubits: List[int] = [],
        target_obstacle_index: int = 0,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.lattice = lattice
        self.shape = shape
        self.control_on_marker_state = control_on_marker_state
        self.additional_control_qubits = additional_control_qubits
        self.target_obstacle_index = target_obstacle_index

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

        control_qubits = (
            self.lattice.ancillae_comparator_index(0)[: self.lattice.num_dims]
            + self.additional_control_qubits
        )

        if self.control_on_marker_state:
            control_qubits = control_qubits + self.lattice.marker_index()

        circuit.mcx(
            control_qubits,
            self.lattice.ancillae_obstacle_index(self.target_obstacle_index)[0],
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

        n_y = len(grid_y_qubits)
        n_monomial = len(result_qubits)
        n_copy = len(copy_qubits)
        needs_padding = n_monomial != n_y

        if needs_padding:
            padding_needed = abs(n_monomial - n_y)
            if padding_needed > n_copy:
                raise CircuitException(
                    f"YMonomial oracle: register size mismatch requires "
                    f"{padding_needed} padding qubits but only {n_copy} "
                    f"copy-register qubits are available. "
                    f"Grid must satisfy |2*n_x - n_y| <= n_x."
                )
            self.logger.warning(
                "YMonomial oracle: monomial register (%d qubits) differs "
                "from y grid register (%d qubits). Using %d copy-register "
                "qubits as zero-padding for the comparison.",
                n_monomial,
                n_y,
                padding_needed,
            )

        multiplication_circuit = RGQFTMultiplier(
            num_state_qubits=len(grid_x_qubits),
            num_result_qubits=n_monomial,
        )

        # Copy x into the copy register
        for qc, qt in zip(grid_x_qubits, copy_qubits):
            circuit.cx(qc, qt)

        # Multiply x * copy -> result
        circuit.compose(
            multiplication_circuit,
            qubits=grid_x_qubits + copy_qubits + result_qubits,
            inplace=True,
        )

        obstacle_qubits = self.lattice.ancillae_obstacle_index(
            self.target_obstacle_index
        )

        if needs_padding:
            # Free the copy register by undoing the copy operation.
            # This leaves all copy qubits in |0>, so a subset can
            # serve as zero-padding for the shorter register in the
            # TwoRegisterComparator.  The comparator preserves both
            # input registers, so the padding qubits remain |0>
            # afterwards and the copy can be safely restored.
            for qc, qt in zip(grid_x_qubits, copy_qubits):
                circuit.cx(qc, qt)

            comparator_size = max(n_y, n_monomial)
            padding_qubits = copy_qubits[: abs(n_monomial - n_y)]

            if n_monomial > n_y:
                # Pad y with zeros in the high bits
                comparator_x_reg = grid_y_qubits + padding_qubits
                comparator_y_reg = result_qubits
            else:
                # Pad result with zeros in the high bits
                comparator_x_reg = grid_y_qubits
                comparator_y_reg = result_qubits + padding_qubits

            comparator_circuit = TwoRegisterComparator(
                comparator_size, ym.comparator_mode
            ).circuit

            circuit.compose(
                comparator_circuit,
                qubits=comparator_x_reg + comparator_y_reg + obstacle_qubits,
                inplace=True,
            )

            # Restore the copy register for the inverse multiplication
            for qc, qt in zip(grid_x_qubits, copy_qubits):
                circuit.cx(qc, qt)
        else:
            comparator_circuit = TwoRegisterComparator(n_y, ym.comparator_mode).circuit

            circuit.compose(
                comparator_circuit,
                qubits=grid_y_qubits + result_qubits + obstacle_qubits,
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


class ABZoneAgnosticSRCheck(LBMPrimitive):
    r"""Determines which spatial dimensions caused a diagonal particle to enter the obstacle.

    For each dimension :math:`d`, this primitive:

    1. Unstreams only the diagonal velocities in dimension :math:`d`.
    2. Applies an oracle targeting :math:`a_{d+1}` to check whether the particle
       is still inside the obstacle after the partial unstream.
    3. Flips :math:`a_{d+1}` so that :math:`a_{d+1} = 1` means
       "dimension :math:`d` caused the entry" (i.e., unstreaming in :math:`d`
       took the particle outside the obstacle).
    4. Restreams the diagonal velocities to restore the original position.

    The oracle for each dimension is constructed internally and targets
    ``ancillae_obstacle_index(dim + 1)`` directly, avoiding the need for
    swap-based ancilla management.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab.reflection.agnosotic_reflection import ABZoneAgnosticSRCheck
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
                "geometry": [
                    {
                        "shape": "cuboid",
                        "x": [1, 3],
                        "y": [1, 3],
                        "boundary": "specular",
                    }
                ],
            }
        )

        ABZoneAgnosticSRCheck(lattice, lattice.discretization, lattice.shapes["specular"]).draw("mpl")

    """

    sr_velocities_to_unstream: Dict[
        LatticeDiscretization, Dict[int, Tuple[bool | None, ...]]
    ] = {
        LatticeDiscretization.D2Q9: {
            5: (True, True),
            6: (False, True),
            7: (False, False),
            8: (True, False),
            1: (True, None),
            2: (None, True),
            3: (False, None),
            4: (None, False),
        }
    }

    def __init__(
        self,
        lattice: ABLattice,
        discretization: LatticeDiscretization,
        shapes: List[Shape],
        check_negative_direction: bool = False,
        additional_control_qubit_indices: List[int] = [],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.lattice = lattice
        self.discretization = discretization
        self.shapes = shapes
        self.additional_control_qubit_indices = additional_control_qubit_indices
        self.check_negative_direction = check_negative_direction

        if discretization not in self.sr_velocities_to_unstream:
            raise LatticeException(
                f"Specular Reflection BCs not supported for {discretization}"
            )

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def __build_oracle_for_dim(self, dim: int) -> QuantumCircuit:
        """Build an oracle circuit targeting ``ancillae_obstacle_index(dim + 1)``.

        Parameters
        ----------
        dim : int
            The spatial dimension (0 for x, 1 for y).

        Returns
        -------
        QuantumCircuit
            Oracle circuit that flips ``a_{dim+1}`` when position is inside obstacle.
        """
        oracle = self.lattice.circuit.copy()
        for shape in self.shapes:
            oracle.compose(
                ABZoneAgnosticReflectionOracle(
                    self.lattice,
                    shape,
                    additional_control_qubits=self.lattice.ancillae_obstacle_index(0),
                    target_obstacle_index=dim + 1,
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )
        return oracle

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        if not self.check_negative_direction:
            # If we are inside the obstacle, but neither x nor y contributed individually to getting her,
            # Then it must have been their combination.
            circuit.x(self.lattice.ancillae_obstacle_index()[1:-1])
            circuit.mcx(
                self.lattice.ancillae_obstacle_index()[:-1],
                self.lattice.ancillae_obstacle_index()[-1],
            )
            circuit.x(self.lattice.ancillae_obstacle_index()[1:-1])

        for dim in range(self.lattice.num_dims):
            # --- Unstream diagonal velocities in this dimension ---
            circuit.compose(
                QFT(len(self.lattice.grid_index(dim))),
                qubits=self.lattice.grid_index(dim),
                inplace=True,
            )
            for velocity_idx, vel_signs in self.sr_velocities_to_unstream[
                self.discretization
            ].items():
                # Unstream = reverse the streaming direction for this dim
                positive = vel_signs[dim]

                if positive is None:
                    continue

                if self.check_negative_direction:
                    positive = not positive
                    velocity_inversion_qubits = [
                        self.lattice.num_grid_qubits + q
                        for q in get_qubits_to_invert(
                            velocity_idx, self.lattice.num_velocity_qubits
                        )
                    ]
                if velocity_inversion_qubits:
                    circuit.x(velocity_inversion_qubits)

                circuit.compose(
                    PhaseShift(
                        num_qubits=len(self.lattice.grid_index(dim)),
                        positive=positive,
                        logger=self.logger,
                    )
                    .circuit.control(
                        self.lattice.num_velocity_qubits
                        + len(self.additional_control_qubit_indices)
                    )
                    .decompose(),
                    qubits=self.additional_control_qubit_indices
                    + self.lattice.velocity_index()
                    + self.lattice.grid_index(dim),
                    inplace=True,
                )

                if velocity_inversion_qubits:
                    circuit.x(velocity_inversion_qubits)

            circuit.compose(
                QFT(len(self.lattice.grid_index(dim)), inverse=True),
                qubits=self.lattice.grid_index(dim),
                inplace=True,
            )

            # --- Oracle targeting a_{dim+1} ---
            # After unstreaming dimension dim, the oracle checks whether
            # the particle is still inside. If inside, a_{dim+1} is set to 1.
            # The subsequent X gate inverts the meaning:
            # a_{dim+1} = 1 means unstreaming took the particle OUT,
            # i.e., dimension dim caused the entry.
            circuit.compose(self.__build_oracle_for_dim(dim), inplace=True)
            circuit.x(self.lattice.ancillae_obstacle_index(dim + 1))

            # --- Restream diagonal velocities in this dimension ---
            circuit.compose(
                QFT(len(self.lattice.grid_index(dim))),
                qubits=self.lattice.grid_index(dim),
                inplace=True,
            )
            for velocity_idx, vel_signs in self.sr_velocities_to_unstream[
                self.discretization
            ].items():
                # Restream = original streaming direction for this dim
                if vel_signs[dim] is None:
                    continue
                positive = not vel_signs[dim]
                if self.check_negative_direction:
                    positive = not positive
                velocity_inversion_qubits = [
                    self.lattice.num_grid_qubits + q
                    for q in get_qubits_to_invert(
                        velocity_idx, self.lattice.num_velocity_qubits
                    )
                ]
                if velocity_inversion_qubits:
                    circuit.x(velocity_inversion_qubits)

                circuit.compose(
                    PhaseShift(
                        num_qubits=len(self.lattice.grid_index(dim)),
                        positive=positive,
                        logger=self.logger,
                    )
                    .circuit.control(
                        self.lattice.num_velocity_qubits
                        + len(self.additional_control_qubit_indices)
                    )
                    .decompose(),
                    qubits=self.additional_control_qubit_indices
                    + self.lattice.velocity_index()
                    + self.lattice.grid_index(dim),
                    inplace=True,
                )

                if velocity_inversion_qubits:
                    circuit.x(velocity_inversion_qubits)

            circuit.compose(
                QFT(len(self.lattice.grid_index(dim)), inverse=True),
                qubits=self.lattice.grid_index(dim),
                inplace=True,
            )

        if self.check_negative_direction:
            # If we are inside the obstacle, but neither x nor y contributed individually to getting her,
            # Then it must have been their combination.
            circuit.x(self.lattice.ancillae_obstacle_index()[1:-1])
            circuit.mcx(
                self.lattice.ancillae_obstacle_index()[:-1],
                self.lattice.ancillae_obstacle_index()[-1],
            )
            circuit.x(self.lattice.ancillae_obstacle_index()[1:-1])
        return circuit

    @override
    def __str__(self):
        return f"[Primitive ABZoneAgnosticSRCheck with lattice {self.lattice}]"
