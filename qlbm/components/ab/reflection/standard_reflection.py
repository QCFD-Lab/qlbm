"""Reflection utilities for the :class:`.ABQLBM` algorithm; generalizations of :cite:`collisionless`."""

from itertools import product
from logging import Logger, getLogger
from time import perf_counter_ns
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, XGate
from qiskit.synthesis import synth_qft_full as QFT
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.ab.reflection.common import (
    ABBounceBackReflectionPermutation,
    ABSpecularReflectionPermutation,
)
from qlbm.components.ab.streaming import ABStreamingOperator
from qlbm.components.base import LBMOperator
from qlbm.components.common.adders import PhaseShift
from qlbm.components.ms.specular_reflection import SpecularWallComparator
from qlbm.lattice.geometry.encodings.ms import ReflectionPoint
from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.lattices.base import AmplitudeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import flatten, get_qubits_to_invert


def set_ancilla_of_point_state(
    lattice: AmplitudeLattice,
    points_data: List[Tuple[ReflectionPoint, List[int]]],
    ignore_velocity_data: bool,
    control_on_marker_state: bool = False,
    target_obstacle_index: int = 0,
) -> QuantumCircuit:
    """
    Toggle the obstacle ancilla qubit of a gridpoint, conditioned on velocity.

    This is a shared utility used by both
    :class:`ABBounceBackReflectionOperator` and
    :class:`ABSpecularReflectionOperator`.

    Parameters
    ----------
    lattice : AmplitudeLattice
        The lattice providing register layout information.
    points_data : List[Tuple[ReflectionPoint, List[int]]]
        Pairs of (gridpoint, velocity indices) to toggle the ancilla for.
    ignore_velocity_data : bool
        If ``True``, toggle the ancilla based on position alone.
    control_on_marker_state : bool
        Whether to additionally control on the marker register.
    target_obstacle_index : int
        Which obstacle ancilla qubit to target (default 0).

    Returns
    -------
    QuantumCircuit
        The circuit toggling the obstacle ancilla.
    """
    circuit = lattice.circuit.copy()

    for point, velocities in points_data:
        grid_qubit_indices_to_invert = [
            lattice.grid_index(0)[0] + qubit for qubit in point.qubits_to_invert
        ]
        if grid_qubit_indices_to_invert:
            circuit.x(grid_qubit_indices_to_invert)

        match lattice.get_encoding():
            case ABEncodingType.AB:
                velocity_data = (
                    [
                        [
                            lattice.velocity_index()[0] + qubit
                            for qubit in get_qubits_to_invert(
                                velocity_index,
                                lattice.num_velocity_qubits,
                            )
                        ]
                        for velocity_index in velocities
                    ]
                    if not ignore_velocity_data
                    else [[]]
                )

                for velocity_qubit_indices_to_invert in velocity_data:
                    if velocity_qubit_indices_to_invert:
                        circuit.x(velocity_qubit_indices_to_invert)

                    control_qubits = lattice.grid_index() + (
                        lattice.velocity_index() if not ignore_velocity_data else []
                    )

                    if control_on_marker_state:
                        control_qubits.extend(lattice.marker_index())

                    target_qubits = lattice.ancillae_obstacle_index(
                        target_obstacle_index
                    )

                    circuit.compose(
                        MCMTGate(
                            XGate(),
                            len(control_qubits),
                            len(target_qubits),
                        ),
                        qubits=control_qubits + target_qubits,
                        inplace=True,
                    )
                    if velocity_qubit_indices_to_invert:
                        circuit.x(velocity_qubit_indices_to_invert)
            case ABEncodingType.OH:
                if ignore_velocity_data:
                    control_qubits = lattice.grid_index() + (
                        lattice.marker_index() if control_on_marker_state else []
                    )
                    circuit.compose(
                        MCMTGate(
                            XGate(),
                            len(control_qubits),
                            len(lattice.ancillae_obstacle_index(target_obstacle_index)),
                        ),
                        qubits=control_qubits
                        + lattice.ancillae_obstacle_index(target_obstacle_index),
                        inplace=True,
                    )
                else:
                    for v in velocities:
                        control_qubits = lattice.grid_index() + (
                            [lattice.velocity_index()[v]]
                        )

                        if control_on_marker_state:
                            control_qubits.extend(lattice.marker_index())

                        target_qubits = lattice.ancillae_obstacle_index(
                            target_obstacle_index
                        )

                        circuit.compose(
                            MCMTGate(
                                XGate(),
                                len(control_qubits),
                                len(target_qubits),
                            ),
                            qubits=control_qubits + target_qubits,
                            inplace=True,
                        )
            case _:
                raise LatticeException(
                    f"Unsupported lattice encoding: {lattice.get_encoding()}"
                )
        if grid_qubit_indices_to_invert:
            circuit.x(grid_qubit_indices_to_invert)

    return circuit


class ABReflectionOperator(LBMOperator):
    r"""
    Implements reflection boundary conditions in the amplitude-based encoding of :class:`.ABQLBM` for :math:`D_dQ_q` discretizations.

    This is the top-level entrypoint that delegates to :class:`ABBounceBackReflectionOperator` and :class:`ABSpecularReflectionOperator` based on the boundary condition types present in the geometry.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABReflectionOperator
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

        ABReflectionOperator(lattice).draw("mpl")

    """

    lattice: AmplitudeLattice

    control_on_marker_state: bool
    """Whether reflection is restricted to the marker-one sector of the state.

    Algorithms that reserve the marker qubit to separate physical from auxiliary
    amplitudes, such as :class:`.ABBGKQLBM`, must apply boundary conditions to the
    physical sector only."""

    def __init__(
        self,
        lattice: ABLattice,
        shapes: Dict[str, List[Shape]] | None = None,
        control_on_marker_state: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)

        self.control_on_marker_state = control_on_marker_state

        if shapes is not None:
            if not self.lattice.has_multiple_geometries():
                self.shapes: Dict[str, List[Shape]] | List[Dict[str, List[Shape]]] = (
                    shapes
                )
            else:
                self.shapes = [shapes]
        elif not self.lattice.has_multiple_geometries():
            self.shapes = self.lattice.geometries[0]
        else:
            self.shapes = list(self.lattice.geometries)

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took "
            f"{perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        if self.lattice.discretization not in [LatticeDiscretization.D2Q9]:
            raise LatticeException("AB reflection only currently supported in D2Q9")

        if not self.lattice.has_multiple_geometries():
            shapes_dict: Dict[str, List[Shape]] = self.shapes  # type: ignore[assignment]
            return self.__create_circuit_d2q9(
                shapes_dict,
                control_on_marker_state=self.control_on_marker_state,
            )
        else:
            circuit = self.lattice.circuit.copy()
            geometry_list: List[Dict[str, List[Shape]]] = self.shapes  # type: ignore[assignment]
            for c, shapes_for_geometry in enumerate(geometry_list):
                qubits_to_invert = [
                    q + self.lattice.marker_index()[0]
                    for q in get_qubits_to_invert(c, self.lattice.num_marker_qubits)
                ]

                if qubits_to_invert:
                    circuit.x(qubits_to_invert)

                circuit.compose(
                    self.__create_circuit_d2q9(
                        shapes_for_geometry,
                        control_on_marker_state=True,
                    ),
                    inplace=True,
                )

                if qubits_to_invert:
                    circuit.x(qubits_to_invert)
            return circuit

    def __create_circuit_d2q9(
        self,
        shapes_dict: Dict[str, List[Shape]],
        control_on_marker_state: bool = False,
    ) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        bb_blocks = shapes_dict.get("bounceback", [])
        sr_blocks = shapes_dict.get("specular", [])

        if bb_blocks:
            circuit.compose(
                ABBounceBackReflectionOperator(
                    self.lattice,  # type: ignore[arg-type]
                    bb_blocks,
                    control_on_marker_state=control_on_marker_state,
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )

        if sr_blocks:
            circuit.compose(
                ABSpecularReflectionOperator(
                    self.lattice,  # type: ignore[arg-type]
                    sr_blocks,
                    control_on_marker_state=control_on_marker_state,
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Operator ABReflection with lattice {self.lattice}]"


class ABBounceBackReflectionOperator(LBMOperator):
    r"""
    Implements bounce-back reflection in the amplitude-based encoding of :class:`.ABQLBM` for :math:`D_2Q_9`.

    Bounce-back reflection reverses all velocity components of particles
    that have entered the obstacle walls. The algorithm proceeds as:

    1. **Mark inner walls** -- Set the obstacle ancilla for gridpoints
       inside each wall segment.
    2. **Mark inner corners** -- Set the obstacle ancilla for the inner
       corner gridpoints.
    3. **Permute and stream** -- Apply the bounce-back velocity
       permutation (full reversal) and stream, both controlled on the
       obstacle ancilla.
    4. **Reset outer walls** -- Reset the obstacle ancilla using the
       outside wall comparators.
    5. **Corner corrections** -- Fix near-corner and outside-corner
       ancilla residuals.
    """

    lattice: AmplitudeLattice

    def __init__(
        self,
        lattice: ABLattice,
        blocks: List[Shape],
        control_on_marker_state: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.blocks = blocks
        self.control_on_marker_state = control_on_marker_state

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took "
            f"{perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        if self.lattice.discretization != LatticeDiscretization.D2Q9:
            raise LatticeException("AB bounce-back reflection only supported in D2Q9")

        circuit = self.lattice.circuit.copy()

        for block in self.blocks:
            circuit.compose(
                self.set_inside_wall_ancilla_state(block),  # type: ignore[arg-type]
                inplace=True,
            )

        circuit.compose(
            set_ancilla_of_point_state(
                self.lattice,
                flatten(
                    [[(p, None) for p in block.corners_inside] for block in self.blocks]  # type: ignore[attr-defined]
                ),
                ignore_velocity_data=True,
                control_on_marker_state=self.control_on_marker_state,
            ),
            inplace=True,
        )

        circuit.compose(self.permute_and_stream(), inplace=True)

        for block in self.blocks:
            circuit.compose(
                self.reset_outside_wall_ancilla_state(block),  # type: ignore[arg-type]
                inplace=True,
            )

        point_data: List[Tuple[ReflectionPoint, List[int]]] = []

        for block in self.blocks:
            for dim in range(self.lattice.num_dims):
                for c, bounds in enumerate(
                    product(*[[False, True]] * self.lattice.num_dims)
                ):
                    point_data.append(
                        (
                            block.near_corner_points_2d[dim * 4 + c],  # type: ignore[attr-defined]
                            block.get_lbm_near_corner_velocity_indices_to_reflect(  # type: ignore[attr-defined]
                                self.lattice.discretization, dim, bounds
                            ),
                        )
                    )
            for c, bounds in enumerate(
                product(*[[False, True]] * self.lattice.num_dims)
            ):
                point_data.append(
                    (
                        block.corners_outside[c],  # type: ignore[attr-defined]
                        block.get_lbm_outside_corner_indices_to_reflect(  # type: ignore[attr-defined]
                            self.lattice.discretization, bounds
                        ),
                    )
                )

        circuit.compose(
            set_ancilla_of_point_state(
                self.lattice,
                point_data,
                ignore_velocity_data=False,
                control_on_marker_state=self.control_on_marker_state,
            ),
            inplace=True,
        )

        return circuit

    def set_inside_wall_ancilla_state(self, block: Block) -> QuantumCircuit:
        """
        Set the obstacle ancilla for gridpoints lying inside the walls of a block.

        Uses the :class:`.SpecularWallComparator` to identify wall positions.
        Inside corner points are not addressed by this primitive.

        Parameters
        ----------
        block : Block
            The solid object to address.

        Returns
        -------
        QuantumCircuit
            The circuit that sets the obstacle ancilla qubit.
        """
        circuit = self.lattice.circuit.copy()

        for dim in range(self.lattice.num_dims):
            for wall in block.walls_inside[dim]:
                comparator_circuit = SpecularWallComparator(
                    self.lattice, wall, self.logger
                ).circuit

                grid_qubit_indices_to_invert = [
                    self.lattice.grid_index(0)[0] + qubit
                    for qubit in wall.data.qubits_to_invert
                ]

                circuit.compose(comparator_circuit, inplace=True)

                if grid_qubit_indices_to_invert:
                    circuit.x(grid_qubit_indices_to_invert)

                control_qubits = (
                    self.lattice.grid_index(wall.dim)
                    + self.lattice.ancillae_comparator_index()
                )

                if self.control_on_marker_state:
                    control_qubits.extend(self.lattice.marker_index())

                target_qubits = self.lattice.ancillae_obstacle_index(0)

                circuit.compose(
                    MCMTGate(
                        XGate(),
                        len(control_qubits),
                        len(target_qubits),
                    ),
                    qubits=control_qubits + target_qubits,
                    inplace=True,
                )

                if grid_qubit_indices_to_invert:
                    circuit.x(grid_qubit_indices_to_invert)

                circuit.compose(comparator_circuit, inplace=True)

        return circuit

    def reset_outside_wall_ancilla_state(self, block: Block) -> QuantumCircuit:
        """
        Reset the obstacle ancilla for gridpoints adjacent to the object in the fluid domain.

        Uses the :class:`.SpecularWallComparator` on the outside walls.
        Near-corner and outside-corner gridpoints will be incorrect after
        this step and require separate correction.

        Parameters
        ----------
        block : Block
            The solid object to address.

        Returns
        -------
        QuantumCircuit
            The circuit that resets the obstacle ancilla qubit.
        """
        circuit = self.lattice.circuit.copy()

        for dim in range(self.lattice.num_dims):
            for bound, wall in enumerate(block.walls_outside[dim]):
                comparator_circuit = SpecularWallComparator(
                    self.lattice, wall, self.logger
                ).circuit

                grid_qubit_indices_to_invert = [
                    self.lattice.grid_index(0)[0] + qubit
                    for qubit in wall.data.qubits_to_invert
                ]

                circuit.compose(comparator_circuit, inplace=True)

                if grid_qubit_indices_to_invert:
                    circuit.x(grid_qubit_indices_to_invert)

                for v in block.get_lbm_wall_velocity_indices_to_reflect(
                    self.lattice.discretization, dim, bool(bound)
                ):
                    match self.lattice.get_encoding():
                        case ABEncodingType.AB:
                            qs = [
                                self.lattice.velocity_index()[0] + q
                                for q in get_qubits_to_invert(
                                    v, self.lattice.num_velocity_qubits
                                )
                            ]

                            if qs:
                                circuit.x(qs)

                            control_qubits = (
                                self.lattice.grid_index(wall.dim)
                                + self.lattice.ancillae_comparator_index()
                                + self.lattice.velocity_index()
                            )

                            if self.control_on_marker_state:
                                control_qubits.extend(self.lattice.marker_index())

                            target_qubits = self.lattice.ancillae_obstacle_index(0)

                            circuit.compose(
                                MCMTGate(
                                    XGate(),
                                    len(control_qubits),
                                    len(target_qubits),
                                ),
                                qubits=control_qubits + target_qubits,
                                inplace=True,
                            )

                            if qs:
                                circuit.x(qs)
                        case ABEncodingType.OH:
                            control_qubits = (
                                self.lattice.grid_index(wall.dim)
                                + self.lattice.ancillae_comparator_index()
                                + [self.lattice.velocity_index()[v]]
                            )

                            if self.control_on_marker_state:
                                control_qubits.extend(self.lattice.marker_index())

                            target_qubits = self.lattice.ancillae_obstacle_index(0)

                            circuit.compose(
                                MCMTGate(
                                    XGate(),
                                    len(control_qubits),
                                    len(target_qubits),
                                ),
                                qubits=control_qubits + target_qubits,
                                inplace=True,
                            )

                        case _:
                            raise LatticeException(
                                f"Unsupported lattice encoding: {self.lattice.get_encoding()}"
                            )

                if grid_qubit_indices_to_invert:
                    circuit.x(grid_qubit_indices_to_invert)

                circuit.compose(comparator_circuit, inplace=True)
        return circuit

    def permute_and_stream(self) -> QuantumCircuit:
        """
        Perform the bounce-back velocity permutation followed by streaming.

        The permutation reverses all velocity components.

        Returns
        -------
        QuantumCircuit
            The combined permutation and streaming circuit.
        """
        circuit = self.lattice.circuit.copy()

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
        return f"[Operator ABBounceBackReflection with lattice {self.lattice}]"


class ABSpecularReflectionOperator(LBMOperator):
    r"""
    Implements specular reflection in the amplitude-based encoding of :class:`.ABQLBM` for :math:`D_2Q_9`.

    This operator uses per-dimension obstacle ancillae (``a_x``, ``a_y``)
    to track which wall a particle has entered through. The algorithm
    proceeds in five phases:

    1. **Mark inner walls** -- For each dimension, set the per-dimension
       obstacle ancilla for all gridpoints lying inside the walls of
       each block.
    2. **Specular permutations** -- Apply per-dimension velocity
       permutations controlled on the corresponding obstacle ancilla.
    3. **Dimension-selective stream** -- For each dimension *d*, stream
       the grid qubits of *d* controlled on ``ancilla[d]``. This
       ensures a particle reflected off an x-wall only moves in x,
       while a corner particle (both ancillae set) moves in both.
    4. **Reset outer walls** -- For each dimension, reset the
       per-dimension obstacle ancilla using the outside wall
       comparators.
    5. **Corner corrections** -- Fix near-corner and outside-corner
       ancilla residuals caused by the wall-based reset overshoot and
       by cardinal velocities at inner corners.

    Parameters
    ----------
    lattice : ABLattice
        The lattice on which to build the reflection circuit.
    blocks : List[Shape]
        The list of specular :class:`.Block` objects.
    control_on_marker_state : bool
        Whether to control all MCX gates on the marker register.
    logger : Logger
        The performance logger.
    """

    lattice: AmplitudeLattice

    def __init__(
        self,
        lattice: ABLattice,
        blocks: List[Shape],
        control_on_marker_state: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.blocks = blocks
        self.control_on_marker_state = control_on_marker_state

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took "
            f"{perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        if self.lattice.discretization != LatticeDiscretization.D2Q9:
            raise LatticeException("AB specular reflection only supported in D2Q9")

        circuit = self.lattice.circuit.copy()

        # Phase 1: Mark inner walls per dimension
        for dim in range(self.lattice.num_dims):
            for block in self.blocks:
                circuit.compose(
                    self._set_inside_wall_ancilla_per_dim(block, dim),  # type: ignore[arg-type]
                    inplace=True,
                )

        # Phase 1b: Correct inner corner ancillae
        circuit.compose(self._correct_inner_corner_ancillae(), inplace=True)

        # Phase 2: Per-dimension specular permutations
        circuit.compose(self._specular_permutations(), inplace=True)

        # Phase 3: Dimension-selective streaming
        circuit.compose(self._dim_selective_stream(), inplace=True)

        # Phase 4: Reset outer walls per dimension
        for dim in range(self.lattice.num_dims):
            for block in self.blocks:
                circuit.compose(
                    self._reset_outside_wall_ancilla_per_dim(block, dim),  # type: ignore[arg-type]
                    inplace=True,
                )

        # Phase 5: Corner corrections
        circuit.compose(self._corner_corrections(), inplace=True)

        return circuit

    def _set_inside_wall_ancilla_per_dim(
        self, block: Block, dim: int
    ) -> QuantumCircuit:
        """Set ``ancilla[dim]`` for gridpoints inside the walls of *dim*.

        Parameters
        ----------
        block : Block
            The obstacle.
        dim : int
            The spatial dimension whose walls to process.

        Returns
        -------
        QuantumCircuit
            Sub-circuit that marks the inner walls of *dim*.
        """
        circuit = self.lattice.circuit.copy()

        for wall in block.walls_inside[dim]:
            comparator_circuit = SpecularWallComparator(
                self.lattice, wall, self.logger
            ).circuit

            grid_qubit_indices_to_invert = [
                self.lattice.grid_index(0)[0] + qubit
                for qubit in wall.data.qubits_to_invert
            ]

            circuit.compose(comparator_circuit, inplace=True)

            if grid_qubit_indices_to_invert:
                circuit.x(grid_qubit_indices_to_invert)

            control_qubits = (
                self.lattice.grid_index(wall.dim)
                + self.lattice.ancillae_comparator_index()
            )

            if self.control_on_marker_state:
                control_qubits.extend(self.lattice.marker_index())

            target_qubits = self.lattice.ancillae_obstacle_index(dim)

            circuit.compose(
                MCMTGate(
                    XGate(),
                    len(control_qubits),
                    len(target_qubits),
                ),
                qubits=control_qubits + target_qubits,
                inplace=True,
            )

            if grid_qubit_indices_to_invert:
                circuit.x(grid_qubit_indices_to_invert)

            circuit.compose(comparator_circuit, inplace=True)

        return circuit

    def _correct_inner_corner_ancillae(self) -> QuantumCircuit:
        """Unset per-dimension ancillae at inner corners for non-entering velocities.

        Phase 1 marks inner walls using position-only comparators.  At
        inner corner gridpoints (where two walls overlap) both ``a_x``
        and ``a_y`` are set for *all* velocities.  However, ``a_d``
        should only be set when the velocity has a component that enters
        through wall *d* from outside.

        For example, at the lower-left corner ``(x_lo, y_lo)`` a
        particle with ``v = 1 (+x, 0)`` entered only through the
        x-wall, so ``a_y`` must be unset.  A particle with ``v = 5
        (+x, +y)`` entered through both walls, so both ancillae stay.

        This method toggles ``a_d`` for every velocity that does **not**
        have the entering component for wall *d*, undoing the erroneous
        Phase 1 marking at each inner corner.

        Returns
        -------
        QuantumCircuit
            Sub-circuit correcting the inner-corner ancillae.
        """
        circuit = self.lattice.circuit.copy()

        for block in self.blocks:
            for c, bounds in enumerate(
                product(*[[False, True]] * self.lattice.num_dims)
            ):
                corner_point = block.corners_inside[c]  # type: ignore[attr-defined]

                for dim in range(self.lattice.num_dims):
                    velocities_to_unset = block.get_lbm_sr_inner_corner_non_entering_velocity_indices(  # type: ignore[attr-defined]
                        self.lattice.discretization,
                        dim,
                        bounds[dim],
                    )

                    circuit.compose(
                        set_ancilla_of_point_state(
                            self.lattice,
                            [(corner_point, velocities_to_unset)],
                            ignore_velocity_data=False,
                            control_on_marker_state=self.control_on_marker_state,
                            target_obstacle_index=dim,
                        ),
                        inplace=True,
                    )

        return circuit

    def _specular_permutations(self) -> QuantumCircuit:
        """Apply per-dimension specular velocity permutations.

        For each dimension *d*, the :class:`.ABSpecularReflectionPermutation`
        is applied controlled on ``ancilla[d]``.

        Returns
        -------
        QuantumCircuit
            Sub-circuit with both controlled permutations.
        """
        circuit = self.lattice.circuit.copy()

        for dim in range(self.lattice.num_dims):
            reflect_in_dim = tuple(d == dim for d in range(self.lattice.num_dims))

            perm = ABSpecularReflectionPermutation(
                self.lattice.num_velocity_qubits,
                self.lattice.discretization,
                self.lattice.get_encoding(),
                reflect_in_dim,
                self.logger,
            )

            circuit.compose(
                perm.circuit.control(1).decompose(),
                qubits=self.lattice.ancillae_obstacle_index(dim)
                + self.lattice.velocity_index(),
                inplace=True,
            )

        return circuit

    def _dim_selective_stream(self) -> QuantumCircuit:
        r"""Stream reflected particles back, one spatial dimension at a time.

        For each dimension *d* the QFT-based Draper-adder streaming is
        applied to the grid qubits of that dimension, with every
        controlled phase-shift gate additionally controlled on
        ``ancilla[d]``.  This ensures that a particle reflected off an
        x-wall (``a_x = 1, a_y = 0``) is only streamed in x, and
        similarly for y-walls.  At corners where both ancillae are set
        the particle is streamed in both dimensions, which is the
        correct specular corner behaviour.

        The per-dimension loop mirrors the structure of
        :class:`.ABStreamingOperator` but substitutes the shared
        ``additional_control_qubit_indices`` with a dimension-specific
        obstacle ancilla.

        Returns
        -------
        QuantumCircuit
            Sub-circuit performing dimension-selective streaming.
        """
        circuit = self.lattice.circuit.copy()

        # Velocity indices that stream in each direction per dimension
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

        for dim, dim_population_to_update in enumerate(dim_indices):
            control_qubit = self.lattice.ancillae_obstacle_index(dim)

            circuit.compose(
                QFT(len(self.lattice.grid_index(dim))),
                qubits=self.lattice.grid_index(dim),
                inplace=True,
            )

            for direction, indices in enumerate(dim_population_to_update):
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
                            self.lattice.num_velocity_qubits + len(control_qubit)
                        )
                        .decompose(),
                        qubits=control_qubit
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

        return circuit

    def _reset_outside_wall_ancilla_per_dim(
        self, block: Block, dim: int
    ) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        for bound, wall in enumerate(block.walls_outside[dim]):
            comparator_circuit = SpecularWallComparator(
                self.lattice, wall, self.logger
            ).circuit

            grid_qubit_indices_to_invert = [
                self.lattice.grid_index(0)[0] + qubit
                for qubit in wall.data.qubits_to_invert
            ]

            circuit.compose(comparator_circuit, inplace=True)

            if grid_qubit_indices_to_invert:
                circuit.x(grid_qubit_indices_to_invert)

            for v in block.get_lbm_wall_velocity_indices_to_reflect(
                self.lattice.discretization, dim, bool(bound)
            ):
                qs = [
                    self.lattice.velocity_index()[0] + q
                    for q in get_qubits_to_invert(v, self.lattice.num_velocity_qubits)
                ]

                if qs:
                    circuit.x(qs)

                control_qubits = (
                    self.lattice.grid_index(wall.dim)
                    + self.lattice.ancillae_comparator_index()
                    + self.lattice.velocity_index()
                )

                if self.control_on_marker_state:
                    control_qubits.extend(self.lattice.marker_index())

                target_qubits = self.lattice.ancillae_obstacle_index(dim)

                circuit.compose(
                    MCMTGate(
                        XGate(),
                        len(control_qubits),
                        len(target_qubits),
                    ),
                    qubits=control_qubits + target_qubits,
                    inplace=True,
                )

                if qs:
                    circuit.x(qs)

            if grid_qubit_indices_to_invert:
                circuit.x(grid_qubit_indices_to_invert)

            circuit.compose(comparator_circuit, inplace=True)

        return circuit

    def _corner_corrections(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        for block in self.blocks:
            # 5a: same-dimension near-corner corrections
            for dim in range(self.lattice.num_dims):
                same_dim_data: List[Tuple[ReflectionPoint, List[int]]] = []
                for c, bounds in enumerate(
                    product(*[[False, True]] * self.lattice.num_dims)
                ):
                    same_dim_data.append(
                        (
                            block.near_corner_points_2d[dim * 4 + c],  # type: ignore[attr-defined]
                            block.get_lbm_near_corner_velocity_indices_to_reflect(  # type: ignore[attr-defined]
                                self.lattice.discretization, dim, bounds
                            ),
                        )
                    )
                circuit.compose(
                    set_ancilla_of_point_state(
                        self.lattice,
                        same_dim_data,
                        ignore_velocity_data=False,
                        control_on_marker_state=self.control_on_marker_state,
                        target_obstacle_index=dim,
                    ),
                    inplace=True,
                )

            # 5b: outside corner corrections (both dimensions)
            for target_dim in range(self.lattice.num_dims):
                corner_data: List[Tuple[ReflectionPoint, List[int]]] = []
                for c, bounds in enumerate(
                    product(*[[False, True]] * self.lattice.num_dims)
                ):
                    corner_data.append(
                        (
                            block.corners_outside[c],  # type: ignore[attr-defined]
                            block.get_lbm_outside_corner_indices_to_reflect(  # type: ignore[attr-defined]
                                self.lattice.discretization, bounds
                            ),
                        )
                    )
                circuit.compose(
                    set_ancilla_of_point_state(
                        self.lattice,
                        corner_data,
                        ignore_velocity_data=False,
                        control_on_marker_state=self.control_on_marker_state,
                        target_obstacle_index=target_dim,
                    ),
                    inplace=True,
                )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Operator ABSpecularReflection with lattice {self.lattice}]"
