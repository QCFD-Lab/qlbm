"""Quantum comparator circuits for the implementation of bounce-back boundary conditions as described in :cite:t:`qmem`."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, XGate
from typing_extensions import override

from qlbm.components.base import CQLBMOperator, LBMPrimitive
from qlbm.components.collisionless.primitives import (
    Comparator,
    ComparatorMode,
)
from qlbm.components.collisionless.specular_reflection import SpecularWallComparator
from qlbm.lattice import CollisionlessLattice
from qlbm.lattice.geometry.encodings.collisionless import (
    ReflectionPoint,
    ReflectionResetEdge,
    ReflectionWall,
)
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.tools.exceptions import CircuitException
from qlbm.tools.utils import flatten

from .primitives import EdgeComparator
from .streaming import ControlledIncrementer


class BounceBackWallComparator(LBMPrimitive):
    r"""
    A primitive used in the collision :class:`BounceBackReflectionOperator` that implements the comparator for the BB boundary conditions as described in :cite:t:`qmem`.

    The comparator sets an ancilla qubit to :math:`\ket{1}` for the components of
    the quantum state whose grid qubits fall within the range spanned by the wall.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`wall`              The :class:`.ReflectionWall` encoding the range spanned by the wall.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.collisionless import BounceBackWallComparator
        from qlbm.lattice import CollisionlessLattice

        # Build an example lattice
        lattice = CollisionlessLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
                "geometry": [{"shape":"cuboid", "x": [5, 6], "y": [1, 2], "boundary": "bounceback"}],
            }
        )

        # Comparing on the indices of the inside x-wall on the lower-bound of the obstacle
        BounceBackWallComparator(
            lattice=lattice, wall=lattice.block_list[0].walls_inside[0][0]
        ).draw("mpl")
    """

    def __init__(
        self,
        lattice: CollisionlessLattice,
        wall: ReflectionWall,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice
        self.wall = wall

        self.circuit = self.create_circuit()

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        # If the wall is inside the object, we build the comparators
        # Differently, as to not overlap
        lb_comparators = [
            Comparator(
                self.lattice.num_gridpoints[wall_alignment_dim].bit_length() + 1,
                self.wall.lower_bounds[c],
                ComparatorMode.GE
                if self.wall.bounceback_loose_bounds[self.wall.dim][c]
                else ComparatorMode.GT,
                logger=self.logger,
            ).circuit
            for c, wall_alignment_dim in enumerate(self.wall.alignment_dims)
        ]

        ub_comparators = [
            Comparator(
                self.lattice.num_gridpoints[wall_alignment_dim].bit_length() + 1,
                self.wall.upper_bounds[c],
                ComparatorMode.LE
                if self.wall.bounceback_loose_bounds[self.wall.dim][c]
                else ComparatorMode.LT,
                logger=self.logger,
            ).circuit
            for c, wall_alignment_dim in enumerate(self.wall.alignment_dims)
        ]

        for c, wall_alignment_dim in enumerate(self.wall.alignment_dims):
            circuit.compose(
                lb_comparators[c],
                qubits=self.lattice.grid_index(wall_alignment_dim)
                + self.lattice.ancillae_comparator_index(c)[
                    :-1  # :-1 Effectively selects only the first (lb) qubit
                ],  # There are two comparator ancillae, for each relevant dimension, one for l and one for u
                inplace=True,
            )

            circuit.compose(
                ub_comparators[c],
                qubits=self.lattice.grid_index(wall_alignment_dim)
                + self.lattice.ancillae_comparator_index(c)[
                    1:  # 1: Effectively selects only the last (ub) qubit
                ],  # There are two comparator ancillae, for each relevant dimension, one for l and one for u.
                inplace=True,
            )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive BounceBackWallComparator on wall={self.wall}]"


class BounceBackReflectionOperator(CQLBMOperator):
    """
    Operator implementing the 2D and 3D Bounce-Back (BB) boundary conditions as described in :cite:t:`qmem`.

    The operator parses information encoded in :class:`.Block` objects to detect particles that
    have virtually streamed into the solid domain before placing them back to their
    previous positions in the fluid domain.
    The pseudocode for this procedure is as follows:

    #. Components of the quantum state that encode particles that have streamed inside the obstacle are identified with :class:`.BounceBackWallComparator` objects;
    #. These components have their velocity direction qubits flipped in all three dimensions;
    #. Particles are streamed outside the solid domain with inverted velocity directions;
    #. Once streamed outside the solid domain, components encoding affected particles have their obstacle ancilla qubit reset based on grid position, velocity direction, and whether they have streamed in the CFL timestep.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`blocks`            A list of  :class:`.Block` objects for which to generate the BB boundary condition circuits.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.collisionless import BounceBackReflectionOperator
        from qlbm.lattice import CollisionlessLattice

        # Build an example lattice
        lattice = CollisionlessLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
                "geometry": [{"shape":"cuboid", "x": [5, 6], "y": [1, 2], "boundary": "bounceback"}],
            }
        )

        BounceBackReflectionOperator(lattice=lattice, blocks=lattice.block_list)
    """

    def __init__(
        self,
        lattice: CollisionlessLattice,
        blocks: List[Block],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.blocks = blocks

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        # Reflect the particles that have streamed
        # Into the inner walls of the object
        for dim in range(self.lattice.num_dims):
            for block in self.blocks:
                for wall in block.walls_inside[dim]:
                    self.reflect_wall(circuit, wall)

        # Perform streaming
        circuit = self.flip_and_stream(circuit)

        # Reset the ancilla qubits for the particles that
        # Have been reflected onto the outer walls of the object
        for dim in range(self.lattice.num_dims):
            for block in self.blocks:
                for wall in block.walls_outside[dim]:
                    self.reflect_wall(circuit, wall)

        # If this is a 2D simulation
        if self.lattice.num_dims == 2:
            # Reset state for near-corner points
            for block in self.blocks:
                for near_corner_point in block.near_corner_points_2d:
                    self.reset_point_state(circuit, near_corner_point)

        # If this is a 3D simulation
        elif self.lattice.num_dims == 3:
            for block in self.blocks:
                # Reset the near-corner edges (24x)
                for near_corner_edge in block.near_corner_edges_3d:
                    self.reset_edge_state(circuit, near_corner_edge)

                # Reset the corner edges (12x)
                for corner_edge in block.corner_edges_3d:
                    self.reset_edge_state(circuit, corner_edge)

                for point in block.overlapping_near_corner_edge_points_3d:
                    self.reset_point_state(circuit, point)

        else:
            raise CircuitException(
                f"CQBM specular reflection is not supported for {self.lattice.num_dims} dimensions."
            )

        # Reset state for outside corners
        for block in self.blocks:
            # Reset the individual points at the corners of the block
            # (8x for 3D, 4x for 2D)
            for corner in block.corners_outside:
                self.reset_point_state(circuit, corner)

        return circuit

    def reflect_wall(
        self,
        circuit: QuantumCircuit,
        wall: ReflectionWall,
    ) -> QuantumCircuit:
        r"""Performs reflection based on information encoded in a :class:`.ReflectionWall` as follows.

        #. A series of :math:`X` gates set the grid qubits to the :math:`\ket{1}` state for the dimension that the wall spans.
        #. Comparator circuits set the comparator ancilla qubits to :math:`\ket{1}` based on the size of the wall in the other dimension(s).
        #. Depending on the use, the directional velocity qubits are also set to :math:`\ket{1}` based on the dimension that the wall spans.
        #. A multi-controlled :math:`X` gate flips the obstacle ancilla qubit, controlled on the qubits set in the previous steps.
        #. The control qubits are set back to their previous state.

        The wall reflection operation is versatile and can be used to both set and re-set the state
        of the obstacle ancilla qubit at different stages of reflection.
        When performing BB reflection, this function is first used to flip the
        ancilla obstacle qubit from :math:`\ket{0}` to :math:`\ket{1}`, irrespective of how particles arrived there.
        Subsequently, an additional controls are placed on the velocity direction qubits to reset the
        ancilla obstacle qubit to :math:`\ket{0}`, after particles have been streamed out of the solid domain.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit on which to perform BB reflection of the wall.
        wall : ReflectionWall
            The wall encoding the reflection logic.
        inside_obstacle : bool
            Whether the wall is inside the obstacle.

        Returns
        -------
        QuantumCircuit
            The circuit performing BB reflection of the wall.
        """
        comparator_circuit = (
            BounceBackWallComparator(self.lattice, wall, self.logger).circuit
            if not wall.data.is_outside_obstacle_bounds  # If the wall is outside the obstacle, the two comparators behave identically
            else SpecularWallComparator(self.lattice, wall, self.logger).circuit
        )

        grid_qubit_indices_to_invert = [
            self.lattice.grid_index(0)[0] + qubit
            for qubit in wall.data.qubits_to_invert
        ]

        circuit.compose(comparator_circuit, inplace=True)

        # If statement required because qiskit does not allow x([])
        # x([]) would only happen for the |11..1>
        # Grid position in the dimension that the wall reflects
        # i.e., the last row/column of the grid
        if grid_qubit_indices_to_invert:
            # Inverting the qubits that are 0 turns the
            # Dimensional grid qubit state encoding this wall to |11...1>
            # Which in turn allows us to control on this row, in combination with the comparator
            circuit.x(grid_qubit_indices_to_invert)

        if wall.data.invert_velocity:
            # Invert the `d`-velocity direction qubit
            circuit.x(self.lattice.velocity_dir_index(wall.dim))

        control_qubits = (
            self.lattice.ancillae_velocity_index(wall.dim)
            + (
                self.lattice.velocity_dir_index(wall.dim)
                if wall.data.is_outside_obstacle_bounds
                else []
            )
            + self.lattice.grid_index(wall.dim)
            + self.lattice.ancillae_comparator_index()
        )

        target_qubits = self.lattice.ancillae_obstacle_index(0)

        circuit.compose(
            MCMT(
                XGate(),
                len(control_qubits),
                len(target_qubits),
            ),
            qubits=control_qubits + target_qubits,
            inplace=True,
        )

        if wall.data.invert_velocity:
            circuit.x(self.lattice.velocity_dir_index(wall.dim))

        if grid_qubit_indices_to_invert:
            # Inverting the qubits that are 0 turns the
            # Dimensional grid qubit state encoding this wall to |11...1>
            # Which in turn allows us to control on this row, in combination with the comparator
            circuit.x(grid_qubit_indices_to_invert)

        circuit.compose(comparator_circuit, inplace=True)

        return circuit

    def reset_edge_state(
        self, circuit: QuantumCircuit, edge: ReflectionResetEdge
    ) -> QuantumCircuit:
        r"""Resets the state of an edge along the side of an obstacle in 3D as follows.

        #. A series of :math:`X` gates set the grid qubits to the :math:`\ket{1}` state for the 2 dimensions that the edge spans.
        #. A comparator circuits sets the comparator ancilla qubits to :math:`\ket{1}` based on the size of the edge in the remaining dimension.
        #. The directional velocity qubits are also set to :math:`\ket{1}` on the specific velocity profile of the targeted particles.
        #. A multi-controlled :math:`X` gate flips the obstacle ancilla qubit, controlled on the qubits set in the previous steps.
        #. The control qubits are set back to their previous state.

        This function resets the ancilla obstacle qubit to :math:`\ket{0}` for particles
        along 36 specific edges of a cube after those particles have been streamed out of the obstacle in the CFL time step.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit on which to perform resetting of the edge state.
        edge : ReflectionResetEdge
            The edge on which to apply the reflection reset logic.

        Returns
        -------
        QuantumCircuit
            The circuit performing the resetting of the edge state.
        """
        comparator_circuit = EdgeComparator(self.lattice, edge, self.logger).circuit

        grid_qubits_indices_to_invert = [
            self.lattice.grid_index(0)[0] + qubit
            for qubit in flatten([wall.qubits_to_invert for wall in edge.walls_joining])
        ]

        control_qubits = flatten(
            [
                self.lattice.ancillae_velocity_index(dim)
                + self.lattice.velocity_dir_index(dim)
                + self.lattice.grid_index(dim)
                for dim in edge.dims_of_edge
            ]
        ) + self.lattice.ancillae_comparator_index(0)

        target_qubits = self.lattice.ancillae_obstacle_index(0)

        circuit.compose(comparator_circuit, inplace=True)

        if grid_qubits_indices_to_invert:
            circuit.x(grid_qubits_indices_to_invert)

        for c, dim in enumerate(edge.dims_of_edge):
            if edge.invert_velocity_in_dimension[c]:
                circuit.x(self.lattice.velocity_dir_index(dim))

        circuit.compose(
            MCMT(
                XGate(),
                len(control_qubits),
                len(target_qubits),
            ),
            qubits=control_qubits + target_qubits,
            inplace=True,
        )

        for c, dim in enumerate(edge.dims_of_edge):
            if edge.invert_velocity_in_dimension[c]:
                circuit.x(self.lattice.velocity_dir_index(dim))

        if grid_qubits_indices_to_invert:
            circuit.x(grid_qubits_indices_to_invert)

        circuit.compose(comparator_circuit, inplace=True)

        return circuit

    def reset_point_state(
        self,
        circuit: QuantumCircuit,
        corner: ReflectionPoint,
    ) -> QuantumCircuit:
        r"""Resets the state of the ancilla obstacle qubit of a single point on the grid as follows.

        #. A series of :math:`X` gates set the grid qubits to the :math:`\ket{1}` state for the dimension that the wall spans.
        #. The directional velocity qubits are also set to :math:`\ket{1}` based on the specific velocity profile of the targeted particles.
        #. A multi-controlled :math:`X` gate flips the obstacle ancilla qubit, controlled on the qubits set in the previous steps.
        #. The control qubits are set back to their previous state.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit on which to perform the resetting of the point state.
        corner : ReflectionPoint
            The corner for which to reset the desired point states.

        Returns
        -------
        QuantumCircuit
            The circuit resetting the point state as desired.
        """
        grid_qubit_indices_to_invert = [
            self.lattice.grid_index(0)[0] + qubit for qubit in corner.qubits_to_invert
        ]

        # If statement required because qiskit does not allow x([])
        # x([]) would only happen for the |11...1> grid point
        # i.e., the bottom-right most grid point.
        if grid_qubit_indices_to_invert:
            # Inverting the qubits that are 0 turns the
            # Grid qubit state encoding this corner lattice point to |11...1>
            # Which in turn allows us to control on this one grid point
            circuit.x(grid_qubit_indices_to_invert)

        for dim in range(corner.num_dims):
            if corner.invert_velocity_in_dimension[dim]:
                circuit.x(self.lattice.velocity_dir_index(dim))

        control_qubits = (
            self.lattice.ancillae_velocity_index()
            + self.lattice.velocity_dir_index()
            + self.lattice.grid_index()
        )

        target_qubits = self.lattice.ancillae_obstacle_index(0)

        circuit.compose(
            MCMT(
                XGate(),
                len(control_qubits),
                len(target_qubits),
            ),
            qubits=control_qubits + target_qubits,
            inplace=True,
        )

        for dim in range(corner.num_dims):
            # Reset the state to the one before the MCMX
            if corner.invert_velocity_in_dimension[dim]:
                circuit.x(self.lattice.velocity_dir_index(dim))

        if grid_qubit_indices_to_invert:
            # Applying the exact same inversion returns
            # The grid qubits to their state before the operation
            circuit.x(grid_qubit_indices_to_invert)

        return circuit

    def flip_and_stream(
        self,
        circuit: QuantumCircuit,
    ):
        """Flips the velocity direction qubit controlled on the ancilla obstacle qubit, before performing streaming.

        Unlike in the regular :class:`.CollisionlessStreamingOperator`, the :class:`.ControlledIncrementer`
        phase shift circuit is additionally controlled on the ancilla obstacle qubit, which
        ensures that only particles whose grid position gets incremented (decremented) are those
        that have streamed inside the solid domain in this CFL time step.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit on which to perform the flip and stream operation.

        """
        control_qubits = self.lattice.ancillae_obstacle_index(0)
        target_qubits = self.lattice.velocity_dir_index()

        circuit.compose(
            MCMT(
                XGate(),
                len(control_qubits),
                len(target_qubits),
            ),
            qubits=control_qubits + target_qubits,
            inplace=True,
        )

        circuit.compose(
            ControlledIncrementer(
                self.lattice, reflection="bounceback", logger=self.logger
            ).circuit,
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Operator BounceBackReflectionOperator against block {self.lattice.blocks}]"
