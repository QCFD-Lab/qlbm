from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, XGate

from qlbm.components.base import CQLBMOperator, LBMPrimitive
from qlbm.components.common import (
    Comparator,
    ComparatorMode,
)
from qlbm.lattice import (
    Block,
    CollisionlessLattice,
    ReflectionPoint,
    ReflectionResetEdge,
    ReflectionWall,
)
from qlbm.tools.exceptions import CircuitException
from qlbm.tools.utils import flatten

from .primitives import ControlledIncrementer


class SpecularWallComparator(LBMPrimitive):
    """
    A primitive used in the collisionless :class:`SpecularReflectionOperator` that implements the 
    comparator for the specular reflection boundary conditions around the wall as described :cite:t:`collisionless`.

    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    :attr:`wall`              The coordinates of the wall within the grid.
    ========================= ======================================================================
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

    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        lb_comparators = [
            Comparator(
                self.lattice.num_gridpoints[wall_alignment_dim].bit_length() + 1,
                self.wall.lower_bounds[c],
                ComparatorMode.GE,
                logger=self.logger,
            ).circuit
            for c, wall_alignment_dim in enumerate(self.wall.alignment_dims)
        ]

        ub_comparators = [
            Comparator(
                self.lattice.num_gridpoints[wall_alignment_dim].bit_length() + 1,
                self.wall.upper_bounds[c],
                ComparatorMode.LE,
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

    def __str__(self) -> str:
        return f"[Primitive SpecularWallComparator on wall={self.wall}]"


class SpecularEdgeComparator(LBMPrimitive):
    """
    A primitive used in the collisionless :class:`SpecularReflectionOperator` that implements the 
    comparator for the specular reflection boundary conditions around the edge as described :cite:t:`collisionless`.

    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    :attr:`edge`              The coordinates of the edge within the grid.
    ========================= ======================================================================
    """
    def __init__(
        self,
        lattice: CollisionlessLattice,
        edge: ReflectionResetEdge,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice
        self.edge = edge

        self.circuit = self.create_circuit()

    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        lb_comparator = Comparator(
            self.lattice.num_gridpoints[self.edge.dim_disconnected].bit_length() + 1,
            self.edge.bounds_disconnected_dim[0],
            ComparatorMode.GE,
            logger=self.logger,
        ).circuit
        ub_comparator = Comparator(
            self.lattice.num_gridpoints[self.edge.dim_disconnected].bit_length() + 1,
            self.edge.bounds_disconnected_dim[1],
            ComparatorMode.LE,
            logger=self.logger,
        ).circuit

        # for c, wall_alignment_dim in enumerate(self.wall.alignment_dims):
        circuit.compose(
            lb_comparator,
            qubits=self.lattice.grid_index(self.edge.dim_disconnected)
            + self.lattice.ancillae_comparator_index(0)[
                :-1  # :-1 Effectively selects only the first (lb) qubit
            ],  # There are two comparator ancillae, for each relevant dimension, one for l and one for u
            inplace=True,
        )

        circuit.compose(
            ub_comparator,
            qubits=self.lattice.grid_index(self.edge.dim_disconnected)
            + self.lattice.ancillae_comparator_index(0)[
                1:  # 1: Effectively selects only the last (ub) qubit
            ],  # There are two comparator ancillae, for each relevant dimension, one for l and one for u.
            inplace=True,
        )

        return circuit

    def __str__(self) -> str:
        return f"[Primitive SpecularEdgeComparator on edge={self.edge}]"


class SpecularReflectionOperator(CQLBMOperator):
    """
    A primitive that implements the bounce back boundary conditions as described :cite:t:`collisiionless`.

    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    :attr:`blocks`            A geometry encoded in a :class:`.Block` object
    ========================= ======================================================================
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

    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        # Reflect the particles that have streamed
        # Into the inner walls of the object
        for dim in range(self.lattice.num_dimensions):
            for block in self.blocks:
                for wall in block.walls_inside[dim]:
                    self.reflect_wall(circuit, wall)

        # Perform streaming
        circuit = self.flip_and_stream(circuit)

        # Reset the ancilla qubits for the particles that
        # Have been reflected onto the outer walls of the object
        for dim in range(self.lattice.num_dimensions):
            for block in self.blocks:
                for wall in block.walls_outside[dim]:
                    self.reflect_wall(circuit, wall)

        # If this is a 2D simulation
        if self.lattice.num_dimensions == 2:
            # Reset state for near-corner points
            for block in self.blocks:
                for near_corner_point in block.near_corner_points_2d:
                    self.reset_point_state(circuit, near_corner_point)

        # If this is a 3D simulation
        elif self.lattice.num_dimensions == 3:
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
                f"CQBM specular reflection is not supported for {self.lattice.num_dimensions} dimensions."
            )

        # Reset state for outside corners
        for block in self.blocks:
            # Reset the individual points at the corners of the block
            # (8x for 3D, 4x for 2D)
            for corner in block.corners_outside:
                self.reset_point_state(circuit, corner)

        return circuit

    def reset_edge_state(
        self, circuit: QuantumCircuit, edge: ReflectionResetEdge
    ) -> QuantumCircuit:
        """_summary_

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
        comparator_circuit = SpecularEdgeComparator(
            self.lattice, edge, self.logger
        ).circuit

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

        target_qubits = flatten(
            [
                self.lattice.ancillae_obstacle_index(reflected_dim)
                for reflected_dim in edge.reflected_velocities
            ]
        )

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

    def reflect_wall(
        self,
        circuit: QuantumCircuit,
        wall: ReflectionWall,
    ) -> QuantumCircuit:
        """_summary_

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit on which to perform resetting of the edge state.
        wall : ReflectionWall
            The wall encoding the reflection logic.

        Returns
        -------
        QuantumCircuit
            The circuit performing specular reflection of the wall.
        """
        grid_qubit_indices_to_invert = [
            self.lattice.grid_index(0)[0] + qubit
            for qubit in wall.data.qubits_to_invert
        ]

        comparator_circuit = SpecularWallComparator(
            self.lattice, wall, self.logger
        ).circuit

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
            + self.lattice.velocity_dir_index(wall.dim)
            + self.lattice.grid_index(wall.dim)
            + self.lattice.ancillae_comparator_index()
        )

        target_qubits = self.lattice.ancillae_obstacle_index(wall.dim)

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

    def flip_and_stream(
        self,
        circuit: QuantumCircuit,
    ):
        """_summary_

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit on which to perform the flip and stream operation. 
        """
        for dim in range(self.lattice.num_dimensions):
            # Flip the direction of the `d`-directional velocity qubit
            circuit.cx(
                self.lattice.ancillae_obstacle_index(dim)[0],
                self.lattice.velocity_dir_index(dim)[0],
            )

        circuit.compose(
            ControlledIncrementer(
                self.lattice, reflection="specular", logger=self.logger
            ).circuit,
            inplace=True,
        )

        return circuit

    def __str__(self) -> str:
        return (
            f"[Operator SpecularReflectionOperator against block {self.lattice.blocks}]"
        )

    def reset_point_state(
        self,
        circuit: QuantumCircuit,
        corner: ReflectionPoint,
    ) -> QuantumCircuit:
        """_summary_

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

        target_qubits = flatten(
            [
                self.lattice.ancillae_obstacle_index(outside_dim_index)
                for outside_dim_index in corner.dims_outside
            ]
        )

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
