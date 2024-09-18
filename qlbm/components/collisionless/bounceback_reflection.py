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
from qlbm.lattice import Block, CollisionlessLattice, ReflectionPoint, ReflectionWall

from .primitives import ControlledIncrementer


class BounceBackWallComparator(LBMPrimitive):
    """
    A primitive used in the collision :class:`BounceBackReflectionOperator` that implements the 
    comparator for the bounce back boundary conditions as described :cite:t:`qmem`.

    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    :attr:`wall`              The coordinates of the wall within the grid.
    :attr:`inside_object`     The coordinates of the grid points adjacent to the wall inside the object.
    ========================= ======================================================================
    """
    def __init__(
        self,
        lattice: CollisionlessLattice,
        wall: ReflectionWall,
        inside_object: bool,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice
        self.wall = wall
        self.inside_object = inside_object

        self.circuit = self.create_circuit()

    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        lb_comparators = [
            Comparator(
                self.lattice.num_gridpoints[wall_alignment_dim].bit_length() + 1,
                self.wall.lower_bounds[c],
                ComparatorMode.GT if self.inside_object else ComparatorMode.GE,
                logger=self.logger,
            ).circuit
            for c, wall_alignment_dim in enumerate(self.wall.alignment_dims)
        ]

        ub_comparators = [
            Comparator(
                self.lattice.num_gridpoints[wall_alignment_dim].bit_length() + 1,
                self.wall.upper_bounds[c],
                ComparatorMode.LT if self.inside_object else ComparatorMode.LE,
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
        return f"[Primitive BounceBackWallComparator on wall={self.wall}, inside={self.inside_object}]"


class BounceBackReflectionOperator(CQLBMOperator):
    """
    A primitive that implements the bounce back boundary conditions as described :cite:t:`qmem`.

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

        # Reflect wall points
        for dim in range(self.lattice.num_dimensions):
            for block in self.blocks:
                for wall in block.walls_inside[dim]:
                    self.reflect_wall(circuit, wall, inside_object=True)

        # Reflect inside corners
        for block in self.blocks:
            for corner in block.corners_inside:
                self.reflect_inner_corner(circuit, corner)

        circuit = self.flip_and_stream(circuit)

        # Reflect state for outside wall points
        for dim in range(self.lattice.num_dimensions):
            for block in self.blocks:
                for wall in block.walls_outside[dim]:
                    self.reflect_wall(circuit, wall, inside_object=False)

        # Reset state for near-corner points
        for block in self.blocks:
            for near_corner_point in block.near_corner_points_2d:
                self.reset_point_state(circuit, near_corner_point)

        # Reset state for outside corners
        for block in self.blocks:
            for corner in block.corners_outside:
                self.reset_point_state(circuit, corner)

        return circuit
    def reflect_wall(
        self,
        circuit: QuantumCircuit,
        wall: ReflectionWall,
        inside_object: bool,
    ) -> QuantumCircuit:
        """_summary_

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit on which to perform bounceback reflection of the wall.
        wall : ReflectionWall
            The wall encoding the reflection logic.
        inside_object : bool
            Whether the wall is inside the object.

        Returns
        -------
        QuantumCircuit
            The circuit performing bounceback reflection of the wall.
        """
        comparator_circuit = BounceBackWallComparator(
            self.lattice, wall, inside_object, self.logger
        ).circuit

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
            circuit.x(self.lattice.velocity_dir_index(wall.dim)[0])

        control_qubits = (
            self.lattice.ancillae_velocity_index(wall.dim)
            + self.lattice.velocity_dir_index(wall.dim)
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
            circuit.x(self.lattice.velocity_dir_index(wall.dim)[0])

        if grid_qubit_indices_to_invert:
            # Inverting the qubits that are 0 turns the
            # Dimensional grid qubit state encoding this wall to |11...1>
            # Which in turn allows us to control on this row, in combination with the comparator
            circuit.x(grid_qubit_indices_to_invert)

        circuit.compose(comparator_circuit, inplace=True)

        return circuit

    def reflect_inner_corner(
        self,
        circuit: QuantumCircuit,
        inner_corner: ReflectionPoint,
    ):
        """_summary_

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit on which to perform bounceback reflection of the wall.
        inner_corner : ReflectionPoint
            The inner corner encoding the reflection logic

        """
        grid_qubit_indices_to_invert = [
            self.lattice.grid_index(0)[0] + qubit
            for qubit in inner_corner.qubits_to_invert
        ]

        if grid_qubit_indices_to_invert:
            circuit.x(grid_qubit_indices_to_invert)

        control_qubits = self.lattice.grid_index()
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

        if grid_qubit_indices_to_invert:
            circuit.x(grid_qubit_indices_to_invert)

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

    def __str__(self) -> str:
        return (
            f"[Operator SpecularReflectionOperator against block {self.lattice.blocks}]"
        )

    def reset_point_state(
        self,
        circuit: QuantumCircuit,
        corner: ReflectionPoint,
        # reset_ancillae: bool,
    ) -> QuantumCircuit:
        """_summary_

        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit to input on which to reset the point state.
        corner : ReflectionPoint
            The point on which to reset the state.

        """
        grid_qubit_indices_to_invert = [
            self.lattice.grid_index(0)[0] + qubit for qubit in corner.qubits_to_invert
        ]

        if grid_qubit_indices_to_invert:
            circuit.x(grid_qubit_indices_to_invert)

        for dim in range(corner.num_dims):
            if corner.invert_velocity_in_dimension[dim]:
                circuit.x(self.lattice.velocity_dir_index(dim)[0])

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
            if corner.invert_velocity_in_dimension[dim]:
                circuit.x(self.lattice.velocity_dir_index(dim)[0])

        if grid_qubit_indices_to_invert:
            circuit.x(grid_qubit_indices_to_invert)

        return circuit
