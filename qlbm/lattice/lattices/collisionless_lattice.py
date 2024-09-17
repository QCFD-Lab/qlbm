from logging import Logger, getLogger
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit, QuantumRegister

from qlbm.lattice.blocks import Block
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import dimension_letter, flatten

from .base import Lattice


class CollisionlessLattice(Lattice):
    """Holds the properties of the lattice to simulate."""

    num_dimensions: int
    num_gridpoints: List[int]
    num_velocities: List[int]
    num_total_qubits: int
    registers: Tuple[QuantumRegister, ...]
    logger: Logger

    def __init__(
        self,
        lattice_data: str | Dict,  # type: ignore
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice_data, logger)
        dimensions, velocities, blocks = self.parse_input_data(lattice_data)  # type: ignore

        self.num_dimensions = len(dimensions)
        self.num_gridpoints = dimensions
        self.num_velocities = velocities
        self.blocks: Dict[str, List[Block]] = blocks
        self.block_list = flatten(list(blocks.values()))
        self.num_ancilla_qubits = 2 * self.num_dimensions + 2 * (
            self.num_dimensions - 1
        )
        self.num_grid_qubits = sum([dim.bit_length() for dim in dimensions])
        self.num_velocity_qubits = sum([v.bit_length() for v in velocities])
        self.num_total_qubits = (
            self.num_ancilla_qubits + self.num_grid_qubits + self.num_velocity_qubits
        )

        temporary_registers = self.get_registers()
        (
            self.ancilla_velocity_register,
            self.ancilla_object_register,
            self.ancilla_comparator_register,
            self.grid_registers,
            self.velocity_registers,
            self.velocity_dir_registers,
        ) = temporary_registers
        self.registers = tuple(flatten(temporary_registers))
        self.circuit = QuantumCircuit(*self.registers)

        logger.info(self.__str__())

    def ancillae_velocity_index(self, dim: int | None = None) -> List[int]:
        """Get the indices of the qubits used as velocity ancillae for the specified dimension.

        Parameters
        ----------
        dim : int | None, optional
            The dimension of the grid for which to retrieve the velocity qubit indices, by default `None`.
            When `dim` is `None`, the indices of ancillae qubits for all dimensions are returned.

        Returns
        -------
        List[int]
            A list of indices of the qubits used as velocity ancillae for the given dimension.

        Raises
        ------
        LatticeException
            If the dimension does not exist.
        """

        if dim is None:
            return list(range(self.num_dimensions))

        if dim >= self.num_dimensions or dim < 0:
            raise LatticeException(
                f"Cannot index ancilla velocity register for dimension {dim} in {self.num_dimensions}-dimensional lattice."
            )

        # The velocity ancillas are on the first register, so no offset
        return [dim]

    def ancillae_obstacle_index(self, dim: int | None = None) -> List[int]:
        """Get the indices of the qubits used as obstacle ancilla for the specified dimension.

        Parameters
        ----------
        dim : int | None, optional
            The dimension of the grid for which to retrieve the obstacle qubit index, by default `None`.
            When `dim` is `None`, the indices of ancillae qubits for all dimensions are returned.

        Returns
        -------
        List[int]
            A list of indices of the qubits used as obstacle ancilla for the given dimension.

        Raises
        ------
        LatticeException
            If the dimension does not exist.
        """

        if dim is None:
            return list(range(self.num_dimensions, 2 * self.num_dimensions))

        if dim >= self.num_dimensions or dim < 0:
            raise LatticeException(
                f"Cannot index ancilla obstacle register for dimension {dim} in {self.num_dimensions}-dimensional lattice."
            )

        # There are `d` ancillae velocity qubits "ahead" of this register
        return [self.num_dimensions + dim]

    def ancillae_comparator_index(self, index: int | None = None) -> List[int]:
        """Get the indices of the qubits used as comparator ancillae for the specified index.

        Parameters
        ----------
        index : int | None, optional
            The index for which to retrieve the comparator qubit indices, by default `None`.
            There are `num_dims-1` available indices (i.e., 1 for 2D and 2 for 3D).
            When `index` is `None`, the indices of ancillae qubits for all dimensions are returned.

        Returns
        -------
        List[int]
            A list of indices of the qubits used as obstacle ancilla for the given dimension.
            By convention, the 0th qubit in the returned list is used
            for lower bound comparison and the 1st is used for upper bound comparisons.

        Raises
        ------
        LatticeException
            If the dimension does not exist.
        """
        # Ahead of this register
        # `d` ancillae velocity qubits
        # `d` ancillae reflection qubits
        # `d` ancillae reflection reset qubits
        # 2 * `d` ancillae comparator qubits for "lower" dimensions
        # These are ordered as follows: lx, ux, ly, uy, lz, uz

        if index is None:
            return list(
                range(
                    2 * self.num_dimensions,
                    2 * self.num_dimensions + 2 * (self.num_dimensions - 1),
                )
            )

        if index >= self.num_dimensions - 1 or index < 0:
            raise LatticeException(
                f"Cannot index ancilla comparator register for index {index} in {self.num_dimensions}-dimensional lattice. Maximum is {self.num_dimensions - 2}."
            )
        previous_qubits = 2 * self.num_dimensions + 2 * index
        return list(range(previous_qubits, previous_qubits + 2))

    def grid_index(self, dim: int | None = None) -> List[int]:
        """Get the indices of the qubits used that encode the grid values for the specified dimension.

        Parameters
        ----------
        dim : int | None, optional
            The dimension of the grid for which to retrieve the grid qubit indices, by default `None`.
            When `dim` is `None`, the indices of all grid qubits for all dimensions are returned.

        Returns
        -------
        List[int]
            A list of indices of the qubits used to encode the grid values for the given dimension.

        Raises
        ------
        LatticeException
            If the dimension does not exist.
        """

        if dim is None:
            return list(
                range(
                    self.num_ancilla_qubits,
                    self.num_ancilla_qubits + self.num_grid_qubits,
                )
            )

        if dim >= self.num_dimensions or dim < 0:
            raise LatticeException(
                f"Cannot index grid register for dimension {dim} in {self.num_dimensions}-dimensional lattice."
            )

        # Ahead of this register are
        # 4 * `d` - 2 ancillae qubits
        # log2(ng_i) qubits for "lower" dimensions
        previous_qubits = self.num_ancilla_qubits + sum(
            [self.num_gridpoints[d].bit_length() for d in range(dim)]
        )

        return list(
            range(
                previous_qubits, previous_qubits + self.num_gridpoints[dim].bit_length()
            )
        )

    def velocity_index(self, dim: int | None = None) -> List[int]:
        """Get the indices of the qubits used that encode the velocity magnitude values for the specified dimension.

        Parameters
        ----------
        dim : int | None, optional
            The dimension of the grid for which to retrieve the velocity qubit indices, by default `None`.
            When `dim` is `None`, the indices of all velocity magnitude qubits for all dimensions are returned.

        Returns
        -------
        List[int]
            A list of indices of the qubits used to encode the velocity magnitude values for the given dimension.

        Raises
        ------
        LatticeException
            If the dimension does not exist.
        """
        if dim is None:
            return list(
                range(
                    self.num_ancilla_qubits + self.num_grid_qubits,
                    self.num_ancilla_qubits
                    + self.num_grid_qubits
                    + sum(
                        self.num_velocities[d].bit_length() - 1
                        for d in range(self.num_dimensions)
                    ),
                )
            )

        if dim >= self.num_dimensions or dim < 0:
            raise LatticeException(
                f"Cannot index velocity register for dimension {dim} in {self.num_dimensions}-dimensional lattice."
            )

        # Ahead of this register are
        # 4 * `d` - 2 ancillae qubits
        # The log2(ng_i) qubits encoding the grid points in each dimension
        # The log2(nv_i) - 1 qubits encoding the velocity magnitudes in lower dimensions
        previous_qubits = (
            self.num_ancilla_qubits
            + self.num_grid_qubits
            + sum(self.num_velocities[d].bit_length() - 1 for d in range(dim))
        )

        # Subtract one from the number of qubits required to
        # Encode the velocity discretization due to the directional
        # Qubit that resides in a different register
        return list(
            range(
                previous_qubits,
                previous_qubits + self.num_velocities[dim].bit_length() - 1,
            )
        )

    def velocity_dir_index(self, dim: int | None = None) -> List[int]:
        """Get the indices of the qubit that encodes the velocity direction values for the specified dimension.

        Parameters
        ----------
        dim : int | None, optional
            The dimension of the grid for which to retrieve the velocity direction qubit index, by default `None`.
            When `dim` is `None`, the indices of all velocity direction qubits for all dimensions are returned.

        Returns
        -------
        List[int]
            A list of indices of the qubit used to encode the velocity direction for the given dimension.

        Raises
        ------
        LatticeException
            If the dimension does not exist.
        """
        if dim is None:
            return list(
                range(
                    self.num_ancilla_qubits
                    + self.num_grid_qubits
                    + self.num_velocity_qubits
                    - self.num_dimensions,
                    self.num_total_qubits,
                )
            )

        if dim >= self.num_dimensions or dim < 0:
            raise LatticeException(
                f"Cannot index velocity direction register for dimension {dim} in {self.num_dimensions}-dimensional lattice."
            )

        # Ahead of this register are
        # 4 * `d` - 2 ancillae qubits
        # The log2(ng_i) qubits encoding the grid points in each dimension
        # The log2(nv_i) - 1 qubits encoding the non-directional velocity magnitudes in each dimension
        previous_qubits = (
            self.num_ancilla_qubits
            + sum(
                self.num_gridpoints[d].bit_length() for d in range(self.num_dimensions)
            )
            + sum(v.bit_length() - 1 for v in self.num_velocities)
        )

        return [previous_qubits + dim]

    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        """Generates the encoding-specific register required for the streaming step.
        For this encoding, different registers encode (i) the velocity direction,
        (ii) the velocity discretization, (iii) the velocity ancillae,
        and (iv) the grid encoding.


        Returns
        -------
        List[int]
            Tuple[QuantumRegister]: The 4-tuple of qubit registers encoding the streaming step.
        """

        # d ancilla qubits tracking whether a velocity is to be streamed
        ancilla_vel_register = [QuantumRegister(self.num_dimensions, name="a_v")]

        # d ancilla qubits used to conditionally reflect velocities
        ancilla_object_register = [QuantumRegister(self.num_dimensions, name="a_o")]

        # 2(d-1) ancilla qubits
        ancilla_comparator_register = [
            QuantumRegister(2 * (self.num_dimensions - 1), name="a_c")
        ]

        # d qubits encoding the velocity direction
        velocity_dir_register = [
            QuantumRegister(1, name=f"v_dir_{dimension_letter(c)}")
            for c in range(len(self.num_velocities))
        ]

        # Velocity qubits
        velocity_registers = [
            QuantumRegister(v.bit_length() - 1, name=f"v_{dimension_letter(c)}")
            for c, v in enumerate(self.num_velocities)
        ]

        # Grid qubits
        grid_registers = [
            QuantumRegister(gp.bit_length(), name=f"g_{dimension_letter(c)}")
            for c, gp in enumerate(self.num_gridpoints)
        ]

        return (
            ancilla_vel_register,
            ancilla_object_register,
            ancilla_comparator_register,
            grid_registers,
            velocity_registers,
            velocity_dir_register,
        )

    def __str__(self) -> str:
        return f"[Lattice with {self.num_gridpoints} gps, {self.num_velocities} vels, and {str(self.blocks)} blocks with {self.num_total_qubits} qubits]"

    def logger_name(self) -> str:
        gp_string = ""
        for c, gp in enumerate(self.num_gridpoints):
            gp_string += f"{gp+1}"
            if c < len(self.num_gridpoints) - 1:
                gp_string += "x"
        return f"{self.num_dimensions}d-{gp_string}-{len(self.block_list)}-obstacle"
