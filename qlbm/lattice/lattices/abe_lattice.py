"""Implementation of the Amplitude-Based Encoding (ABE) lattice for generic DdQq discretizations."""

from logging import getLogger
from typing import Dict, List, Tuple

from numpy import ceil, log2
from qiskit import QuantumCircuit, QuantumRegister
from typing_extensions import override

from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import dimension_letter, flatten, is_two_pow

from .base import Lattice


class ABELattice(Lattice):
    """TODO."""

    discretization: LatticeDiscretization
    """The discretization of the lattice, one of :class:`.LatticeDiscretization`."""

    num_gridpoints: List[int]
    """The number of gridpoints in each dimension of the lattice.
    **Important** : for easier compatibility with binary arithmetic, the number of gridpoints
    specified in the input dictionary is one larger than the one held in the ``Lattice``."""

    shapes: Dict[str, List[Shape]]
    """The shapes of the lattice, which are used to define the geometry of the lattice.
    The key consists of the type of the shape and the name of the shape, e.g. "bounceback" or "specular".
    """

    num_base_qubits: int
    """The number of qubits required to represent the lattice."""

    registers: Tuple[QuantumRegister, ...]

    def __init__(
        self,
        lattice_data,
        logger=getLogger("qlbm"),
    ):
        super().__init__(lattice_data, logger)
        self.num_gridpoints, self.num_velocities, self.shapes, self.discretization = (
            self.parse_input_data(lattice_data)
        )  # type: ignore
        self.geometries: List[Dict[str, List[Shape]]] = [self.shapes]
        self.num_dims = len(self.num_gridpoints)
        self.num_velocities_per_point = (
            LatticeDiscretizationProperties.get_num_velocities(self.discretization)
        )

        for dim in range(self.num_dims):
            if not is_two_pow(self.num_gridpoints[dim] + 1):  # type: ignore
                raise LatticeException(
                    f"Lattice has a number of grid points that is not divisible by 2 in dimension {dimension_letter(dim)}."
                )

        self.num_grid_qubits = int(
            sum(map(lambda x: ceil(log2(x)), self.num_gridpoints))
        )
        self.num_velocity_qubits = int(ceil(log2(self.num_velocities_per_point)))
        self.num_base_qubits = self.num_grid_qubits + self.num_velocity_qubits

        self.num_obstacle_qubits = self.__num_obstacle_qubits()
        self.num_comparator_qubits = 2 * (self.num_dims - 1)
        self.num_ancilla_qubits = self.num_comparator_qubits + self.num_obstacle_qubits

        self.num_total_qubits = self.num_base_qubits + self.num_ancilla_qubits

        temporary_registers = self.get_registers()
        (
            self.grid_registers,
            self.velocity_registers,
            self.ancilla_comparator_register,
            self.ancilla_object_register,
        ) = temporary_registers

        self.registers = tuple(flatten(temporary_registers))
        self.circuit = QuantumCircuit(*self.registers)

    def grid_index(self, dim: int | None = None) -> List[int]:
        """Get the indices of the qubits used that encode the grid values for the specified dimension.

        Parameters
        ----------
        dim : int | None, optional
            The dimension of the grid for which to retrieve the grid qubit indices, by default ``None``.
            When ``dim`` is ``None``, the indices of all grid qubits for all dimensions are returned.

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
                    self.num_grid_qubits,
                )
            )

        if dim >= self.num_dims or dim < 0:
            raise LatticeException(
                f"Cannot index grid register for dimension {dim} in {self.num_dims}-dimensional lattice."
            )

        previous_qubits = sum([self.num_gridpoints[d].bit_length() for d in range(dim)])

        return list(
            range(
                previous_qubits, previous_qubits + self.num_gridpoints[dim].bit_length()
            )
        )

    def velocity_index(self) -> List[int]:
        """Get the indices of the qubits used that encode the discrete velocities.

        Returns
        -------
        List[int]
            A list of indices of the qubits that encode the velocity discretization.
        """
        return list(
            range(
                self.num_grid_qubits,
                self.num_grid_qubits + self.num_velocity_qubits,
            )
        )

    def ancillae_comparator_index(self, index: int | None = None) -> List[int]:
        """Get the indices of the qubits used as comparator ancillae for the specified index.

        Parameters
        ----------
        index : int | None, optional
            The index for which to retrieve the comparator qubit indices, by default ``None``.
            There are `num_dims-1` available indices (i.e., 1 for 2D and 2 for 3D).
            When `index` is ``None``, the indices of ancillae qubits for all dimensions are returned.

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
        if index is None:
            return list(
                range(
                    self.num_base_qubits,
                    self.num_base_qubits + 2 * (self.num_dims - 1),
                )
            )

        if index >= self.num_dims - 1 or index < 0:
            raise LatticeException(
                f"Cannot index ancilla comparator register for index {index} in {self.num_dims}-dimensional lattice. Maximum is {self.num_dims - 2}."
            )

        return list(
            range(
                self.num_base_qubits, self.num_base_qubits + self.num_comparator_qubits
            )
        )

    def ancillae_obstacle_index(self, index: int | None = None) -> List[int]:
        """Get the indices of the qubits used as obstacle ancilla for the specified dimension.

        Parameters
        ----------
        index : int | None, optional
            The index of the grid for which to retrieve the obstacle qubit index, by default ``None``.
            When ``index`` is ``None``, the indices of ancillae qubits for all dimensions are returned.
            For 2D lattices with only bounce-back boundary-conditions, only one obstacle
            qubit is required.
            For all other configurations, the algorithm uses ``2d-2`` obstacle qubits.

        Returns
        -------
        List[int]
            A list of indices of the qubits used as obstacle ancilla for the given dimension.

        Raises
        ------
        LatticeException
            If the dimension does not exist.
        """
        if index is None:
            return list(
                range(
                    self.num_base_qubits + self.num_comparator_qubits,
                    self.num_base_qubits
                    + self.num_comparator_qubits
                    + self.num_obstacle_qubits,
                )
            )

        if index >= self.num_obstacle_qubits or index < 0:
            raise LatticeException(
                f"Cannot index ancilla obstacle register for index {index}. Maximum index for this lattice is {self.num_obstacle_qubits - 1}."
            )

        return [self.num_base_qubits + self.num_comparator_qubits + index]

    def __num_obstacle_qubits(self) -> int:
        all_obstacle_bounceback: bool = len(
            [
                b
                for b in flatten(list(self.shapes.values()))
                if b.boundary_condition == "bounceback"
            ]
        ) == len(flatten(list(self.shapes.values())))
        if all_obstacle_bounceback:
            # A single qubit suffices to determine
            # Whether particles have streamed inside the object
            return 1
        # If there is at least one object with specular reflection
        # 2 ancilla qubits are required for velocity inversion
        else:
            return self.num_dims

    @override
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
        # d ancilla qubits used to conditionally reflect velocities
        ancilla_object_register = [
            QuantumRegister(self.num_obstacle_qubits, name="a_o")
        ]

        # 2(d-1) ancilla qubits
        ancilla_comparator_register = [
            QuantumRegister(self.num_comparator_qubits, name="a_c")
        ]

        # Velocity qubits
        velocity_registers = [QuantumRegister(self.num_velocity_qubits, name="v")]

        # Grid qubits
        grid_registers = [
            QuantumRegister(gp.bit_length(), name=f"g_{dimension_letter(c)}")
            for c, gp in enumerate(self.num_gridpoints)
        ]

        return (
            grid_registers,
            velocity_registers,
            ancilla_comparator_register,
            ancilla_object_register,
        )

    @override
    def logger_name(self) -> str:
        gp_string = ""
        for c, gp in enumerate(self.num_gridpoints):
            gp_string += f"{gp + 1}"
            if c < len(self.num_gridpoints) - 1:
                gp_string += "x"
        return f"abelattice-{self.num_dims}d-{gp_string}-{len(flatten(list(self.shapes.values())))}-obstacle"

    @override
    def has_multiple_geometries(self):
        return False  # multiple geometries unsupported for CQBM
