"""Implementation of the :class:`.Lattice` base specific to the 2D and 3D :class:`.LQLGA` algorithm."""

from itertools import product
from math import prod
from typing import Dict, List, Tuple, cast, override

from qiskit import QuantumCircuit, QuantumRegister

from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.lattices.base import Lattice
from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)
from qlbm.tools.exceptions import LatticeException


class LQLGALattice(Lattice):
    r"""
    Lattice class for the :class:`.LQLGA` algorithm.

    ================================= ========================================================================================
    Attribute                         Summary
    ================================= ========================================================================================
    :attr:`num_gridpoints`            The number of gridpoints in each dimension of the lattice.
    :attr:`num_velocities`            The number of discrete velocities in each dimension of the lattice.
    :attr:`num_dims`                  The number of dimensions of the lattice.
    :attr:`discretization`            The discretization of the lattice.
    :attr:`num_velocities_per_point`  The number of discrete velocities per gridpoint.
    :attr:`num_base_qubits`           The number of qubits required to represent the lattice without velocities.
    :attr:`num_total_qubits`          The total number of qubits required to represent the lattice, including velocities.
    :attr:`registers`                 The list of quantum registers for the lattice, one for each gridpoint.
    :attr:`circuit`                   The quantum circuit representing the lattice, initialized with the registers.
    ================================ ========================================================================================

    The registers encoded in the lattice and their accessors are given below.
    For the size of each register,
    and :math:`d` is the total number of dimensions: 2 or 3.

    .. list-table:: Register allocation
        :widths: 25 25 25 50
        :header-rows: 1

        * - Register
          - Size
          - Access Method
          - Description
        * - :attr:`velocity_register`
          - :math:`N_g \cdot q`
          - :meth:`velocity_index_flat` and `:meth:`velocity_index_tuple`
          - :math:`N_g` registers sized according to the number of discrete velocities of the lattice.
    """

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

    velocity_register: QuantumRegister
    """The quantum register representing the velocities of the lattice."""

    def __init__(self, lattice_data, logger=...):
        super().__init__(lattice_data, logger)

        self.num_gridpoints, self.num_velocities, self.shapes, self.discretization = (
            self.parse_input_data(lattice_data)
        )  # type: ignore
        self.num_dims = len(self.num_gridpoints)
        self.num_velocities_per_point = (
            LatticeDiscretizationProperties.get_num_velocities(self.discretization)
        )

        self.num_base_qubits = (
            prod(map(lambda x: x + 1, self.num_gridpoints))
            * self.num_velocities_per_point
        )

        self.num_total_qubits = self.num_base_qubits

        self.registers = self.get_registers()

        self.velocity_register = self.registers

        self.circuit = QuantumCircuit(*self.registers)

    @override
    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        velocity_registers = [
            QuantumRegister(
                LatticeDiscretizationProperties.get_num_velocities(self.discretization),
                name=rf"v^{{{gp_tuple}}}",
            )
            for gp_tuple in list(
                product(*[range(ng + 1) for ng in self.num_gridpoints])
            )
        ]

        return velocity_registers  # type: ignore

    def gridpoint_index_tuple(self, gridpoint: Tuple[int, ...]) -> int:
        """
        Get the lexicographic index of a gridpoint in the lattice.

        Parameters
        ----------
        gridpoint : Tuple[int, ...]
            The gridpoint formatted as a tuple of indices for each dimension.

        Returns
        -------
        int
            The lexicographic index of the gridpoint in the lattice.

        Raises
        ------
        LatticeException
            If the gridpoint index is out of bounds for the lattice.
        """
        if len(gridpoint) != self.num_dims:
            raise LatticeException(
                f"Gridpoint {gridpoint} has incorrect number of dimensions. Expected {self.num_dims}, got {len(gridpoint)}."
            )
        if any(
            gp < 0 or gp >= ng + 1 for gp, ng in zip(gridpoint, self.num_gridpoints)
        ):
            raise LatticeException(
                f"Gridpoint {gridpoint} is out of bounds for the lattice with gridpoints {self.num_gridpoints}."
            )

        flat_index = 0
        multiplier = 1
        num_dims = len(self.num_gridpoints)

        for dim in reversed(range(num_dims)):
            flat_index += gridpoint[dim] * multiplier
            multiplier *= self.num_gridpoints[dim] + 1

        return flat_index

    def gridpoint_index_flat(self, gridpoint: int) -> Tuple[int, ...]:
        """
        Get the tuple representation of a gridpoint index in the lattice.

        Parameters
        ----------
        gridpoint : int
            The lexicographic index of the gridpoint in the lattice.

        Returns
        -------
        Tuple[int, ...]
            The tuple representation of the gridpoint index in the lattice.

        Raises
        ------
        LatticeException
            If the gridpoint index is out of bounds for the lattice.
        """
        if (
            gridpoint < 0
            or gridpoint >= self.num_total_qubits // self.num_velocities_per_point
        ):
            raise LatticeException(
                f"Gridpoint {gridpoint} is out of bounds for the lattice with {self.num_total_qubits // self.num_velocities_per_point} gridpoints."
            )
        indices = []
        for size in reversed(self.num_gridpoints):
            indices.append(gridpoint % (size + 1))
            gridpoint //= size + 1
        return cast(Tuple[int, ...], tuple(reversed(indices)))

    def velocity_index_flat(self, gridpoint: int, velocity: int) -> int:
        """
        Get the index of a qubit representing a particular velocity channel of a given gridpoint.

        Parameters
        ----------
        gridpoint : int
            The lexicographic index of the gridpoint in the lattice.
        velocity : int
            The index of the velocity channel (0-indexed).

        Returns
        -------
        int
            The index of the qubit representing the velocity channel at the specified gridpoint.

        Raises
        ------
        LatticeException
            If the gridpoint or velocity indices are out of bounds for the lattice.
        """
        if velocity < 0 or velocity >= self.num_velocities_per_point:
            raise LatticeException(
                f"Velocity {velocity} is out of bounds for the lattice with {self.num_velocities_per_point} velocities per point."
            )

        if (
            gridpoint < 0
            or gridpoint >= self.num_total_qubits // self.num_velocities_per_point
        ):
            raise LatticeException(
                f"Gridpoint {gridpoint} is out of bounds for the lattice with {self.num_total_qubits // self.num_velocities_per_point} gridpoints."
            )
        return gridpoint * self.num_velocities_per_point + velocity

    def velocity_index_tuple(self, gridpoint: Tuple[int, ...], velocity: int) -> int:
        """
        Get the index of a qubit representing a particular velocity channel of a given gridpoint.

        Parameters
        ----------
        gridpoint : Tuple[int, ...]
            The gridpoint formatted as a tuple of indices for each dimension.
        velocity : int
            The index of the velocity channel (0-indexed).

        Returns
        -------
        int
            The index of the qubit representing the velocity channel at the specified gridpoint.

        Raises
        ------
        LatticeException
            If the gridpoint or velocity indices are out of bounds for the lattice.
        """
        if velocity < 0 or velocity >= self.num_velocities_per_point:
            raise LatticeException(
                f"Velocity {velocity} is out of bounds for the lattice with {self.num_velocities_per_point} velocities per point."
            )
        return (
            self.gridpoint_index_tuple(gridpoint) * self.num_velocities_per_point
            + velocity
        )

    def get_velocity_qubits_of_line(self, line_index: int) -> Tuple[int, int]:
        r"""
        Returns the velocity qubits of the positive and negative directions of a streaming line.

        Assumes that the lattice follows a :math:`D_{d}Q_{q}` discretization, where if :math:`q` is even, there are
        :math:`\lceil \frac{q}{2} \rceil` streaming lines. This is to be generalized in the future.

        Parameters
        ----------
        line_index : int
            The index of the line to get the velocity qubits for. Counted from 0 according to the regular discretization taxonomy.

        Returns
        -------
        List[int]
            The list of velocity qubits for the specified line.
        """
        if line_index < 0 or line_index > self.num_velocities_per_point // 2:
            raise LatticeException(
                f"Streaming Line index {line_index} is out of bounds for the lattice with {self.num_velocities_per_point // 2} lines."
            )

        return (
            (self.num_velocities_per_point % 2) + line_index,
            (self.num_velocities_per_point % 2)
            + line_index
            + self.num_velocities_per_point // 2,
        )

    @override
    def logger_name(self) -> str:
        gp_string = ""
        for c, gp in enumerate(self.num_gridpoints):
            gp_string += f"{gp + 1}"
            if c < len(self.num_gridpoints) - 1:
                gp_string += "x"
        return f"lqlga-d{self.num_dims}-q{self.num_velocities_per_point}-{gp_string}"
