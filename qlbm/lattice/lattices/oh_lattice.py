"""Implementation of the One-Hot (OH) encoding lattice for generic DdQq discretizations."""

from logging import getLogger
from typing import Dict, List, Tuple

from numpy import ceil, log2
from qiskit import QuantumCircuit, QuantumRegister
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.lattices.base import AmplitudeLattice
from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import dimension_letter, flatten, is_two_pow

from .ab_lattice import ABLattice


class OHLattice(ABLattice):
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
    """The registers of the lattice."""

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
        self.num_velocity_qubits = self.num_velocities_per_point
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

    @override
    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        """Generates the encoding-specific register required for the streaming step.

        For this encoding, different registers encode
        (i) the logarithmically compressed grid,
        (ii) the uncompressed discrete velocities,
        (iii) the comparator qubits,
        (iv) the object qubits.

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
        return f"ohlattice-{self.num_dims}d-{gp_string}-{len(flatten(list(self.shapes.values())))}-obstacle"

    @override
    def has_multiple_geometries(self):
        return False  # multiple geometries unsupported for ABQLBM right now

    @override
    def get_encoding(self) -> ABEncodingType:
        return ABEncodingType.OH

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
