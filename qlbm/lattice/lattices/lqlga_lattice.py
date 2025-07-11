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
    discretization: LatticeDiscretization

    num_gridpoints: List[int]

    shapes: Dict[str, List[Shape]]

    def __init__(self, lattice_data, logger=...):
        super().__init__(lattice_data, logger)

        self.num_gridpoints, self.num_velocities, self.shapes = self.parse_input_data(
            lattice_data
        )  # type: ignore
        self.num_dims = len(self.num_gridpoints)
        self.discretization = self.__get_discretization()
        self.num_velocities_per_point = (
            LatticeDiscretizationProperties.get_num_velocities(self.discretization)
        )

        self.num_base_qubits = (
            prod(map(lambda x: x + 1, self.num_gridpoints))
            * self.num_velocities_per_point
        )

        self.num_total_qubits = self.num_base_qubits

        self.registers = self.get_registers()

        self.circuit = QuantumCircuit(*self.registers)

    def __get_discretization(self) -> LatticeDiscretization:
        if self.num_dims == 1:
            if self.num_velocities[0] == 1:
                return LatticeDiscretization.D1Q2
            raise LatticeException(
                f"Unsupported number of velocities for 1D: {self.num_velocities[0] + 1}. Only D1Q2 is supported at the moment."
            )

        if self.num_dims == 2:
            if self.num_velocities[0] == 1 and self.num_velocities[1] == 1:
                return LatticeDiscretization.D2Q4
            raise LatticeException(
                f"Unsupported number of velocities for 2D: {(self.num_velocities[0] + 1, self.num_velocities[1] + 1)}. Only D2Q4 is supported at the moment."
            )

        if self.num_dims == 3:
            if (
                self.num_velocities[0] == 1
                and self.num_velocities[1] == 1
                and self.num_velocities[2] == 1
            ):
                return LatticeDiscretization.D3Q6
            raise LatticeException(
                f"Unsupported number of velocities for 3D: {(self.num_velocities[0] + 1, self.num_velocities[1] + 1)}. Only D3Q6 is supported at the moment."
            )

        raise LatticeException("Only 1-3D discretizations are currently available.")

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

        return velocity_registers

    def gridpoint_index_tuple(self, gridpoint: Tuple[int, ...]) -> int:
        flat_index = 0
        multiplier = 1
        num_dims = len(self.num_gridpoints)

        for dim in reversed(range(num_dims)):
            flat_index += gridpoint[dim] * multiplier
            multiplier *= self.num_gridpoints[dim] + 1

        return flat_index

    def gridpoint_index_flat(self, gridpoint: int) -> Tuple[int, ...]:
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
            The index of the line to get the velocity qubits for.

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
