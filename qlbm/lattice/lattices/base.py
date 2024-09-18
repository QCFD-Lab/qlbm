import json
from abc import ABC, abstractmethod
from enum import Enum
from logging import Logger, getLogger
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit, QuantumRegister

from qlbm.lattice.blocks import Block
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import dimension_letter, flatten, is_two_pow


class LatticeDiscretization(Enum):
    D2Q4 = (1,)


class Lattice(ABC):
    """Holds the properties of the lattice to simulate."""

    num_dimensions: int
    num_gridpoints: List[int]
    num_velocities: List[int]
    num_total_qubits: int
    registers: Tuple[QuantumRegister, ...]
    logger: Logger
    circuit: QuantumCircuit
    blocks: Dict[str, List[Block]]

    def __init__(
        self,
        lattice_data: str | Dict,  # type: ignore
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__()
        self.logger = logger

    def parse_input_data(
        self,
        lattice_data: str | Dict,  # type: ignore
    ) -> Tuple[List[int], List[int], Dict[str, List[Block]]]:
        if isinstance(lattice_data, str):
            with open(lattice_data, "r") as f:
                lattice_dict = json.load(f)
                return self.__parse_input_dict(lattice_dict)  # type: ignore
        elif isinstance(lattice_data, Dict):  # type: ignore
            return self.__parse_input_dict(lattice_data)  # type: ignore
        else:
            raise LatticeException(f"Invalid lattice specification: {lattice_data}.")

    def __parse_input_dict(
        self,
        input_dict: Dict,  # type: ignore
    ) -> Tuple[List[int], List[int], Dict[str, List[Block]]]:
        if "lattice" not in input_dict:
            raise LatticeException('Input configuration missing "lattice" properties.')

        lattice_dict: Dict[str, Dict[str, int]] = input_dict["lattice"]  # type: ignore

        # Check whether the lattice specification is consistent
        if "dim" not in lattice_dict:
            raise LatticeException('Lattice configuration missing "dim" properties.')

        if "velocities" not in lattice_dict:
            raise LatticeException(
                'Lattice configuration missing "velocities" properties.'
            )

        if len(lattice_dict["dim"]) != len(lattice_dict["velocities"]):  # type: ignore
            raise LatticeException(
                "Lattice configuration dimensionality is inconsistent."
            )

        num_dimensions = len(lattice_dict["dim"])  # type: ignore

        if num_dimensions not in [1, 2, 3]:
            raise LatticeException(
                f"Only 1, 2, and 3-dimensional lattices are supported. Provided lattice has {len(lattice_dict['dim'])} dimensions."  # type: ignore
            )

        # Check whether the number of grid points and velocities is compatible
        for dim in range(num_dimensions):
            dim_index = dimension_letter(dim)

            if not is_two_pow(lattice_dict["dim"][dim_index]):  # type: ignore
                raise LatticeException(
                    f"Lattice {dim_index}-dimension has a number of grid points that is not divisible by 2."
                )

            if not is_two_pow(lattice_dict["velocities"][dim_index]):  # type: ignore
                raise LatticeException(
                    f"Lattice {dim_index}-dimension has a number of velocities that is not divisible by 2."
                )

        # The lattice properties are ok
        grid_list: List[int] = [
            # -1 because the bit_length() would "overshoot" for powers of 2
            lattice_dict["dim"][dimension_letter(dim)] - 1
            for dim in range(num_dimensions)
        ]
        velocity_list: List[int] = [
            # -1 because the bit_length() would "overshoot" for powers of 2
            lattice_dict["velocities"][dimension_letter(dim)] - 1
            for dim in range(num_dimensions)
        ]

        parsed_obstacles: Dict[str, List[Block]] = {"specular": [], "bounceback": []}

        if "geometry" not in input_dict:
            return grid_list, velocity_list, parsed_obstacles

        geometry_list: List[Dict[str, List[int]]] = input_dict["geometry"]  # type: ignore

        # Check whether obstacles are well-defined
        for c, obstacle_dict in enumerate(geometry_list):  # type: ignore
            if "boundary" not in obstacle_dict:
                raise LatticeException(
                    f"Obstacle {c + 1} specification includes no boundary conditions."
                )

            if (
                len(obstacle_dict) - 1 != num_dimensions
            ):  # -1 to account for the boundary specification
                raise LatticeException(
                    f"Obstacle {c+1} has {len(obstacle_dict) - 1} dimensions whereas the lattice has {num_dimensions}."
                )

            for dim in range(num_dimensions):
                dim_index = dimension_letter(dim)

                if len(obstacle_dict[dim_index]) != 2:
                    raise LatticeException(
                        f"Obstacle {c + 1} is ill-formed in dimension {dim_index}."
                    )

                if sorted(obstacle_dict[dim_index]) != obstacle_dict[dim_index]:
                    raise LatticeException(
                        f"Obstacle {c + 1} {dim_index}-dimension bounds are not increasing."
                    )

                if (
                    obstacle_dict[dim_index][0] < 0
                    or obstacle_dict[dim_index][-1] > lattice_dict["dim"][dim_index]
                ):
                    raise LatticeException(
                        f"Obstacle {c + 1} is out of bounds in the {dim_index}-dimension."
                    )

            if obstacle_dict["boundary"] not in parsed_obstacles:
                raise LatticeException(
                    f"Obstacle {c + 1} boundary conditions ('{obstacle_dict['boundary']}') are not supported. Supported boundary conditions are {list(parsed_obstacles)}."
                )

            # TODO check overlap
            parsed_obstacles[obstacle_dict["boundary"]].append(  # type: ignore
                Block(
                    [
                        (
                            obstacle_dict[dimension_letter(numeric_dim_index)][0],
                            obstacle_dict[dimension_letter(numeric_dim_index)][1],
                        )
                        for numeric_dim_index in range(num_dimensions)
                    ],
                    [
                        (grid_list[numeric_dim_index]).bit_length()
                        for numeric_dim_index in range(num_dimensions)
                    ],
                )
            )

        return grid_list, velocity_list, parsed_obstacles

    def to_json(self) -> str:
        lattice_dict: Dict = {  # type: ignore
            "lattice": {
                "dim": {
                    dimension_letter(dim): self.num_gridpoints[dim] + 1
                    for dim in range(self.num_dimensions)
                },
                "velocities": {
                    dimension_letter(dim): self.num_velocities[dim] + 1
                    for dim in range(self.num_dimensions)
                },
            },
        }

        lattice_dict["geometry"] = flatten(
            [
                [block.to_dict(boundary_type) for block in self.blocks[boundary_type]]
                for boundary_type in self.blocks
            ]
        )

        return json.dumps(lattice_dict)

    @abstractmethod
    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        pass

    @abstractmethod
    def logger_name(self) -> str:
        pass
