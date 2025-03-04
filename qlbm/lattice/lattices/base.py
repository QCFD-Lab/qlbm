"""Base class for all algorithm-specific Lattices."""

import json
from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit, QuantumRegister

from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.geometry.shapes.circle import Circle
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import dimension_letter, flatten, is_two_pow


class Lattice(ABC):
    r"""Base class for all algorithm-specific Lattices.

    A ``Lattice`` object performs the following functions:

    #. Parse high-level input from JSON files or Python dictionaries into appropriate quantum registers and other information used to infer quantum circuits.
    #. Validate the soundness of the input information and raise warnings if algorithmic assumptions are violated.
    #. Provide parameterized inputs for the inference of quantum circuits that comprise QLBMs.

    The inheritance structure of ``Lattice``\ s is such that each QLBM uses a specialized implementation of this base class.
    This allows the same parsing procedures to be used for all algorithms, while additional validity checks can be built on top.
    All ``Lattice`` objects share the following attributes:

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`num_dims`            The number of dimensions of the lattice.
    :attr:`num_gridpoints`      A ``List[int]`` of the number of gridpoints of the lattice in each dimension.
                                **Important**\ : for easier compatibility with binary arithmetic, the number of gridpoints
                                specified in the input dicitionary is one larger than the one held in the ``Lattice``.
                                That is, for a ``16x64`` lattice, the ``num_gridpoints`` attribute will have the value ``[15, 63]``.
    :attr:`num_grid_qubits`     The total number of qubits required to encode the lattice grid.
    :attr:`num_velocity_qubits` The total number of qubits required to encode the velocity discretization of the lattice.
    :attr:`num_ancilla_qubits`  The total number of ancilla (non-velocity, non-grid) qubits required for the quantum circuit to simulate this lattice.
    :attr:`num_total_qubits`    The total number of qubits required for the quantum circuit to simulate the lattice.
                                This is the sum of the number of grid, velocity, and ancilla qubits.
    :attr:`registers`           A ``Tuple[qiskit.QuantumRegister, ...]`` that holds registers responsible for specific operations of the QLBM algorithm.
    :attr:`circuit`             An empty ``qiskit.QuantumCircuit`` with labeled registers that quantum components use as a base.
                                Each quantum component that is parameterized by a ``Lattice`` makes a copy of this quantum circuit
                                to which it appends its designated logic.
    :attr:`blocks`              A ``Dict[str, List[Block]]`` that contains all of the :class:`.Block`\ s encoding the solid geometry of the lattice.
                                The key of the dictionary is the specific kind of boundary condition of the obstacle (i.e., ``"bounceback"`` or ``"specular"``).
    :attr:`logger`              The performance logger, by default ``getLogger("qlbm")``.
    =========================== ======================================================================

    A lattice can be constructed from from either an input file or a Python dictionary.
    A sample configuration might look as follows:

    .. code-block:: json

        {
            "lattice": {
                "dim": {
                    "x": 16,
                    "y": 16
                },
                "velocities": {
                    "x": 4,
                    "y": 4
                }
            },
            "geometry": [
                {
                    "x": [9, 12],
                    "y": [3, 6],
                    "boundary": "specular"
                },
                {
                    "x": [9, 12],
                    "y": [9, 12],
                    "boundary": "bounceback"
                }
            ]
        }
    """

    num_dims: int
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
        r"""
        Parses the lattice input data, provided in either a file path or a dictionary.

        Parameters
        ----------
        lattice_data : str | Dict
            Either a file path to read a JSON-formatted specification from,
            or a dictionary formatted as in the main class description.

        Returns
        -------
        Tuple[List[int], List[int], Dict[str, List[Block]]]
            A tuple containing
            (i) a list of the number of gridpoints per dimension,
            (ii) a list of the number of velicities per dimension,
            and (iii) a dictionary containing the solid :class:`.Block`\ s.
            The key of the dictionary is the specific kind of
            boundary condition of the obstacle (i.e., ``"bounceback"`` or ``"specular"``).

        Raises
        ------
        LatticeException
            If the specification violates any algorithmic assumptions.
        """
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
    ) -> Tuple[List[int], List[int], Dict[str, List[Shape]]]:
        r"""
        Parses the lattice input data, provided as a dictionary.

        Parameters
        ----------
        input_dict : Dict
            The dictionary encoding the lattice.

        Returns
        -------
        Tuple[List[int], List[int], Dict[str, List[Shape]]]
            A tuple containing
            (i) a list of the number of gridpoints per dimension,
            (ii) a list of the number of velicities per dimension,
            and (iii) a dictionary containing the solid :class:`.Shape`\ s.
            The key of the dictionary is the specific kind of
            boundary condition of the obstacle (i.e., ``"bounceback"`` or ``"specular"``).

        Raises
        ------
        LatticeException
            A specific exception informing the user of the exact kind of assumption that the specification violates.
        """
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

        parsed_obstacles: Dict[str, List[Shape]] = {"specular": [], "bounceback": []}

        if "geometry" not in input_dict:
            return grid_list, velocity_list, parsed_obstacles

        geometry_list: List[Dict[str, List[int]]] = input_dict["geometry"]  # type: ignore

        # Check whether obstacles are well-defined
        for c, obstacle_dict in enumerate(geometry_list):  # type: ignore
            if "boundary" not in obstacle_dict:
                raise LatticeException(
                    f"Obstacle {c + 1} specification includes no boundary conditions."
                )

            if obstacle_dict["boundary"] not in parsed_obstacles:
                raise LatticeException(
                    f"Obstacle {c + 1} boundary conditions ('{obstacle_dict['boundary']}') are not supported. Supported boundary conditions are {list(parsed_obstacles)}."
                )

            if "shape" not in obstacle_dict:
                raise LatticeException(
                    f"Obstacle {c + 1} specification includes no shape."
                )

            if obstacle_dict["shape"] not in ["cuboid", "sphere"]:
                raise LatticeException(
                    f'Obstacle {c + 1} has unsupported shape "{obstacle_dict["shape"]}". Supported shapes are cuboid and sphere.'
                )
            # Parsing blocks
            if obstacle_dict["shape"] == "cuboid":
                if (
                    len(obstacle_dict) - 2 != num_dimensions
                ):  # -1 to account for the boundary specification
                    raise LatticeException(
                        f"Obstacle {c + 1} has {len(obstacle_dict) - 2} dimensions whereas the lattice has {num_dimensions}."
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
                        obstacle_dict["boundary"],  # type: ignore
                    )
                )
            elif obstacle_dict["shape"] == "sphere":
                if "center" not in obstacle_dict:
                    raise LatticeException(
                        f"Obstacle {c + 1}: sphere obstacle does not specify a center."
                    )
                if "radius" not in obstacle_dict:
                    raise LatticeException(
                        f"Obstacle {c + 1}: sphere obstacle does not specify a radius."
                    )
                if len(obstacle_dict["center"]) != num_dimensions:
                    raise LatticeException(
                        f"Obstacle {c + 1}: center is {len(obstacle_dict['center'])}-dimensional whereas the lattice is {num_dimensions}-dimensional."
                    )
                if int(obstacle_dict["radius"]) <= 0:  # type: ignore
                    raise LatticeException(
                        f"Obstacle {c + 1}: radius {obstacle_dict['radius']} is not a natural number."
                    )
                parsed_obstacles[obstacle_dict["boundary"]].append(  # type: ignore
                    Circle(
                        tuple(int(coord) for coord in obstacle_dict["center"]),
                        int(obstacle_dict["radius"]),  # type: ignore
                        [
                            (grid_list[numeric_dim_index]).bit_length()
                            for numeric_dim_index in range(num_dimensions)
                        ],
                        obstacle_dict["boundary"],  # type: ignore
                    )
                )

        return grid_list, velocity_list, parsed_obstacles

    def to_json(self) -> str:
        """
        Serialize the lattice specification to JSON format.

        Returns
        -------
        str
            The JSON representation of the lattice.
        """
        lattice_dict: Dict = {  # type: ignore
            "lattice": {
                "dim": {
                    dimension_letter(dim): self.num_gridpoints[dim] + 1
                    for dim in range(self.num_dims)
                },
                "velocities": {
                    dimension_letter(dim): self.num_velocities[dim] + 1
                    for dim in range(self.num_dims)
                },
            },
        }

        lattice_dict["geometry"] = flatten(
            [
                [block.to_dict() for block in self.blocks[boundary_type]]
                for boundary_type in self.blocks
            ]
        )

        return json.dumps(lattice_dict)

    @abstractmethod
    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        """
        Generates the registers on which the quantum circuits will be placed.

        Returns
        -------
        Tuple[List[QuantumRegister], ...]
            A fixed number of registers according to the lattice specification.
        """
        pass

    @abstractmethod
    def logger_name(self) -> str:
        """
        An identifiable name to be used in the logger to help with benchmarking and analysis.

        Returns
        -------
        str
            A string that can be used to sufficiently identify the lattice specification.
        """
        pass
