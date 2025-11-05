"""Base class for all algorithm-specific Lattices."""

import json
from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit, QuantumRegister

from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.geometry.shapes.circle import Circle
from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import dimension_letter, flatten


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
    :attr:`num_gridpoints`      The number of gridpoints in each dimension.
    :attr:`num_grid_qubits`     The total number of qubits required to encode the grid.
    :attr:`num_velocities`      The number of discrete velocities in each dimension.
    :attr:`num_velocity_qubits` The total number of qubits required to encode the velocity discretization of the lattice.
    :attr:`num_ancilla_qubits`  The total number of ancllary qubits required for the quantum circuit to simulate this lattice.
    :attr:`num_total_qubits`    The total number of qubits required for the quantum circuit to simulate the lattice.
    :attr:`registers`           The qubit registers of the quantum algorithm.
    :attr:`circuit`             The blueprint quantum circuit for all components of the algorithm.
    :attr:`discretization`      The discretization of the lattice, as an enum value of :class:`.LatticeDiscretization`.
    :attr:`shapes`              A list of the solid geometry objects.
    :attr:`logger`              The performance logger.
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
    """
    The number of dimensions of the system.
    """
    num_gridpoints: List[int]
    """
    The number of gridpoints of the lattice in each dimension.
    
    .. warning::

        For easier compatibility with binary arithmetic, the number of gridpoints 
        specified in the input dictionary is one larger than the one held in the :class:`.Lattice` s.
        That is, for a :math:`16 \\times 64` lattice, the ``num_gridpoints`` attribute will have the value ``[15, 63]``.
    """

    num_velocities: List[int]
    """
    The number of velocities in each dimension. This will be refactored in the future to support :math:`D_dQ_q` discretizations.

    .. warning::

        For easier compatibility with binary arithmetic, the number of velocities 
        specified in the input dictionary is one larger than the one held in the :class:`.Lattice` s.
        If the numbers of discrete velocities are :math:`4` and :math:`2`, the ``num_velocities`` attribute will have the value ``[3, 1]``.
    """

    num_grid_qubits: int
    """
    The number of qubits required to encode the grid.
    """

    num_velocity_qubits: int
    """
    The number of qubits required to encode the velocity discretization of the lattice.
    """

    num_ancilla_qubits: int
    """
    The number of ancilla (non-velocity, non-grid) qubits required for the quantum circuit to simulate this lattice.
    """

    num_total_qubits: int
    """
    The total number of qubits required for the quantum circuit to simulate the lattice. This is the sum of the number of grid, velocity, and ancilla qubits.
    """

    velocity_register: Tuple[QuantumRegister, ...]
    """
    A tuple that holds registers responsible for specific operations of the QLBM algorithm.
    """
    circuit: QuantumCircuit
    """
    An empty ``qiskit.QuantumCircuit`` with labeled registers that quantum components use as a base.
    Each quantum component that is parameterized by a :class:`.Lattice` makes a copy of this quantum circuit
    to which it appends its designated logic.
    """

    shapes: Dict[str, List[Shape]]
    """
    Contains all of the :class:`.Shape`s encoding the solid geometry of the lattice. The key of the dictionary is the specific kind of boundary condition of the obstacle (i.e., ``"bounceback"`` or ``"specular"``).
    """

    logger: Logger
    """
    The performance logger, by default ``getLogger("qlbm")``.
    """

    register: Tuple[List[QuantumRegister], ...]
    """
    A tuple of lists of :class:`qiskit.QuantumRegister` s that are used to store the quantum information of the lattice.
    """

    discretization: LatticeDiscretization
    """
    The discretization of the lattice, as an enum value of :class:`.LatticeDiscretization`.
    """

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
        compressed_grid: bool = True,
    ) -> Tuple[List[int], List[int], Dict[str, List[Shape]], LatticeDiscretization]:
        r"""
        Parses the lattice input data, provided in either a file path or a dictionary.

        Parameters
        ----------
        lattice_data : str | Dict
            Either a file path to read a JSON-formatted specification from,
            or a dictionary formatted as in the main class description.
        compressed_grid : bool
            Whether the grid data is compressed into logarithmically many qubits.

        Returns
        -------
        Tuple[List[int], List[int], Dict[str, List[Shape]], LatticeDiscretization]
            A tuple containing
            (i) a list of the number of gridpoints per dimension,
            (ii) a list of the number of velicities per dimension,
            (iii) a dictionary containing the solid :class:`.Shape`\ s,
            and (iv) the discretization enum of the lattice.
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
    ) -> Tuple[List[int], List[int], Dict[str, List[Shape]], LatticeDiscretization]:
        r"""
        Parses the lattice input data, provided as a dictionary.

        Parameters
        ----------
        input_dict : Dict
            The dictionary encoding the lattice.

        Returns
        -------
        Tuple[List[int], List[int], Dict[str, List[Shape]], LatticeDiscretization]
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

        num_dimensions = len(lattice_dict["dim"])  # type: ignore

        if num_dimensions not in [1, 2, 3]:
            raise LatticeException(
                f"Only 1, 2, and 3-dimensional lattices are supported. Provided lattice has {len(lattice_dict['dim'])} dimensions."  # type: ignore
            )

        # Set for access to the geometry parsing utilities
        self.num_dims = num_dimensions

        grid_list: List[int] = [
            # -1 because the bit_length() would "overshoot" for powers of 2
            lattice_dict["dim"][dimension_letter(dim)] - 1
            for dim in range(num_dimensions)
        ]

        self.num_gridpoints = grid_list

        discretization: LatticeDiscretization = LatticeDiscretization.CFLDISCRETIZATION
        velocity_list: List[int] = []

        # Check if velocities is a string (DdQq format) or dict
        if isinstance(lattice_dict["velocities"], str):  # type: ignore
            # Parse DdQq format (e.g., "D2Q4" means 2 dimensions, 4 velocities total)
            velocity_spec = lattice_dict["velocities"].upper()  # type: ignore
            if not velocity_spec.startswith("D") or "Q" not in velocity_spec:
                raise LatticeException(
                    f"Invalid velocity specification format: {lattice_dict['velocities']}. Expected format like 'd2q4'."
                )

            try:
                parts = velocity_spec[1:].split("Q")
                spec_dims = int(parts[0])
                total_velocities = int(parts[1])
                velocity_list = []

                if spec_dims != num_dimensions:
                    raise LatticeException(
                        f"Velocity specification dimensions ({spec_dims}) do not match lattice dimensions ({num_dimensions})."
                    )

                discretization = LatticeDiscretizationProperties.get_discretization(
                    num_dimensions, total_velocities
                )

            except (ValueError, IndexError):
                raise LatticeException(
                    f"Invalid velocity specification format: {lattice_dict['velocities']}. Expected format like 'D<x>Q<y>'."
                )

        else:
            if len(lattice_dict["dim"]) != len(lattice_dict["velocities"]):  # type: ignore
                raise LatticeException(
                    "Lattice configuration dimensionality is inconsistent."
                )

            velocity_list = [
                # -1 because the bit_length() would "overshoot" for powers of 2
                lattice_dict["velocities"][dimension_letter(dim)] - 1
                for dim in range(num_dimensions)
            ]

        if "geometry" not in input_dict:
            return (
                grid_list,
                velocity_list,
                {"specular": [], "bounceback": []},
                discretization,
            )

        geometry_list: List[Dict[str, List[int]]] = input_dict["geometry"]  # type: ignore

        parsed_obstacles = self.parse_geometry_dict(geometry_list)

        return grid_list, velocity_list, parsed_obstacles, discretization

    def parse_geometry_dict(self, geometry_list) -> Dict[str, List[Shape]]:
        """
        Parses the 'geometry' section of the lattice specification.

        Parameters
        ----------
        geometry_list : List[Dict]
            A list of the geometry components. See demos for concrete examples.

        Returns
        -------
        Dict[str, List[Shape]]
            A dictionary where keys consist of boundary conditions (specular or bounceback) and entries consists of all shapes of that boundary condition.
        """
        parsed_obstacles: Dict[str, List[Shape]] = {"specular": [], "bounceback": []}

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
                    len(obstacle_dict) - 2 != self.num_dims
                ):  # -1 to account for the boundary specification
                    raise LatticeException(
                        f"Obstacle {c + 1} has {len(obstacle_dict) - 2} dimensions whereas the lattice has {self.num_dims}."
                    )
                for dim in range(self.num_dims):
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
                        or obstacle_dict[dim_index][-1] > self.num_gridpoints[dim]
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
                            for numeric_dim_index in range(self.num_dims)
                        ],
                        [
                            (self.num_gridpoints[numeric_dim_index]).bit_length()
                            for numeric_dim_index in range(self.num_dims)
                        ],
                        obstacle_dict["boundary"],  # type: ignore
                        num_gridpoints=self.num_gridpoints,
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
                if len(obstacle_dict["center"]) != self.num_dims:
                    raise LatticeException(
                        f"Obstacle {c + 1}: center is {len(obstacle_dict['center'])}-dimensional whereas the lattice is {self.num_dims}-dimensional."
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
                            (self.num_gridpoints[numeric_dim_index]).bit_length()
                            for numeric_dim_index in range(self.num_dims)
                        ],
                        obstacle_dict["boundary"],  # type: ignore
                    )
                )

        return parsed_obstacles

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
                }
                if self.discretization == LatticeDiscretization.CFLDISCRETIZATION
                else LatticeDiscretizationProperties.string_representation[
                    self.discretization
                ],  # type: ignore
            },
        }

        lattice_dict["geometry"] = flatten(
            [
                [block.to_dict() for block in self.shapes[boundary_type]]
                for boundary_type in self.shapes
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

    @abstractmethod
    def has_multiple_geometries(self) -> bool:
        """
        Whether multiple lattice geometries are simulated simultaneously.

        Returns
        -------
        bool
            Whether multiple lattice geometries are simulated simultaneously.
        """
        pass


class AmplitudeLattice(Lattice, ABC):
    r"""
    Abstract Lattice class for QLBM algorithms that use the ampltiude-based encoding.

    The amplitude-based encoding generally maps LBM populations :math:`f_i` onto basis states as :math:`\sqrt{f_i}\ket{x}\ket{v}`,
    with :math:`x` the position and :math:`v` the velocity.
    Amplitude-based encdoings generally compress both the grid register and the velocity register into logarithmically
    many qubits.

    ``qlbm`` currently has 2 amplitude-based lattices: the :class:`.MSLattice` and :class:`.ABLattice` used in the :class:`.MSQLBM` and :class:`.ABQLBM`, respectively.
    """

    def __init__(
        self,
        lattice_data,
        logger=getLogger("qlbm"),
    ):
        super(AmplitudeLattice, self).__init__(lattice_data, logger)

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def velocity_index(self, dim: int | None = None) -> List[int]:
        """Get the indices of the qubits used that encode the velocity magnitude values for the specified dimension.

        Parameters
        ----------
        dim : int | None, optional
            The dimension of the grid for which to retrieve the velocity qubit indices, by default ``None``.
            When ``dim`` is ``None``, the indices of all velocity qubits for all dimensions are returned.
            Note: ``dim`` should only only be supplied to the MSLattice.
            Other amplitude lattices do not support a dimensional breakdown, and ``dim`` should not be passed as an argument.

        Returns
        -------
        List[int]
            A list of indices of the qubits used to encode the velocity magnitude values for the given dimension.

        Raises
        ------
        LatticeException
            If the dimension does not exist.
        """
        pass
