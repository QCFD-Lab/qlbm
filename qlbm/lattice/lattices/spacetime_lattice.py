from enum import Enum
from logging import Logger, getLogger
from typing import Dict, List, Tuple, cast

from qiskit import QuantumCircuit, QuantumRegister

from qlbm.lattice.blocks import Block
from qlbm.tools.exceptions import CircuitException, LatticeException
from qlbm.tools.utils import dimension_letter, flatten

from .base import Lattice


class VonNeumannNeighborType(Enum):
    ORIGIN = (0,)
    INTERMEDIATE = (1,)
    EXTREME = (2,)


class VonNeumannNeighbor:
    def __init__(
        self,
        relative_coordinates: Tuple[int, int],
        neighborhood_index: int,
        neighbor_type: VonNeumannNeighborType,
    ):
        self.coordinates_relative = relative_coordinates
        self.coordinates_inverse = [-1 * coord for coord in relative_coordinates]
        self.neighbor_index = neighborhood_index
        self.neighbor_type = neighbor_type

    def __eq__(self, other):
        if not isinstance(other, VonNeumannNeighbor):
            return NotImplemented

        return (
            all(
                c0 == c1
                for c0, c1 in zip(self.coordinates_relative, other.coordinates_relative)
            )
            and all(
                c0 == c1
                for c0, c1 in zip(self.coordinates_inverse, other.coordinates_inverse)
            )
            and self.neighbor_index == other.neighbor_index
            and self.neighbor_type == other.neighbor_type
        )

    def get_absolute_values(
        self, origin: Tuple[int, int], relative: bool
    ) -> Tuple[int, int]:
        return cast(
            Tuple[int, int],
            tuple(
                c0 + c1
                for c0, c1 in zip(
                    (
                        self.coordinates_relative
                        if relative
                        else self.coordinates_inverse
                    ),
                    origin,
                )
            ),
        )

    def velocity_index_to_swap(self, point_class: int, timestep: int) -> int:
        if timestep == 1:
            return (point_class + 2) % 4
        else:
            raise CircuitException("Not implemented.")


class SpaceTimeLattice(Lattice):
    """
    Implementation of the :class:`.Lattice` base specific to the 2D and 3D :class:`.SpaceTimeQLBM` algorithm developed by :cite:t:`spacetime`.

    .. warning::
        The STQBLM algorithm is a based on typical :math:`D_dQ_q` discretizations.
        The current implementation only supports :math:`D_2Q_4` for one time step.
        This is work in progress.
        Multiple steps are possible through ``qlbm``\ 's reinitialization mechanism.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`num_timesteps`       The number of time steps the lattice should be simulated for.
    :attr:`num_dims`            The number of dimensions of the lattice.
    :attr:`num_gridpoints`      A ``List[int]`` of the number of gridpoints of the lattice in each dimension.
                                **Important**\ : for easier compatibility with binary arithmetic, the number of gridpoints
                                specified in the input dicitionary is one larger than the one held in the ``Lattice``.
                                That is, for a ``16x64`` lattice, the ``num_gridpoints`` attribute will have the value ``[15, 63]``.
    :attr:`num_grid_qubits`     The total number of qubits required to encode the lattice grid.
    :attr:`num_velocity_qubits` The total number of qubits required to encode the velocity discretization of the lattice.
    :attr:`num_ancilla_qubits`  The total number of ancilla (non-velocity, non-grid) qubits required for the quantum circuit to simulate this lattice.
                                There are no ancilla qubits for the Space-Time QLBM.
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

    The registers encoded in the lattice and their accessors are given below.
    For the size of each register,
    :math:`N_{g_j}` is the number of grid points of dimension :math:`j` (i.e., 64, 128),
    :math:`N_{v_j}` is the number of discrete velocities of dimension :math:`j` (i.e., 2, 4),
    and :math:`d` is the total number of dimensions: 2 or 3.

    .. list-table:: Register allocation
        :widths: 25 25 25 50
        :header-rows: 1

        * - Register
          - Size
          - Access Method
          - Description
        * - :attr:`grid_registers`
          - :math:`\Sigma_{1\leq j \leq d} \left \lceil{\log N_{g_j}} \\right \\rceil`
          - :meth:`grid_index`
          - The qubits encoding the physical grid.
        * - :attr:`velocity_registers`
          - :math:`\min(N_g \cdot N_v, \\frac{N_v^2\cdot N_t \cdot (N_t + 1)}{2} + N_v)`
          - :meth:`velocity_index`
          - The qubits encoding local and neighboring velocities.

    A lattice can be constructed from from either an input file or a Python dictionary.
    Currently, only the :math:`D_2Q_4` discretization is supported, and no boundary conditions are implemented.
    A sample configuration might look as follows. Keep in mind that the velocity and geometry section
    should not be altered in this current implementation.

    .. code-block:: json

        {
            "lattice": {
                "dim": {
                    "x": 16,
                    "y": 16
                },
                "velocities": {
                    "x": 2,
                    "y": 2
                }
            },
            "geometry": []
        }

    The register setup can be visualized by constructing a lattice object:

    .. plot::
        :include-source:

        from qlbm.lattice import SpaceTimeLattice

        SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
            }
        ).circuit.draw("mpl")
    """

    # ! Only works for D2Q4

    # Points with 3 neighbors with higher Manhattan distances
    # In order:
    #   ^   |   ^   |   ^   |   x   |
    # x   > | <   > | <   x | <   > |
    #   v   |   x   |   v   |   v   |
    extreme_point_classes: List[Tuple[int, Tuple[int, int]]] = [
        (0, (1, 0)),
        (1, (0, 1)),
        (2, (-1, 0)),
        (3, (0, -1)),
    ]

    # Points with 2 neighbors with higher Manhattan distances
    # In order:
    #   ^   |   ^   |   x   |   x   |
    # x   > | <   x | <   x | x   > |
    #   x   |   x   |   v   |   v   |
    intermediate_point_classes: List[Tuple[int, Tuple[int, int]]] = [
        (0, (-1, 1)),
        (1, (-1, -1)),
        (2, (1, -1)),
        (3, (1, 1)),
    ]

    # The origin point's neighbors all have higher Manhattan distances (1)
    #   ^   |
    # <   > |
    #   v   |
    origin_point_class: List[int] = [0]

    def __init__(
        self,
        num_timesteps: int,
        lattice_data: str | Dict,  # type: ignore
        logger: Logger = getLogger("qlbm/"),
    ):
        super().__init__(lattice_data, logger)
        self.num_gridpoints, self.num_velocities, self.blocks = self.parse_input_data(
            lattice_data
        )  # type: ignore
        # Adding the length corrects for the fact that the parser subtracts 1
        # from the input to get the correct bit length.
        self.total_gridpoints: int = (
            sum(self.num_gridpoints) + len(self.num_gridpoints)
        ) * (sum(self.num_gridpoints) + len(self.num_gridpoints))
        self.num_velocities_per_point: int = sum(2**v for v in self.num_velocities)
        self.num_timesteps = num_timesteps
        self.num_grid_qubits: int = sum(
            num_gridpoints_in_dim.bit_length()
            for num_gridpoints_in_dim in self.num_gridpoints
        )
        self.num_velocity_qubits: int = self.num_required_velocity_qubits(num_timesteps)
        self.num_ancilla_qubits = 0
        self.num_total_qubits = (
            self.num_ancilla_qubits + self.num_grid_qubits + self.num_velocity_qubits
        )
        self.block_list: List[Block] = []
        temporary_registers = self.get_registers()
        (
            self.grid_registers,
            self.velocity_registers,
        ) = temporary_registers
        self.registers = tuple(flatten(temporary_registers))
        self.circuit = QuantumCircuit(*self.registers)
        (
            self.extreme_point_indices,
            self.intermediate_point_indices,
        ) = self.get_neighbor_indices()
        self.num_dims = len(self.num_gridpoints)

        logger.info(self.__str__())

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
            return list(range(self.num_grid_qubits))

        if dim >= self.num_dims or dim < 0:
            raise LatticeException(
                f"Cannot index grid register for dimension {dim} in {self.num_dims}-dimensional lattice."
            )

        previous_qubits = sum(self.num_gridpoints[d].bit_length() for d in range(dim))

        return list(
            range(
                previous_qubits, previous_qubits + self.num_gridpoints[dim].bit_length()
            )
        )

    def velocity_index(
        self,
        point_neighborhood_index: int,
        velocity_direction: int | None = None,
    ) -> List[int]:
        """Get the indices of the qubits used that encode the velocity for a specific neighboring grid point and direction.

        Parameters
        ----------
        point_neighborhood_index : int
            The index of the grid point neighbor.
        velocity_direction : int | None, optional
            The index of the discrete velocity according to the LBM discretization, by default None.
            When ``velocity_direction`` is ``None``, the indices of all velocity qubits of the neighbor are returned.

        Returns
        -------
        List[int]
            A list of indices of the qubits that encode the specific neighbor, velocity pair.
        """
        if velocity_direction is None:
            return list(
                range(
                    self.num_grid_qubits
                    + point_neighborhood_index * self.num_velocities_per_point,
                    self.num_grid_qubits
                    + (point_neighborhood_index + 1) * (self.num_velocities_per_point),
                )
            )
        return [
            self.num_grid_qubits
            + point_neighborhood_index * self.num_velocities_per_point
            + velocity_direction
        ]

    def grid_neighbors(
        self, coordinates: Tuple[int, int], up_to_distance: int
    ) -> List[List[int]]:
        """


        Parameters
        ----------
        coordinates : Tuple[int, int]
            _description_
        up_to_distance : int
            _description_

        Returns
        -------
        List[List[int]]
            _description_
        """
        return [
            [
                (coordinates[0] + x_offset) % (self.num_gridpoints[0] + 1),
                (coordinates[1] + y_offset) % (self.num_gridpoints[1] + 1),
            ]
            for x_offset in range(-up_to_distance, up_to_distance + 1)
            for y_offset in range(
                abs(x_offset) - up_to_distance, up_to_distance + 1 - abs(x_offset)
            )
        ]

    def num_required_velocity_qubits(
        self,
        num_timesteps: int,
    ) -> int:
        """
        Computes the number of required velocity qubits for a number of time steps to be simulated, :math:`\min(N_g \cdot N_v, \\frac{N_v^2\cdot N_t \cdot (N_t + 1)}{2} + N_v)`.

        Parameters
        ----------
        num_timesteps : int
            The number of time steps to be simulated.

        Returns
        -------
        int
            The number of velocity qubits required.
        """
        # Generalization of equation 17 of the paper
        # ! TODO generalize to 3D
        return min(
            self.total_gridpoints * self.num_velocities_per_point,
            int(
                self.num_velocities_per_point
                * self.num_velocities_per_point
                * num_timesteps
                * (num_timesteps + 1)
                * 0.5
                + self.num_velocities_per_point
            ),
        )

    def num_gridpoints_within_distance(self, manhattan_distance: int) -> int:
        """
        Calculate the number of grid points within a given Manhattan distance.

        Parameters
        ----------
        manhattan_distance : int
            The Manhattan distance up to which (and including) to include the gridpoints.

        Returns
        -------
        int
            The number of gridpoints, including the origin, within the given Manhattan distance.
        """
        return int(
            self.num_required_velocity_qubits(manhattan_distance)
            / self.num_velocities_per_point
        )

    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        # Grid qubits
        grid_registers = [
            QuantumRegister(gp.bit_length(), name=f"g_{dimension_letter(c)}")
            for c, gp in enumerate(self.num_gridpoints)
        ]

        # Velocity qubits
        velocity_registers = [
            QuantumRegister(
                self.num_required_velocity_qubits(
                    self.num_timesteps,
                ),  # The number of velocity qubits required at time t
                name="v",
            )
        ]

        return (grid_registers, velocity_registers)

    def von_neumann_neighbor_indices(
        self, manhattan_distance_from_origin: int
    ) -> List[int]:
        """
        Get the indices of the neighbors up to a given Manhattan distance, in a von Neumann neighborhood structure.

        Parameters
        ----------
        manhattan_distance_from_origin : int
            The exact Manhattan distance from the origin for which to retrieve the indices.

        Returns
        -------
        List[int]
            The indices of the qubits encoding the velocity of neighboring gridpoints exactly ``manhattan_distance_from_origin`` away from the origin.
        """
        if manhattan_distance_from_origin == 0:
            return [0]

        total_neighbors = int(
            (
                self.num_required_velocity_qubits(manhattan_distance_from_origin)
                / self.num_velocities_per_point
            )
        )
        neighbors_lower_distance = int(
            (
                self.num_required_velocity_qubits(manhattan_distance_from_origin)
                / self.num_velocities_per_point
            )
        )

        return list(range(neighbors_lower_distance, total_neighbors))

    def get_neighbor_indices(
        self,
    ) -> Tuple[
        Dict[int, List[VonNeumannNeighbor]],
        Dict[int, Dict[int, List[VonNeumannNeighbor]]],
    ]:
        """
        Get all of the information encoding the neighborhood structure of the lattice grid.
        We differentiate between two kinds of grid points, based on their relative position as neighbors relative to the origin.
        `Extreme points` are grid points that, in the expanding Manhattan distance stencil, have 3 neighbors with higher Manhattan distances.
        All other points (except the origin) are `intermediate points`, which have 2 neighbors with higher Manhattan distances and 2 neighbors with lower ones.
        Both extreme and intermediate points are further broken down into `classes`, based on which side of the origin they are on.
        The classes follow the same numerical ordering as the local :math:`D_2Q_4` discretization:
        0 encodes right, 1 up, 2 left, 3 down.
        These differences are relevant when constructing the :class:`.SpaceTimeStreamingOperator`.
        The structure of the output of this function contains two dictionaries, one for each kind of neighbor:

        #. A dictionary containing extreme points. The keys of the dictionary are Manhattan distances, and the entries are the indices of the neighbors, ordered by class.
        #. A dictionary containing intermediate points. The keys of the dictionary are Manhattan distances, and the entries are themselves dictionaries. For each nested dictionary, the key is the class, and the value is a list consisting of the point indices belonging to that class.

        Returns
        -------
        Tuple[ Dict[int, List[VonNeumannNeighbor]], Dict[int, Dict[int, List[VonNeumannNeighbor]]], ]
            The information encoding the lattice neighborhood structure.
        """
        # ! This ONLY works for D2Q4!
        extreme_point_neighbor_indices: Dict[int, List[VonNeumannNeighbor]] = {}
        intermediate_point_neighbor_indices: Dict[
            int, Dict[int, List[VonNeumannNeighbor]]
        ] = {
            manhattan_distance: {}
            for manhattan_distance in range(2, self.num_timesteps + 1)
        }

        for manhattan_distance in range(1, self.num_timesteps + 1):
            total_neighbors_within_distance: int = self.num_gridpoints_within_distance(
                manhattan_distance
            )

            previous_extreme_point_neighbors: List[VonNeumannNeighbor] = (
                extreme_point_neighbor_indices[manhattan_distance - 1]
                if manhattan_distance > 1
                else (
                    [
                        VonNeumannNeighbor(
                            (0, 0),
                            1,
                            VonNeumannNeighborType.ORIGIN,
                        )
                        for _ in range(4)
                    ]
                )
            )

            extreme_point_neighbor_indices[manhattan_distance] = [
                VonNeumannNeighbor(
                    cast(
                        Tuple[int, int],
                        tuple(
                            c0 + c1
                            for c0, c1 in zip(
                                previous_extreme_point_neighbors[
                                    neighbor_class
                                ].coordinates_relative,
                                self.extreme_point_classes[neighbor_class][1],
                            )
                        ),
                    ),
                    extreme_point_index,
                    VonNeumannNeighborType.EXTREME,
                )
                for neighbor_class, extreme_point_index in enumerate(
                    range(
                        previous_extreme_point_neighbors[0].neighbor_index
                        + self.num_velocities_per_point * (manhattan_distance - 1),
                        total_neighbors_within_distance,
                        manhattan_distance,
                    )
                )
            ]

            if manhattan_distance < 2:
                continue

            intermediate_point_neighbor_indices[manhattan_distance] = {
                intermediate_point_class[0]: [
                    VonNeumannNeighbor(
                        cast(
                            Tuple[int, int],
                            tuple(
                                c0 + (relative_intermediate_point_index + 1) * c1
                                for c0, c1 in zip(
                                    extreme_point_neighbor_indices[manhattan_distance][
                                        neighbor_class
                                    ].coordinates_relative,
                                    self.intermediate_point_classes[neighbor_class][1],
                                )
                            ),
                        ),
                        absolute_intermediate_point_index,
                        VonNeumannNeighborType.INTERMEDIATE,
                    )
                    for relative_intermediate_point_index, absolute_intermediate_point_index in enumerate(
                        range(
                            extreme_point_neighbor_indices[manhattan_distance][
                                intermediate_point_class[0]
                            ].neighbor_index
                            + 1,
                            (
                                extreme_point_neighbor_indices[manhattan_distance][
                                    intermediate_point_class[0] + 1
                                ].neighbor_index
                                if neighbor_class
                                != len(self.intermediate_point_classes) - 1
                                else total_neighbors_within_distance
                            ),
                        )
                    )
                ]
                for neighbor_class, intermediate_point_class in enumerate(
                    self.intermediate_point_classes
                )
            }

        return extreme_point_neighbor_indices, intermediate_point_neighbor_indices

    def coordinates_to_quadrant(self, distance: Tuple[int, int]) -> int:
        """
        Maps a given point to the quadrant it belongs to.
        Quadrants are ordered in counterclockwise fashion, starting on the top right at 0:
         1 | 0
        _______
         2 | 3

        Points along lines that intersect with the origin (i.e., (0, 5), (-8, 0)) belong to quadrants as well:

        #. :math:`x>0,y=0 \to q_0`
        #. :math:`x=0,y>0 \to q_1`
        #. :math:`x=<0,y=0 \to q_2`
        #. :math:`x=0,y<0 \to q_3`

        Parameters
        ----------
        distance : Tuple[int, int]
            The coordinates of the point to place in a quadrant, expressed as its distance from the origin in the x and y axes.

        Returns
        -------
        int
            The quadrant the point belongs to.
        """

        if distance[1] == 0:
            if distance[0] > 0:
                return 0
            return 2

        if distance[0] == 0:
            if distance[1] > 0:
                return 1
            return 3

        if distance[1] > 0:
            if distance[0] > 0:
                return 0
            return 1

        if distance[0] < 0:
            return 2
        return 3

    def get_index_of_neighbor(self, distance: Tuple[int, int]) -> int:
        """
        Calculate the index of a given grid point within the von Neumann neighborhood.
        The indices are assigned in counterclockwise fashion in increasing order of their
        Manhattan distances from the origin, in the same way that velocity indices are typically
        labelled within LBM :math:`D_dQ_q` discretizations.
        This is helpful for determining the placements of :math:`SWAP` gates that perform streaming.

        Parameters
        ----------
        distance : Tuple[int, int]
            The coordinates of the point to place in a quadrant, expressed as its distance from the origin in the x and y axes.

        Returns
        -------
        int
            The index of the point.
        """
        if distance[0] == 0 and distance[1] == 0:
            return 0

        distance_from_origin = sum(map(abs, distance))
        is_extreme_point = distance[0] == 0 or distance[1] == 0
        num_points_lower_distances = self.num_gridpoints_within_distance(
            distance_from_origin - 1
        )
        quadrant: int = self.coordinates_to_quadrant(distance)

        if is_extreme_point:
            return num_points_lower_distances + quadrant * distance_from_origin
        else:
            ordering_dim = 0 if quadrant in [1, 3] else 1
            distance_across_ordering_dim = abs(distance[ordering_dim])
            return (
                num_points_lower_distances
                + quadrant * distance_from_origin
                + distance_across_ordering_dim
            )

    def get_streaming_lines(
        self, dimension: int, direction: bool, timestep: int | None = None
    ) -> List[List[int]]:
        """
        Returns the `stream line` for a given dimension, direction, and time step.
        A stream line consists of all particles within the point's neighborhood, that
        (1) have the same discrete velocity and (2) can "reach" each other
        by only traveling across the same discrete velocity channel.
        The streaming operator of the STQBM algorithm is sensitive to the order
        in which particles in such stream lines are swapped.

        Parameters
        ----------
        dimension : int
            The dimension (``0`` or ``1``) that the stream line spans.
        direction : bool
            The direction (``False`` for negative, ``True`` for positive) that the stream line traverses.
        timestep : int | None, optional
            The time step for which to compute the stream lines, by default None. More time steps require larger neighborhoods and thus more (and longer) stream lines.

        Returns
        -------
        List[List[int]]
            A list of neighbor indices that are along the same stream lines.
        """
        neighbors_in_line = []
        if timestep is None:
            timestep = self.num_timesteps
        for offset in range(-timestep + 1, timestep):
            start = -self.num_timesteps + abs(offset)
            end = self.num_timesteps - abs(offset)
            step = 1

            if (dimension == 0 and direction) or (dimension == 1 and not direction):
                start, end = end, start
                step *= -1

            neighbors_in_line.append(
                [
                    self.get_index_of_neighbor(
                        (
                            i if dimension == 0 else offset,
                            offset if dimension == 0 else i,
                        )
                    )
                    for i in range(start, end + step, step)
                ]
            )
        return neighbors_in_line

    def logger_name(self) -> str:
        gp_string = ""
        for c, gp in enumerate(self.num_gridpoints):
            gp_string += f"{gp+1}"
            if c < len(self.num_gridpoints) - 1:
                gp_string += "x"
        return f"{self.num_dims}d-{gp_string}-q4"
