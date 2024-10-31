from logging import Logger, getLogger
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit, QuantumRegister

from qlbm.lattice.lattices.base import Lattice
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import flatten

from .builder_base import (
    SpaceTimeLatticeBuilder,
)
from .d1q2 import D1Q2SpaceTimeLatticeBuilder
from .d2q4 import D2Q4SpaceTimeLatticeBuilder


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
                                specified in the input dictionary is one larger than the one held in the ``Lattice``.
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

    def __init__(
        self,
        num_timesteps: int,
        lattice_data: str | Dict,  # type: ignore
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(lattice_data, logger)
        self.num_gridpoints, self.num_velocities, self.blocks = self.parse_input_data(
            lattice_data
        )  # type: ignore
        self.num_dims = len(self.num_gridpoints)
        self.num_timesteps = num_timesteps

        self.properties: SpaceTimeLatticeBuilder = self.__get_builder()

        self.num_velocities_per_point = self.properties.get_num_velocities_per_point()

        temporary_registers = self.properties.get_registers()
        (
            self.grid_registers,
            self.velocity_registers,
        ) = temporary_registers
        self.registers = tuple(flatten(temporary_registers))
        self.circuit = QuantumCircuit(*self.registers)
        (
            self.extreme_point_indices,
            self.intermediate_point_indices,
        ) = self.properties.get_neighbor_indices()

        logger.info(self.__str__())

    def __get_builder(self) -> SpaceTimeLatticeBuilder:
        if self.num_dims == 1:
            if self.num_velocities[0] == 1:
                return D1Q2SpaceTimeLatticeBuilder(
                    self.num_timesteps, self.num_gridpoints, self.blocks, self.logger
                )
            raise LatticeException(
                f"Unsupported number of velocities for 1D: {self.num_velocities[0]}. Only D1Q2 is supported at the moment."
            )

        if self.num_dims == 2:
            if self.num_velocities[0] == 1 and self.num_velocities[1] == 1:
                return D2Q4SpaceTimeLatticeBuilder(
                    self.num_timesteps, self.num_gridpoints, self.blocks, self.logger
                )
            raise LatticeException(
                f"Unsupported number of velocities for 2D: {self.num_velocities}. Only D2Q4 is supported at the moment."
            )

        raise LatticeException(
            "Only 1D and 2D discretizations are currently available."
        )

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
            return list(range(self.properties.get_num_grid_qubits()))

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
                    self.properties.get_num_grid_qubits()
                    + point_neighborhood_index * self.num_velocities_per_point,
                    self.properties.get_num_grid_qubits()
                    + (point_neighborhood_index + 1) * (self.num_velocities_per_point),
                )
            )
        return [
            self.properties.get_num_grid_qubits()
            + point_neighborhood_index * self.num_velocities_per_point
            + velocity_direction
        ]

    def __grid_neighbors(
        self, coordinates: Tuple[int, int], up_to_distance: int
    ) -> List[List[int]]:
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

    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        return self.properties.get_registers()

    def logger_name(self) -> str:
        gp_string = ""
        for c, gp in enumerate(self.num_gridpoints):
            gp_string += f"{gp+1}"
            if c < len(self.num_gridpoints) - 1:
                gp_string += "x"
        return f"{self.num_dims}d-{gp_string}-q4"
