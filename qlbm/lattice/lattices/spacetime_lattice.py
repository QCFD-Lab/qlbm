"""Implementation of the :class:`.Lattice` base specific to the 2D and 3D :class:`.SpaceTimeQLBM` algorithm developed by :cite:t:`spacetime`."""

from logging import Logger, getLogger
from typing import Dict, List, Tuple, cast

from qiskit import QuantumCircuit, QuantumRegister
from typing_extensions import override

from qlbm.lattice.lattices.base import Lattice
from qlbm.lattice.spacetime.d1q2 import D1Q2SpaceTimeLatticeBuilder
from qlbm.lattice.spacetime.d2q4 import D2Q4SpaceTimeLatticeBuilder
from qlbm.lattice.spacetime.properties_base import SpaceTimeLatticeBuilder
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import flatten


class SpaceTimeLattice(Lattice):
    r"""
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
          - :math:`\Sigma_{1\leq j \leq d} \left \lceil{\log N_{g_j}} \right \rceil`
          - :meth:`grid_index`
          - The qubits encoding the physical grid.
        * - :attr:`velocity_registers`
          - :math:`\min(N_g \cdot N_v, \frac{N_v^2\cdot N_t \cdot (N_t + 1)}{2} + N_v)`
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
        filter_inside_blocks: bool = True,
        include_measurement_qubit: bool = False,
        use_volumetric_ops: bool = False,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(lattice_data, logger)
        self.filter_inside_blocks = filter_inside_blocks
        self.include_measurement_qubit = include_measurement_qubit
        self.use_volumetric_ops = use_volumetric_ops

        self.num_gridpoints, self.num_velocities, self.blocks = self.parse_input_data(
            lattice_data
        )  # type: ignore
        self.num_dims = len(self.num_gridpoints)
        self.num_timesteps = num_timesteps

        self.properties: SpaceTimeLatticeBuilder = self.__get_builder()
        self.num_total_qubits = (
            self.properties.get_num_grid_qubits()
            + self.properties.get_num_velocity_qubits()
            + self.properties.get_num_ancilla_qubits()
        )

        self.num_velocities_per_point = self.properties.get_num_velocities_per_point()

        temporary_registers = self.properties.get_registers()
        (self.grid_registers, self.velocity_registers, self.ancilla_registers) = (
            temporary_registers
        )
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
                    self.num_timesteps,
                    self.num_gridpoints,
                    include_measurement_qubit=self.include_measurement_qubit,
                    use_volumetric_ops=self.use_volumetric_ops,
                    logger=self.logger,
                )
            raise LatticeException(
                f"Unsupported number of velocities for 1D: {self.num_velocities[0] + 1}. Only D1Q2 is supported at the moment."
            )

        if self.num_dims == 2:
            if self.num_velocities[0] == 1 and self.num_velocities[1] == 1:
                return D2Q4SpaceTimeLatticeBuilder(
                    self.num_timesteps,
                    self.num_gridpoints,
                    include_measurement_qubit=self.include_measurement_qubit,
                    use_volumetric_ops=self.use_volumetric_ops,
                    logger=self.logger,
                )
            raise LatticeException(
                f"Unsupported number of velocities for 2D: {(self.num_velocities[0] + 1, self.num_velocities[1] + 1)}. Only D2Q4 is supported at the moment."
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

    def ancilla_mass_index(self) -> List[int]:
        """
        Get the index of the qubit used as the mass measurement ancilla.

        Returns
        -------
        List[int]
            The index of the mass measurement qubit.

        Raises
        ------
        LatticeException
            If the mass measurement qubit is toggled off.
        """
        if not self.include_measurement_qubit:
            raise LatticeException(
                "Lattice contains no mass ancilla qubits. To enable the mass ancilla qubit, construct the Lattice with include_measurement_qubit=True."
            )

        # Ahead of this register
        # All grid qubits
        # All velocity qubits
        return [
            self.properties.get_num_grid_qubits()
            + self.properties.get_num_velocity_qubits()
        ]

    def ancilla_comparator_index(self, index: int | None = None) -> List[int]:
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
            If the dimension does not exist or if the lattice is set up such that it contains no ancilla qubits for volumetric operations.
        """
        if not self.use_volumetric_ops:
            raise LatticeException(
                "Lattice contains no comparator ancilla qubits. To enable comparator (volumetric) operations, construct the Lattice with use_volumetric_ops=True."
            )
        # Ahead of this register
        # All grid qubits
        # All velocity qubits
        # `1` ancilla mass qubit (if toggled)
        # 2 * `d` ancillae comparator qubits for "lower" dimensions
        # These are ordered as follows: lx, ux, ly, uy, lz, uz
        start_index = (
            self.properties.get_num_grid_qubits()
            + self.properties.get_num_velocity_qubits()
            + (1 if self.include_measurement_qubit else 0)
        )

        if index is None:
            return list(
                range(
                    start_index,
                    start_index + 2 * self.num_dims,
                )
            )

        if index >= self.num_dims or index < 0:
            raise LatticeException(
                f"Cannot index ancilla comparator register for index {index} in {self.num_dims}-dimensional lattice. Maximum is {self.num_dims - 1}."
            )
        previous_qubits = start_index + 2 * index
        return list(range(previous_qubits, previous_qubits + 2))

    def volumetric_ancilla_qubit_combinations(
        self, overflow_occurred: List[bool]
    ) -> List[List[int]]:
        """
        Get all combinations of ancilla qubit indices required for volumetric operations.

        Volumetric operations perform actions on contiguous volumes of space in the lattice.
        These volumes are defined by lower and upper bounds in each dimension.
        Since the locality of the data structure may be affected by periodic boundary conditions,
        volumetrics must be adjusted to account for all possible overflow scenarios.
        This is done by performing the operations in different orders on the adjusted bounds.
        This method returns the sequence of ancilla qubit indices required to perform the operations soundly.

        Parameters
        ----------
        overflow_occurred : List[bool]
            A :math:`d`-length list of booleans indicating whether overflow occurred in each dimension.

        Returns
        -------
        List[List[int]]
            The sequence of ancilla qubit indices required to perform the volumetric operations.

        Raises
        ------
        LatticeException
            If volumetric operations are not enabled in the lattice.
        """
        if not self.use_volumetric_ops:
            raise LatticeException(
                "Lattice contains no comparator ancilla qubits. To enable comparator (volumetric) operations, construct the Lattice with use_volumetric_ops=True."
            )

        sequences: List[List[int]] = [[]]
        for dim, overflow in enumerate(overflow_occurred):
            if not overflow:
                sequences = [
                    seq + self.ancilla_comparator_index(dim) for seq in sequences
                ]
            else:
                sequences = sequences * 2
                sequences = [
                    seq
                    + [self.ancilla_comparator_index(dim)[c >= (len(sequences) // 2)]]
                    for c, seq in enumerate(sequences)
                ]

        return sequences

    @override
    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        return self.properties.get_registers()

    def is_inside_an_obstacle(self, gridpoint: Tuple[int, ...]) -> bool:
        """
        Whether a gridpoint is inside the volume of any obstacle in the lattice.

        Parameters
        ----------
        gridpoint : Tuple[int, ...]
            The :math:`d`-dimensional gridpoint to check.

        Returns
        -------
        bool
            Whether the gridpoint is inside any obstacle.
        """
        return any(
            block.contains_gridpoint(gridpoint)
            for block in flatten(self.blocks.values())
        )

    @override
    def logger_name(self) -> str:
        gp_string = ""
        for c, gp in enumerate(self.num_gridpoints):
            gp_string += f"{gp + 1}"
            if c < len(self.num_gridpoints) - 1:
                gp_string += "x"
        return f"{self.num_dims}d-{gp_string}-q4"

    def comparator_periodic_volume_bounds(
        self, bounds: List[Tuple[int, int]]
    ) -> List[Tuple[Tuple[int, int], Tuple[bool, bool]]]:
        r"""
        Computes the lower and upper bounds for the :class:`.Comparator`\ s used to perform volumetric operations in the :class:`.SpaceTimeQLBM`.

        For any given lower and upper bounds in 1, 2, or 3 dimensions,
        modulo operations are applied that detect whether periodic boundary conditions
        are required.
        If that is the case, the directions in which the bounds overflow
        becomes the opposite kind of bound.
        For instance, a :math:`-2 \leq x \leq 7` interval that would require
        a :math:`-2 \leq x \leq 7` on a :math:`16 \\times 16` would require a :math:`\geq 2` comparator
        and a :math:`\leq 7` comparator.
        Since :math:`-2` is not part of the domain, it gets mapped to :math:`14`,
        and a different operation, based on two :math:`\geq` comparators is required.

        Parameters
        ----------
        bounds : List[Tuple[int, int]]
            The absolute bounds of the volume in each dimensions.

        Returns
        -------
        List[Tuple[Tuple[int, int], Tuple[bool, bool]]]
            The bounds adjusted for periodicity and whether overflow occurs for each bound.
        """
        adjusted_bounds: List[Tuple[Tuple[int, int], Tuple[bool, bool]]] = []

        for dim, bs in enumerate(bounds):
            new_bounds: Tuple[int, int] = cast(
                Tuple[int, int],
                tuple(
                    (
                        b % (self.num_gridpoints[dim] + 1)
                        if b > self.num_gridpoints[dim]
                        else (b + self.num_gridpoints[dim] + 1 if b < 0 else b)
                    )
                    for b in bs
                ),
            )
            overflow_occurred: Tuple[bool, bool] = cast(
                Tuple[bool, bool], tuple((x != y for x, y in zip(new_bounds, bs)))
            )
            adjusted_bounds.append((new_bounds, overflow_occurred))

        return adjusted_bounds
