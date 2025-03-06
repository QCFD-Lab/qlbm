"""Implementation of the :class:`.Lattice` base specific to the 2D and 3D :class:`.CQLBM` algorithm developed by :cite:t:`collisionless`."""

from logging import Logger, getLogger
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit, QuantumRegister
from typing_extensions import override

from qlbm.lattice.geometry.shapes.block import Block
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import dimension_letter, flatten

from .base import Lattice


class CollisionlessLattice(Lattice):
    r"""
    Implementation of the :class:`.Lattice` base specific to the 2D and 3D :class:`.CQLBM` algorithm developed by :cite:t:`collisionless`.

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
        * - :attr:`ancilla_velocity_register`
          - :math:`d`
          - :meth:`ancillae_velocity_index`
          - The qubits controlling the streaming operation based on the CFL counter.
        * - :attr:`ancilla_obstacle_register`
          - :math:`d` or :math:`1`, See :ref:`adaptability`.
          - :meth:`ancillae_obstacle_index`
          - The qubits used to detect whether particles have streamed into obstacles. Used for reflection.
        * - :attr:`ancilla_comparator_register`
          - :math:`2(d-1)`
          - :meth:`ancillae_comparator_index`
          - The qubits used to for :class:`.Comparator`\ s. Used for reflection.
        * - :attr:`grid_registers`
          - :math:`\Sigma_{1\leq j \leq d} \left \lceil{\log N_{g_j}} \right \rceil`
          - :meth:`grid_index`
          - The qubits encoding the physical grid.
        * - :attr:`velocity_registers`
          - :math:`\Sigma_{1\leq j \leq d} \left \lceil{\log N_{v_j}} \right \rceil - 1`
          - :meth:`velocity_index`
          - The qubits encoding speeds.
        * - :attr:`velocity_dir_registers`
          - :math:`d`
          - :meth:`velocity_dir_index`
          - The qubits encoding velocity direction (positive or negative).

    .. _adaptability:

    Adaptable Lattice Register
    **************************

    The :class:`.BounceBackReflectionOperator` and :class:`.SpecularReflectionOperator` have different requirements for the number of qubits.
    If a lattice contains at least one SR-conditioned object, then :math:`d` ancilla qubits are required
    to flag whether the particle has collided with the surface of the object, its edge (in 3D), or its corner.
    This information influences which directional qubits are inverted.

    The BB boundary conditions are simpler in that they only require :math:`1` ancilla qubit
    to detect whether a particle has collided with the object.
    All velocities are inverted, irrespective of the interaction with the object.
    As such, if only the lattice only contains BB objects, a single
    ancilla qubit is required for reflection across all objects.
    The lattice object infers this at construction time and adjusts the
    relative index of all other registers accordingly.


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

    The register setup can be visualized by constructing a lattice object:

    .. plot::
        :include-source:

        from qlbm.lattice import CollisionlessLattice

        CollisionlessLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
                "geometry": [{"shape":"cuboid", "x": [5, 6], "y": [1, 2], "boundary": "bounceback"}],
            }
        ).circuit.draw("mpl")
    """

    num_dims: int
    num_gridpoints: List[int]
    num_velocities: List[int]
    num_total_qubits: int
    registers: Tuple[QuantumRegister, ...]
    logger: Logger

    def __init__(
        self,
        lattice_data: str | Dict,  # type: ignore
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice_data, logger)
        dimensions, velocities, blocks = self.parse_input_data(lattice_data)  # type: ignore

        self.num_dims = len(dimensions)
        self.num_gridpoints = dimensions
        self.num_velocities = velocities
        self.blocks: Dict[str, List[Block]] = blocks
        self.block_list: List[Block] = flatten(list(blocks.values()))
        self.num_comparator_qubits = 2 * (self.num_dims - 1)
        self.num_obstacle_qubits = self.__num_obstacle_qubits()
        self.num_ancilla_qubits = (
            self.num_dims + self.num_comparator_qubits + self.num_obstacle_qubits
        )
        self.num_grid_qubits = sum([dim.bit_length() for dim in dimensions])
        self.num_velocity_qubits = sum([v.bit_length() for v in velocities])
        self.num_total_qubits = (
            self.num_ancilla_qubits + self.num_grid_qubits + self.num_velocity_qubits
        )

        temporary_registers = self.get_registers()
        (
            self.ancilla_velocity_register,
            self.ancilla_object_register,
            self.ancilla_comparator_register,
            self.grid_registers,
            self.velocity_registers,
            self.velocity_dir_registers,
        ) = temporary_registers
        self.registers = tuple(flatten(temporary_registers))
        self.circuit = QuantumCircuit(*self.registers)

        logger.info(self.__str__())

    def ancillae_velocity_index(self, dim: int | None = None) -> List[int]:
        """Get the indices of the qubits used as velocity ancillae for the specified dimension.

        Parameters
        ----------
        dim : int | None, optional
            The dimension of the grid for which to retrieve the velocity qubit indices, by default ``None``.
            When ``dim`` is ``None``, the indices of ancillae qubits for all dimensions are returned.

        Returns
        -------
        List[int]
            A list of indices of the qubits used as velocity ancillae for the given dimension.

        Raises
        ------
        LatticeException
            If the dimension does not exist.
        """
        if dim is None:
            return list(range(self.num_dims))

        if dim >= self.num_dims or dim < 0:
            raise LatticeException(
                f"Cannot index ancilla velocity register for dimension {dim} in {self.num_dims}-dimensional lattice."
            )

        # The velocity ancillas are on the first register, so no offset
        return [dim]

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
            return list(range(self.num_dims, self.num_dims + self.num_obstacle_qubits))

        if index >= self.num_obstacle_qubits or index < 0:
            raise LatticeException(
                f"Cannot index ancilla obstacle register for index {index}. Maximum index for this lattice is {self.num_obstacle_qubits - 1}."
            )

        # There are `d` ancillae velocity qubits "ahead" of this register
        return [self.num_dims + index]

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
        # Ahead of this register
        # `d` ancillae velocity qubits
        # `num_obstacle_qubits` ancillae obstacle qubits
        # 2 * `d` ancillae comparator qubits for "lower" dimensions
        # These are ordered as follows: lx, ux, ly, uy, lz, uz
        if index is None:
            return list(
                range(
                    self.num_dims + self.num_obstacle_qubits,
                    self.num_dims + self.num_obstacle_qubits + 2 * (self.num_dims - 1),
                )
            )

        if index >= self.num_dims - 1 or index < 0:
            raise LatticeException(
                f"Cannot index ancilla comparator register for index {index} in {self.num_dims}-dimensional lattice. Maximum is {self.num_dims - 2}."
            )
        previous_qubits = self.num_dims + self.num_obstacle_qubits + 2 * index
        return list(range(previous_qubits, previous_qubits + 2))

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
                    self.num_ancilla_qubits,
                    self.num_ancilla_qubits + self.num_grid_qubits,
                )
            )

        if dim >= self.num_dims or dim < 0:
            raise LatticeException(
                f"Cannot index grid register for dimension {dim} in {self.num_dims}-dimensional lattice."
            )

        # Ahead of this register are
        # 4 * `d` - 2 ancillae qubits
        # log2(ng_i) qubits for "lower" dimensions
        previous_qubits = self.num_ancilla_qubits + sum(
            [self.num_gridpoints[d].bit_length() for d in range(dim)]
        )

        return list(
            range(
                previous_qubits, previous_qubits + self.num_gridpoints[dim].bit_length()
            )
        )

    def velocity_index(self, dim: int | None = None) -> List[int]:
        """Get the indices of the qubits used that encode the velocity magnitude values for the specified dimension.

        Parameters
        ----------
        dim : int | None, optional
            The dimension of the grid for which to retrieve the velocity qubit indices, by default ``None``.
            When ``dim`` is ``None``, the indices of all velocity magnitude qubits for all dimensions are returned.

        Returns
        -------
        List[int]
            A list of indices of the qubits used to encode the velocity magnitude values for the given dimension.

        Raises
        ------
        LatticeException
            If the dimension does not exist.
        """
        if dim is None:
            return list(
                range(
                    self.num_ancilla_qubits + self.num_grid_qubits,
                    self.num_ancilla_qubits
                    + self.num_grid_qubits
                    + sum(
                        self.num_velocities[d].bit_length() - 1
                        for d in range(self.num_dims)
                    ),
                )
            )

        if dim >= self.num_dims or dim < 0:
            raise LatticeException(
                f"Cannot index velocity register for dimension {dim} in {self.num_dims}-dimensional lattice."
            )

        # Ahead of this register are
        # 4 * `d` - 2 ancillae qubits
        # The log2(ng_i) qubits encoding the grid points in each dimension
        # The log2(nv_i) - 1 qubits encoding the velocity magnitudes in lower dimensions
        previous_qubits = (
            self.num_ancilla_qubits
            + self.num_grid_qubits
            + sum(self.num_velocities[d].bit_length() - 1 for d in range(dim))
        )

        # Subtract one from the number of qubits required to
        # Encode the velocity discretization due to the directional
        # Qubit that resides in a different register
        return list(
            range(
                previous_qubits,
                previous_qubits + self.num_velocities[dim].bit_length() - 1,
            )
        )

    def velocity_dir_index(self, dim: int | None = None) -> List[int]:
        """Get the indices of the qubit that encodes the velocity direction values for the specified dimension.

        Parameters
        ----------
        dim : int | None, optional
            The dimension of the grid for which to retrieve the velocity direction qubit index, by default ``None``.
            When ``dim`` is ``None``, the indices of all velocity direction qubits for all dimensions are returned.

        Returns
        -------
        List[int]
            A list of indices of the qubits used to encode the velocity direction for the given dimension.

        Raises
        ------
        LatticeException
            If the dimension does not exist.
        """
        if dim is None:
            return list(
                range(
                    self.num_ancilla_qubits
                    + self.num_grid_qubits
                    + self.num_velocity_qubits
                    - self.num_dims,
                    self.num_total_qubits,
                )
            )

        if dim >= self.num_dims or dim < 0:
            raise LatticeException(
                f"Cannot index velocity direction register for dimension {dim} in {self.num_dims}-dimensional lattice."
            )

        # Ahead of this register are
        # 4 * `d` - 2 ancillae qubits
        # The log2(ng_i) qubits encoding the grid points in each dimension
        # The log2(nv_i) - 1 qubits encoding the non-directional velocity magnitudes in each dimension
        previous_qubits = (
            self.num_ancilla_qubits
            + sum(self.num_gridpoints[d].bit_length() for d in range(self.num_dims))
            + sum(v.bit_length() - 1 for v in self.num_velocities)
        )

        return [previous_qubits + dim]

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
        # d ancilla qubits tracking whether a velocity is to be streamed
        ancilla_vel_register = [QuantumRegister(self.num_dims, name="a_v")]

        # d ancilla qubits used to conditionally reflect velocities
        ancilla_object_register = [
            QuantumRegister(self.num_obstacle_qubits, name="a_o")
        ]

        # 2(d-1) ancilla qubits
        ancilla_comparator_register = [
            QuantumRegister(self.num_comparator_qubits, name="a_c")
        ]

        # d qubits encoding the velocity direction
        velocity_dir_register = [
            QuantumRegister(1, name=f"v_dir_{dimension_letter(c)}")
            for c in range(len(self.num_velocities))
        ]

        # Velocity qubits
        velocity_registers = [
            QuantumRegister(v.bit_length() - 1, name=f"v_{dimension_letter(c)}")
            for c, v in enumerate(self.num_velocities)
        ]

        # Grid qubits
        grid_registers = [
            QuantumRegister(gp.bit_length(), name=f"g_{dimension_letter(c)}")
            for c, gp in enumerate(self.num_gridpoints)
        ]

        return (
            ancilla_vel_register,
            ancilla_object_register,
            ancilla_comparator_register,
            grid_registers,
            velocity_registers,
            velocity_dir_register,
        )

    def __num_obstacle_qubits(self) -> int:
        all_obstacle_bounceback: bool = len(
            [b for b in self.block_list if b.boundary_condition == "bounceback"]
        ) == len(self.block_list)
        if all_obstacle_bounceback:
            # A single qubit suffices to determine
            # Whether particles have streamed inside the object
            return 1
        # If there is at least one object with specular reflection
        # 2 ancilla qubits are requried for velocity inversion
        else:
            return self.num_dims

    @override
    def __str__(self) -> str:
        return f"[Lattice with {self.num_gridpoints} gps, {self.num_velocities} vels, and {str(self.blocks)} blocks with {self.num_total_qubits} qubits]"

    @override
    def logger_name(self) -> str:
        gp_string = ""
        for c, gp in enumerate(self.num_gridpoints):
            gp_string += f"{gp+1}"
            if c < len(self.num_gridpoints) - 1:
                gp_string += "x"
        return f"{self.num_dims}d-{gp_string}-{len(self.block_list)}-obstacle"
