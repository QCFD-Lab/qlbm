"""Implementation of the Amplitude-Based (AB) encoding lattice for generic DdQq discretizations."""

from logging import getLogger
from typing import Dict, List, Tuple

from numpy import ceil, log2
from qiskit import QuantumCircuit, QuantumRegister
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.lattice.geometry.shapes.base import Shape
from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import dimension_letter, flatten, is_two_pow

from .base import AmplitudeLattice


class ABLattice(AmplitudeLattice):
    r"""
    Implementation of the :class:`.Lattice` base specific to the 2D and 3D :class:`.ABQLBM` algorithm developed in :cite:t:`collisionless`.

    This lattice is only built from :math:`D_dQ_q` specifications.
    For multi-speed implementations, see :class:`.MSQLBM`.

    The registers encoded in the lattice and their accessors are given below.
    For the size of each register,
    :math:`N_{g_j}` is the number of grid points of dimension :math:`j` (i.e., 64, 128),
    :math:`q` is the number of discrete velocities, for instance, 9.

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
          - :math:`\lceil\log_2 q \rceil`
          - :meth:`velocity_index`
          - The qubits encoding the :math:`q` discrete velocities.
        * - :attr:`ancilla_obstacle_register`
          - :math:`1`
          - :meth:`ancillae_obstacle_index`
          - The qubits used to detect whether particles have streamed into obstacles. Used for reflection.
        * - :attr:`ancilla_comparator_register`
          - :math:`2(d-1)`
          - :meth:`ancillae_comparator_index`
          - The qubits used to for :class:`.Comparator`\ s. Used for reflection.

    A lattice can be constructed from from either an input file or a Python dictionary:

    .. code-block:: json

        {
            "lattice": {
                "dim": {
                    "x": 16,
                    "y": 16
                },
                "velocities": "d2q9"
            },
            "geometry": [
                {
                    "x": [9, 12],
                    "y": [3, 6],
                    "boundary": "bounceback"
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

        from qlbm.lattice import ABLattice

        ABLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "D2Q9"},
                "geometry": [],
            }
        ).circuit.draw("mpl")
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
        self.num_velocity_qubits = int(ceil(log2(self.num_velocities_per_point)))
        self.num_base_qubits = self.num_grid_qubits + self.num_velocity_qubits

        self.num_obstacle_qubits = self.__num_obstacle_qubits()
        self.num_comparator_qubits = 2 * (self.num_dims - 1)
        self.num_ancilla_qubits = self.num_comparator_qubits + self.num_obstacle_qubits

        self.num_marker_qubits = (
            int(ceil(log2(len(self.geometries))))
            if self.has_multiple_geometries()
            else 0
        )

        self.num_total_qubits = (
            self.num_base_qubits + self.num_ancilla_qubits + self.num_marker_qubits
        )

        self.num_accumulation_qubits = 0

        self.__update_registers()

    def __update_registers(self):
        self.num_total_qubits = (
            self.num_base_qubits + self.num_ancilla_qubits + self.num_marker_qubits
        )

        temp_registers = self.get_registers()

        (
            self.grid_registers,
            self.velocity_registers,
            self.ancilla_comparator_register,
            self.ancilla_object_register,
            self.marker_register,
            self.accumulation_register,
        ) = temp_registers

        self.registers = tuple(flatten(temp_registers))

        self.circuit = QuantumCircuit(*self.registers)

    def set_geometries(self, geometries):
        """
        Updates the geometry setup of the lattice.

        For a given lattice (set number of gridpoints and velocity discretization),
        set multiple geometry configurations to simulate simultaneously.

        .. plot::
            :include-source:

            from qlbm.lattice import ABLattice

            lattice = ABLattice(
                {
                    "lattice": {
                        "dim": {"x": 16, "y": 16},
                        "velocities": "D2Q9",
                    },
                },
            )

            lattice.circuit.draw("mpl")

        Parameters
        ----------
        geometries : Dict
            A list of geometries to simulate on the same lattice.
        """
        self.geometries = [self.parse_geometry_dict(g) for g in geometries]
        if len(self.geometries) == 1:
            # Remove this in the future...
            self.shapes = self.geometries[0]

        self.num_marker_qubits = (
            int(ceil(log2(len(self.geometries))))
            if self.has_multiple_geometries()
            else 0
        )
        self.__update_registers()

    @override
    def grid_index(self, dim: int | None = None) -> List[int]:
        if dim is None:
            return list(
                range(
                    self.num_grid_qubits,
                )
            )

        if dim >= self.num_dims or dim < 0:
            raise LatticeException(
                f"Cannot index grid register for dimension {dim} in {self.num_dims}-dimensional lattice."
            )

        previous_qubits = sum([self.num_gridpoints[d].bit_length() for d in range(dim)])

        return list(
            range(
                previous_qubits, previous_qubits + self.num_gridpoints[dim].bit_length()
            )
        )

    @override
    def velocity_index(self, dim: int | None = None) -> List[int]:
        if dim is not None:
            raise LatticeException(
                "ABLattice does not support a dimensional breakdown of velocities."
            )
        return list(
            range(
                self.num_grid_qubits,
                self.num_grid_qubits + self.num_velocity_qubits,
            )
        )

    @override
    def ancillae_comparator_index(self, index: int | None = None) -> List[int]:
        if index is None:
            return list(
                range(
                    self.num_base_qubits,
                    self.num_base_qubits + 2 * (self.num_dims - 1),
                )
            )

        if index >= self.num_dims - 1 or index < 0:
            raise LatticeException(
                f"Cannot index ancilla comparator register for index {index} in {self.num_dims}-dimensional lattice. Maximum is {self.num_dims - 2}."
            )

        return list(
            range(
                self.num_base_qubits, self.num_base_qubits + self.num_comparator_qubits
            )
        )

    @override
    def ancillae_obstacle_index(self, index: int | None = None) -> List[int]:
        if index is None:
            return list(
                range(
                    self.num_base_qubits + self.num_comparator_qubits,
                    self.num_base_qubits
                    + self.num_comparator_qubits
                    + self.num_obstacle_qubits,
                )
            )

        if index >= self.num_obstacle_qubits or index < 0:
            raise LatticeException(
                f"Cannot index ancilla obstacle register for index {index}. Maximum index for this lattice is {self.num_obstacle_qubits - 1}."
            )

        return [self.num_base_qubits + self.num_comparator_qubits + index]

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

    @override
    def get_registers(self) -> Tuple[List[QuantumRegister], ...]:
        """Generates the encoding-specific register required for the streaming step.

        For this encoding, different registers encode
        (i) the logarithmically compressed grid,
        (ii) the logarithmically compressed discrete velocities,
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

        marker_register = (
            [
                QuantumRegister(
                    int(ceil(log2(len(self.geometries)))),
                    name="m",
                )
            ]
            if self.has_multiple_geometries()
            else []
        )

        accumulation_register = (
            [QuantumRegister(self.num_accumulation_qubits, name="acc")]
            if self.has_accumulation_register()
            else []
        )

        return (
            grid_registers,
            velocity_registers,
            ancilla_comparator_register,
            ancilla_object_register,
            marker_register,
            accumulation_register,
        )

    @override
    def logger_name(self) -> str:
        gp_string = ""
        for c, gp in enumerate(self.num_gridpoints):
            gp_string += f"{gp + 1}"
            if c < len(self.num_gridpoints) - 1:
                gp_string += "x"
        return f"ablattice-{self.num_dims}d-{gp_string}-{len(flatten(list(self.shapes.values())))}-obstacle"

    @override
    def has_multiple_geometries(self) -> bool:
        return len(self.geometries) > 1

    def has_accumulation_register(self) -> bool:
        """
        Whether the lattice has a register that accumulates quantities at each step.

        Returns
        -------
        bool
            Whether the lattice has a register that accumulates quantities at each step.
        """
        return self.num_accumulation_qubits > 0

    def use_accumulation_register(self):
        """
        Sets up the accumulation register of the lattice.

        The amplitude-based accumulation method is only currently supported for 1 time step,
        at the end of the simulation. More detail on amplitude accumulation can be found
        in :cite:t:`qsearch`.
        """
        self.num_accumulation_qubits = 1

        self.__update_registers()

    @override
    def marker_index(self) -> List[int]:
        return list(
            range(
                self.num_base_qubits + self.num_ancilla_qubits,
                self.num_base_qubits + self.num_ancilla_qubits + self.num_marker_qubits,
            )
        )

    @override
    def accumulation_index(self) -> List[int]:
        return list(
            range(
                self.num_base_qubits + self.num_ancilla_qubits + self.num_marker_qubits,
                self.num_base_qubits
                + self.num_ancilla_qubits
                + self.num_marker_qubits
                + self.num_accumulation_qubits,
            )
        )

    @override
    def get_encoding(self) -> ABEncodingType:
        return ABEncodingType.AB

    @override
    def get_base_circuit(self):
        return QuantumCircuit(
            *flatten(
                [
                    self.grid_registers,
                    self.velocity_registers,
                    self.ancilla_comparator_register,
                    self.ancilla_object_register,
                ]
            ),
        )
