"""Prepares the initial state for the :class:`.SpaceTimeQLBM` one gridpoint at a time."""

from logging import Logger, getLogger
from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, XGate
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.lattice.spacetime.properties_base import VonNeumannNeighbor
from qlbm.tools.utils import bit_value, flatten


class PointWiseSpaceTimeInitialConditions(LBMPrimitive):
    r"""Prepares the initial state for the :class:`.SpaceTimeQLBM`.

    Initial conditions are supplied in a ``List[Tuple[Tuple[int, int], Tuple[bool, bool, bool, bool]]]``
    containing, for each population to be initialized, two nested tuples.

    The first tuple position of the population(s) on the grid (i.e., ``(2, 5)``).
    The second tuple contains velocity of the population(s) at that location.
    Since the maximum number of velocities is pre-determined and the computational basis state encoding favors boolean logic,
    the input is provided as a tuple of ``boolean``\ s.
    That is, ``(True, True, False, False)`` would mean there are two populations at the same gridpoint,
    with velocities :math:`q_0` and :math:`q_1` according to the :math:`D_2Q_4` discretization.
    Together, the ``grid_data`` argument of the constructor can be supplied as, for instance, ``[((3, 7), (False, True, False, True))]``.

    The initialization follows the following steps:

    * For each (position, velocity) pair:
        #. Set the gird qubits encoding the position to :math:`\ket{1}^{\otimes n_g}` using :math:`X` gates;
        #. Set each of the toggled velocities to :math:`\ket{1}` by means of :math:`MCX` gates, controlled on the qubits set in the previous step;
        #. Undo the operation of step 1 (i.e., repeat the :math:`X` gates);
        #. Repeat steps 1-3 for all neighboring velocity qubits, adjusting for grid position and relative velocity index.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`grid_data`         The information encoding the particle probability distribution, formatted as (position, velocity) tuples.
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.spacetime.initial import PointWiseSpaceTimeInitialConditions
        from qlbm.lattice import SpaceTimeLattice

        # Build an example lattice
        lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": 4, "y": 8}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
            },
        )

        # Draw the initial conditions for two particles at (3, 7), traveling in the +y and -y directions
        PointWiseSpaceTimeInitialConditions(lattice=lattice, grid_data=[((3, 7), (False, True, False, True))]).draw("mpl")
    """

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        grid_data: List[Tuple[Tuple[int, ...], Tuple[bool, ...]]] = [
            ((2, 5), (True, True, True, True)),
            ((3, 4), (False, True, False, True)),
        ],
        filter_inside_blocks: bool = True,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)

        self.lattice = lattice
        self.grid_data = grid_data
        self.filter_inside_blocks = filter_inside_blocks

        self.circuit = self.create_circuit()

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()
        circuit.h(self.lattice.grid_index())

        # Set the state for the origin
        for grid_point_data in self.grid_data:
            if self.filter_inside_blocks:
                if self.lattice.is_inside_an_obstacle(grid_point_data[0]):
                    continue
            circuit.compose(self.set_grid_value(grid_point_data[0]), inplace=True)
            circuit.compose(
                MCMT(
                    XGate(),
                    num_ctrl_qubits=self.lattice.properties.get_num_grid_qubits(),
                    num_target_qubits=sum(
                        grid_point_data[1]
                    ),  # The sum is equal to the number of velocities set to true
                ),
                qubits=list(
                    self.lattice.grid_index()
                    + flatten(
                        [
                            self.lattice.velocity_index(0, c)
                            for c, is_velocity_enabled in enumerate(grid_point_data[1])
                            if is_velocity_enabled
                        ]
                    )
                ),
                inplace=True,
            )
            circuit.compose(self.set_grid_value(grid_point_data[0]), inplace=True)

            # Set the velocity state for neighbors in increasing velocity
            for manhattan_distance in range(1, self.lattice.num_timesteps + 1):
                for neighbor in self.lattice.extreme_point_indices[manhattan_distance]:
                    circuit.compose(
                        self.set_neighbor_velocity(
                            grid_point_data[0], grid_point_data[1], neighbor
                        ),
                        inplace=True,
                    )

                # No intermediate points at Manhattan distance 1
                # Or in 1D
                if manhattan_distance < 2 or self.lattice.num_dims < 2:
                    continue

                for neighbor in flatten(
                    list(
                        self.lattice.intermediate_point_indices[
                            manhattan_distance
                        ].values()
                    )
                ):
                    circuit.compose(
                        self.set_neighbor_velocity(
                            grid_point_data[0], grid_point_data[1], neighbor
                        ),
                        inplace=True,
                    )

        return circuit

    def set_grid_value(self, point_coordinates: Tuple[int, ...]) -> QuantumCircuit:
        r"""Applies :math:`X` gates to that convert the coordinates of a given gridpoint to :math:`\ket{1}^{\otimes n}`.

        Parameters
        ----------
        point_coordinates : Tuple[int, ...]
            The tuple encoding the gridpoint to convert to `\ket{1}^{\otimes n}`

        Returns
        -------
        QuantumCircuit
            The circuit that sets the gridpoint to the desired state.
        """
        # ! TODO: rename, refactor into primitive
        circuit = self.lattice.circuit.copy()

        for dim, num_gp in enumerate(self.lattice.num_gridpoints):
            for qubit_index in range(num_gp.bit_length()):
                if not bit_value(point_coordinates[dim], qubit_index):
                    circuit.x(self.lattice.grid_index(dim)[0] + qubit_index)

        return circuit

    def set_neighbor_velocity(
        self,
        point_coordinates: Tuple[int, ...],
        velocity_values: Tuple[bool, ...],
        neighbor: VonNeumannNeighbor,
    ) -> QuantumCircuit:
        """
        Sets the velocity state of the neighbor of a particular grid location to a given state.

        Parameters
        ----------
        point_coordinates : Tuple[int, ...]
            THe coordinates of the origin.
        velocity_values : Tuple[bool, ...]
            The velocity values to set, encoded as whether the velocity is enabled.
        neighbor : VonNeumannNeighbor
            The neighbor relative to the origin for which to set the velocity value.

        Returns
        -------
        QuantumCircuit
            The circuit that sets the velocity state.
        """
        # ! TODO: rename, refactor into primitive
        circuit = self.lattice.circuit.copy()
        absolute_neighbor_coordinates = neighbor.get_absolute_values(
            point_coordinates, relative=False
        )
        if self.filter_inside_blocks:
            if self.lattice.is_inside_an_obstacle(absolute_neighbor_coordinates):
                return self.lattice.circuit.copy()
        circuit.compose(
            self.set_grid_value(absolute_neighbor_coordinates), inplace=True
        )
        circuit.compose(
            MCMT(
                XGate(),
                num_ctrl_qubits=self.lattice.properties.get_num_grid_qubits(),
                num_target_qubits=sum(
                    velocity_values
                ),  # The sum is equal to the number of velocities set to true
            ),
            inplace=True,
            qubits=list(
                self.lattice.grid_index()
                + flatten(
                    [
                        self.lattice.velocity_index(neighbor.neighbor_index, c)
                        for c, is_velocity_enabled in enumerate(velocity_values)
                        if is_velocity_enabled
                    ]
                )
            ),
        )
        circuit.compose(
            self.set_grid_value(absolute_neighbor_coordinates), inplace=True
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive PointWiseSpaceTimeInitialConditions with data={self.grid_data} and lattice={self.lattice}]"
