"""Quantum circuits used for reflection in the :class:`ABQLBM` algorithm."""

from itertools import product
from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, XGate
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.ab.streaming import ABStreamingOperator
from qlbm.components.base import LBMOperator, LBMPrimitive
from qlbm.components.ms.specular_reflection import SpecularWallComparator
from qlbm.lattice.geometry.encodings.ms import ReflectionPoint
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.lattices.base import AmplitudeLattice
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import flatten, get_qubits_to_invert


class ABReflectionOperator(LBMOperator):
    """
    Implements bounceback reflection in the amplitude-based encoding of :class:`.ABQLBM` for :math:`D_dQ_q` discretizations.

    Example usage:

    .. code-block:: python

        from qlbm.components.ab import ABReflectionOperator
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
                "geometry": [
                    {
                        "shape": "cuboid",
                        "x": [1, 3],
                        "y": [1, 3],
                        "boundary": "bounceback",
                    }
                ],
            }
        )

        # Drawing the circuit might take a while
        ABReflectionOperator(lattice, blocks=lattice.shapes["bounceback"])

    """

    lattice: AmplitudeLattice

    def __init__(
        self,
        lattice: ABLattice,
        blocks: List[Block],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)

        self.blocks = blocks

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        if self.lattice.discretization == LatticeDiscretization.D2Q9:
            return self.__create_circuit_d2q9()

        raise LatticeException("AB reflection only currently supported in D2Q9")

    def __create_circuit_d2q9(self):
        circuit = self.lattice.circuit.copy()

        # Mark populations inside the object
        for block in self.blocks:
            circuit.compose(self.set_inside_wall_ancilla_state(block), inplace=True)

        circuit.compose(
            self.set_ancilla_of_point_state(
                flatten(
                    [[(p, None) for p in block.corners_inside] for block in self.blocks]
                ),
                ignore_velocity_data=True,
            ),
            inplace=True,
        )

        # Stream the particles back
        circuit.compose(self.permute_and_stream(), inplace=True)

        # Reset the ancilla state of reflected populations
        for block in self.blocks:
            circuit.compose(self.reset_outside_wall_ancilla_state(block), inplace=True)

        # Re-reset near corner point ancillas
        point_data: List[Tuple[ReflectionPoint, List[int]]] = []

        for block in self.blocks:
            for dim in range(self.lattice.num_dims):
                for c, bounds in enumerate(
                    product(*[[False, True]] * self.lattice.num_dims)
                ):
                    point_data.append(
                        (
                            block.near_corner_points_2d[dim * 4 + c],
                            block.get_lbm_near_corner_velocity_indices_to_reflect(
                                self.lattice.discretization, dim, bounds
                            ),
                        )
                    )
            for c, bounds in enumerate(
                product(*[[False, True]] * self.lattice.num_dims)
            ):
                point_data.append(
                    (
                        block.corners_outside[c],
                        block.get_lbm_outside_corner_indices_to_reflect(
                            self.lattice.discretization, bounds
                        ),
                    )
                )
        # Re-reset the ancilla state of the populations that
        # Shouldn't have been flipped in the previous step
        circuit.compose(
            self.set_ancilla_of_point_state(point_data, ignore_velocity_data=False),
            inplace=True,
        )

        return circuit

    def set_inside_wall_ancilla_state(self, block: Block) -> QuantumCircuit:
        """
        Sets the state of the ancilla qubit for all the gridpoints lying inside the walls of the block.

        This is done using the :class:`.SpecularWallComparator` originally designed
        for the :class:`MSQLBM` algorithm.
        The inside corner points are not addressed.

        Parameters
        ----------
        block : Block
            The solid object to address.

        Returns
        -------
        QuantumCircuit
            The circuit that sets the appropriate state on the object ancilla qubit.
        """
        circuit = self.lattice.circuit.copy()

        for dim in range(self.lattice.num_dims):
            for wall in block.walls_inside[dim]:
                comparator_circuit = SpecularWallComparator(
                    self.lattice, wall, self.logger
                ).circuit

                grid_qubit_indices_to_invert = [
                    self.lattice.grid_index(0)[0] + qubit
                    for qubit in wall.data.qubits_to_invert
                ]

                circuit.compose(comparator_circuit, inplace=True)

                if grid_qubit_indices_to_invert:
                    circuit.x(grid_qubit_indices_to_invert)

                control_qubits = (
                    self.lattice.grid_index(wall.dim)
                    + self.lattice.ancillae_comparator_index()
                )

                target_qubits = self.lattice.ancillae_obstacle_index(0)

                circuit.compose(
                    MCMTGate(
                        XGate(),
                        len(control_qubits),
                        len(target_qubits),
                    ),
                    qubits=control_qubits + target_qubits,
                    inplace=True,
                )

                if grid_qubit_indices_to_invert:
                    circuit.x(grid_qubit_indices_to_invert)

                circuit.compose(comparator_circuit, inplace=True)

        return circuit

    def reset_outside_wall_ancilla_state(self, block: Block) -> QuantumCircuit:
        """
        Resets the state of the obstacle ancilla qubit for all the gridpoints that are directly adjacent to the object, but in the fluid domain.

        This is done using the :class:`.SpecularWallComparator` originally designed
        for the :class:`MSQLBM` algorithm. The state of obstacle ancilla of the
        the near-corner gridpoints will be incorrect following the application of this primitive
        and needs to be corrected.
        The outside corner points are not addressed.

        Parameters
        ----------
        block : Block
            The solid object to address.

        Returns
        -------
        QuantumCircuit
            The circuit that sets the appropriate state on the object ancilla qubit.
        """
        circuit = self.lattice.circuit.copy()

        for dim in range(self.lattice.num_dims):
            for bound, wall in enumerate(block.walls_outside[dim]):
                comparator_circuit = SpecularWallComparator(
                    self.lattice, wall, self.logger
                ).circuit

                grid_qubit_indices_to_invert = [
                    self.lattice.grid_index(0)[0] + qubit
                    for qubit in wall.data.qubits_to_invert
                ]

                circuit.compose(comparator_circuit, inplace=True)

                if grid_qubit_indices_to_invert:
                    circuit.x(grid_qubit_indices_to_invert)

                # Reset the state for each velocity we care about:
                for v in block.get_lbm_wall_velocity_indices_to_reflect(
                    self.lattice.discretization, dim, bool(bound)
                ):
                    match self.lattice.get_encoding():
                        case ABEncodingType.AB:
                            qs = [
                                self.lattice.velocity_index()[0] + q
                                for q in get_qubits_to_invert(
                                    v, self.lattice.num_velocity_qubits
                                )
                            ]

                            if qs:
                                circuit.x(qs)

                            control_qubits = (
                                self.lattice.grid_index(wall.dim)
                                + self.lattice.ancillae_comparator_index()
                                + self.lattice.velocity_index()  # The reset step is additionally controlled on the velocity register
                            )

                            target_qubits = self.lattice.ancillae_obstacle_index(0)

                            circuit.compose(
                                MCMTGate(
                                    XGate(),
                                    len(control_qubits),
                                    len(target_qubits),
                                ),
                                qubits=control_qubits + target_qubits,
                                inplace=True,
                            )

                            if qs:
                                circuit.x(qs)
                        case ABEncodingType.OH:
                            control_qubits = (
                                self.lattice.grid_index(wall.dim)
                                + self.lattice.ancillae_comparator_index()
                                + [
                                    self.lattice.velocity_index()[
                                        v
                                    ]  # Only one velocity index required here
                                ]
                            )

                            target_qubits = self.lattice.ancillae_obstacle_index(0)

                            circuit.compose(
                                MCMTGate(
                                    XGate(),
                                    len(control_qubits),
                                    len(target_qubits),
                                ),
                                qubits=control_qubits + target_qubits,
                                inplace=True,
                            )

                        case _:
                            raise LatticeException(
                                f"Unsupported lattice encoding: {self.lattice.get_encoding()}"
                            )

                if grid_qubit_indices_to_invert:
                    circuit.x(grid_qubit_indices_to_invert)

                circuit.compose(comparator_circuit, inplace=True)

        return circuit

    def set_ancilla_of_point_state(
        self,
        points_data: List[Tuple[ReflectionPoint, List[int]]],
        ignore_velocity_data: bool,
    ) -> QuantumCircuit:
        """
        Sets the state of the obstacle ancilla qubit of a given gridpoint, conditioned on the velocity profile.

        Parameters
        ----------
        points_data : List[Tuple[ReflectionPoint, List[int]]]
            The tuple of gridpoint and velocity profile to set the ancilla state for.
        ignore_velocity_data : bool
            Whether to ignore the velocity data. Setting this to ``True`` will flip the state of the ancilla qubit based on position alone.

        Returns
        -------
        QuantumCircuit
            The circuit that sets the appropriate state on the object ancilla qubit.
        """
        circuit = self.lattice.circuit.copy()

        for point, velocities in points_data:
            grid_qubit_indices_to_invert = [
                self.lattice.grid_index(0)[0] + qubit
                for qubit in point.qubits_to_invert
            ]
            if grid_qubit_indices_to_invert:
                circuit.x(grid_qubit_indices_to_invert)

            match self.lattice.get_encoding():
                case ABEncodingType.AB:
                    velocity_data = (
                        [
                            [
                                self.lattice.velocity_index()[0] + qubit
                                for qubit in get_qubits_to_invert(
                                    velocity_index, self.lattice.num_velocity_qubits
                                )
                            ]
                            for velocity_index in velocities
                        ]
                        if not ignore_velocity_data
                        else [[]]
                    )

                    # Reset the state for each velocity we care about:
                    for velocity_qubit_indices_to_invert in velocity_data:
                        if velocity_qubit_indices_to_invert:
                            circuit.x(velocity_qubit_indices_to_invert)

                        control_qubits = (
                            self.lattice.grid_index()
                            + (
                                self.lattice.velocity_index()
                                if not ignore_velocity_data
                                else []
                            )  # The reset step is additionally controlled on the velocity register
                        )

                        target_qubits = self.lattice.ancillae_obstacle_index(0)

                        circuit.compose(
                            MCMTGate(
                                XGate(),
                                len(control_qubits),
                                len(target_qubits),
                            ),
                            qubits=control_qubits + target_qubits,
                            inplace=True,
                        )
                        if velocity_qubit_indices_to_invert:
                            circuit.x(velocity_qubit_indices_to_invert)
                case ABEncodingType.OH:
                    if ignore_velocity_data:
                        circuit.compose(
                            MCMTGate(
                                XGate(),
                                len(self.lattice.grid_index()),
                                len(self.lattice.ancillae_obstacle_index(0)),
                            ),
                            qubits=self.lattice.grid_index()
                            + self.lattice.ancillae_obstacle_index(0),
                            inplace=True,
                        )
                    else:
                        for v in velocities:
                            control_qubits = (
                                self.lattice.grid_index()
                                + (
                                    [self.lattice.velocity_index()[v]]
                                )  # Only one velocity control
                            )

                            target_qubits = self.lattice.ancillae_obstacle_index(0)

                            circuit.compose(
                                MCMTGate(
                                    XGate(),
                                    len(control_qubits),
                                    len(target_qubits),
                                ),
                                qubits=control_qubits + target_qubits,
                                inplace=True,
                            )
                case _:
                    raise LatticeException(
                        f"Unsupported lattice encoding: {self.encoding}"
                    )
            if grid_qubit_indices_to_invert:
                circuit.x(grid_qubit_indices_to_invert)

        return circuit

    def permute_and_stream(self) -> QuantumCircuit:
        """
        Performs the permutation of basis states that implements bounceback reflection in the amplitude-based encoding.

        Returns
        -------
        QuantumCircuit
            The permutation acting on only the velocity register.
        """
        circuit = self.lattice.circuit.copy()

        # Permute the velocities according to reflection rules
        circuit.compose(
            ABReflectionPermutation(
                self.lattice.num_velocity_qubits,
                self.lattice.discretization,
                self.lattice.get_encoding(),
                self.logger,
            )
            .circuit.control(1)
            .decompose(),
            qubits=self.lattice.ancillae_obstacle_index()
            + self.lattice.velocity_index(),
            inplace=True,
        )

        circuit.compose(
            ABStreamingOperator(
                self.lattice, self.lattice.ancillae_obstacle_index(), self.logger
            ).circuit,
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Operator ABStreaming with lattice {self.lattice}]"


class ABReflectionPermutation(LBMPrimitive):
    """
    Permutes velocity state to implement reflection in the amplitude-based encoding for :math:`D_dQ_q` discretizations.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABEncodingType, ABReflectionPermutation
        from qlbm.lattice import LatticeDiscretization

        ABReflectionPermutation(4, LatticeDiscretization.D2Q9, ABEncodingType.AB).draw("mpl")

    """

    num_qubits: int
    """
    The number of qubits that encode the velocity state.
    """

    discretization: LatticeDiscretization
    """
    The lattice discretization the permutation adheres to.
    """

    encoding: ABEncodingType
    """
    The type of encoding to permute for.
    """

    def __init__(
        self,
        num_qubits: int,
        discretization: LatticeDiscretization,
        encoding: ABEncodingType,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.num_qubits = num_qubits
        self.discretization = discretization
        self.encoding = encoding

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        if self.discretization == LatticeDiscretization.D2Q9:
            return self.__create_circuit_d2q9()

        raise LatticeException("AB reflection only currently supported in D2Q9")

    def __create_circuit_d2q9(self):
        circuit = QuantumCircuit(self.num_qubits)
        match self.encoding:
            case ABEncodingType.OH:
                circuit.swap(1, 3)
                circuit.swap(2, 4)
                circuit.swap(5, 7)
                circuit.swap(6, 8)

            case ABEncodingType.AB:
                # 1 <-> 3
                circuit.x([0, 1])
                circuit.mcx([0, 1, 3], 2)
                circuit.x([0, 1])

                # 2 <-> 4
                circuit.x([0, 3])
                circuit.cx(1, 2)
                circuit.mcx([0, 2, 3], 1)
                circuit.cx(1, 2)
                circuit.x([0, 3])

                # 5 <-> 7
                circuit.x(0)
                circuit.mcx([0, 1, 3], 2)
                circuit.x(0)

                # 6 <-> 8
                circuit.cx(0, 1)
                circuit.cx(0, 2)
                circuit.x(3)
                circuit.mcx([1, 2, 3], 0)
                circuit.cx(0, 2)
                circuit.cx(0, 1)
                circuit.x(3)

            case _:
                raise LatticeException(f"Unsupported lattice encoding: {self.encoding}")

        return circuit.reverse_bits() if self.encoding == ABEncodingType.AB else circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABReflectionPermutation with {self.num_qubits} qubits on {self.discretization}]"
