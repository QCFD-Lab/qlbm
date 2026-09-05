r"""The end-to-end algorithm of the Amplitude-Based QLBM with an angle-encoded BGK collision."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.ab.collision.bgk_collision import ABBGKCollisionOperator
from qlbm.components.ab.reflection.standard_reflection import ABReflectionOperator
from qlbm.components.ab.streaming import ABStreamingOperator
from qlbm.components.base import LBMAlgorithm
from qlbm.lattice.lattices.ab_bgk_lattice import ABBGKLattice


class ABBGKQLBM(LBMAlgorithm):
    r"""
    Implementation of the **A** mplitude **B** ased QLBM with a **BGK** collision.

    Where :class:`.ABQLBM` interleaves streaming and boundary conditions only, this
    algorithm adds a :math:`\tau=1` BGK collision to the start of every time step, so a
    full LBM cycle is carried out per circuit:

    #. :class:`.ABBGKCollisionOperator` relaxes every grid point onto its local
       equilibrium in a single five-qubit gate.
    #. :class:`.ABStreamingOperator` transports the post-collision populations,
       controlled on the marker qubit.
    #. :class:`.ABReflectionOperator` applies the boundary conditions of the lattice
       geometry, likewise controlled on the marker qubit.

    The collision is only unitary because it splits the state into a physical and an
    auxiliary sector; see :class:`.ABBGKLattice` for the meaning of the marker qubit.
    Streaming and reflection are therefore both restricted to the physical sector,
    leaving the auxiliary amplitudes untouched.

    Because the collision maps the same five qubits from a branch--angle state onto
    populations, and not back again, a time step cannot simply be repeated on its own
    output. Every step starts from a freshly encoded flow field, which
    :class:`.ABBGKReinitializer` produces from the state at the end of the previous
    step. This makes the algorithm a hybrid loop, and is what allows the encoding to
    represent the nonlinearity of the equilibrium exactly rather than to linearize it.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABBGKQLBM
        from qlbm.lattice import ABBGKLattice

        lattice = ABBGKLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "d2q9"},
                "geometry": [
                    {
                        "shape": "cuboid",
                        "x": [3, 4],
                        "y": [3, 4],
                        "boundary": "bounceback",
                    }
                ],
            }
        )

        ABBGKQLBM(lattice).draw("mpl")

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`lattice`
          - The :class:`.ABBGKLattice` based on which the properties of the algorithm are inferred.
        * - :attr:`logger`
          - The performance logger, by default ``getLogger("qlbm")``.
    """

    lattice: ABBGKLattice
    """The lattice to construct the algorithm for."""

    def __init__(
        self,
        lattice: ABBGKLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(*self.lattice.registers)

        circuit.compose(
            ABBGKCollisionOperator(
                self.lattice,
                logger=self.logger,
            ).circuit,
            inplace=True,
        )

        circuit.compose(
            ABStreamingOperator(
                self.lattice,
                additional_control_qubit_indices=self.lattice.marker_index(),
                logger=self.logger,
            ).circuit,
            inplace=True,
        )

        if self.lattice.shape_list:
            circuit.compose(
                ABReflectionOperator(
                    self.lattice,
                    None,
                    control_on_marker_state=True,
                    logger=self.logger,
                ).circuit,
                inplace=True,
            )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Algorithm ABBGKQLBM with lattice {self.lattice}]"
