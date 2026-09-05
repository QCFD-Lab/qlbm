r"""Quantum circuits used for collision in the :class:`.ABBGKQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from typing_extensions import override

from qlbm.components.ab.collision.angle_encoding import ABAngleEncodedEquilibrium
from qlbm.components.base import LBMOperator, LBMPrimitive
from qlbm.lattice.bgk.angle_encoding import D2Q9AngleEncoding
from qlbm.lattice.lattices.ab_bgk_lattice import ABBGKLattice


class ABBGKCollisionOperator(LBMOperator):
    r"""
    Collision operator for the :class:`.ABBGKQLBM` algorithm.

    Applies the :math:`32 \times 32` unitary of :meth:`.D2Q9AngleEncoding.collision_unitary`
    to the four velocity qubits and the marker qubit of an :class:`.ABBGKLattice`.
    Because the operator touches no grid qubit, the single gate performs the
    :math:`\tau=1` BGK collision at every grid point of the lattice simultaneously.

    On input, the five qubits carry the branch--angle state that
    :class:`.ABBGKInitialConditions` prepares. On output, they carry the normalized
    equilibrium populations in the marker-one sector:

    .. math::
        \ket{\psi(\theta_x, \theta_y; \beta)}
        \mapsto
        \sum_{i=0}^{8} \frac{f_i^{eq}}{\alpha} \ket{1}_M \ket{i}_V + \ket{0}_M \ket{\text{aux}},

    with the auxiliary marker-zero amplitudes accounting for the loss of norm that
    makes an inherently non-unitary collision implementable as a unitary.

    Since only marker-one amplitudes are physical, every operator applied after this
    one must be controlled on the marker qubit. :class:`.ABBGKQLBM` takes care of that
    for streaming and reflection.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab.collision import ABBGKCollisionOperator
        from qlbm.lattice import ABBGKLattice

        lattice = ABBGKLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        ABBGKCollisionOperator(lattice).draw("mpl")

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`lattice`
          - The :class:`.ABBGKLattice` based on which the properties of the operator are inferred.
        * - :attr:`logger`
          - The performance logger, by default ``getLogger("qlbm")``.
    """

    lattice: ABBGKLattice
    """The lattice to construct the component for."""

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
        circuit = self.lattice.circuit.copy()

        circuit.append(
            UnitaryGate(
                self.lattice.encoding.collision_unitary(),
                label="BGK collision",
            ),
            self.lattice.collision_index(),
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Operator ABBGKCollision with lattice {self.lattice}]"


class ABLocalBGKCollision(LBMPrimitive):
    r"""
    The complete five-qubit angle-encoded collision for a single macroscopic velocity.

    Composes :class:`.ABAngleEncodedEquilibrium` with the collision unitary, and is
    therefore the smallest self-contained demonstration of the encoding: it maps a
    velocity pair onto the amplitudes :math:`f_i^{eq} / \alpha` stored in the
    marker-one sector of a five-qubit register. The lattice-wide
    :class:`.ABBGKCollisionOperator` performs the same map, but at every grid point at
    once and on a state prepared by :class:`.ABBGKInitialConditions`.

    Example usage:

    .. testcode::

        import numpy as np
        from qiskit.quantum_info import Statevector

        from qlbm.components.ab.collision import ABLocalBGKCollision

        collision = ABLocalBGKCollision(0.05, -0.02)
        amplitudes = Statevector(collision.circuit).data.real

        # The marker-one sector holds the equilibrium populations, scaled by alpha.
        populations = collision.encoding.alpha * amplitudes[16:25]

        print(
            np.allclose(
                populations, collision.encoding.equilibrium(1.0, 0.05, -0.02)
            )
        )

    .. testoutput::

        True

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`velocity_x`
          - The :math:`x` velocity component to collide, bounded by the encoding.
        * - :attr:`velocity_y`
          - The :math:`y` velocity component to collide, bounded by the encoding.
        * - :attr:`encoding`
          - The :class:`.D2Q9AngleEncoding` to use, by default a new instance.
        * - :attr:`logger`
          - The performance logger, by default ``getLogger("qlbm")``.
    """

    velocity_x: float
    r"""The encoded :math:`x` velocity component."""

    velocity_y: float
    r"""The encoded :math:`y` velocity component."""

    encoding: D2Q9AngleEncoding
    """The angle encoding used to build the collision."""

    def __init__(
        self,
        velocity_x: float,
        velocity_y: float,
        encoding: D2Q9AngleEncoding | None = None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.velocity_x = float(velocity_x)
        self.velocity_y = float(velocity_y)
        self.encoding = D2Q9AngleEncoding() if encoding is None else encoding

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        register = QuantumRegister(self.encoding.NUM_COLLISION_QUBITS, "collision_io")
        circuit = QuantumCircuit(register, name="local_bgk_collision")

        circuit.compose(
            ABAngleEncodedEquilibrium(
                self.velocity_x, self.velocity_y, self.encoding, self.logger
            ).circuit,
            inplace=True,
        )

        circuit.append(
            UnitaryGate(self.encoding.collision_unitary(), label="BGK collision"),
            list(register),
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABLocalBGKCollision with velocity ({self.velocity_x}, {self.velocity_y})]"
