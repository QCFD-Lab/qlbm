r"""Quantum circuits that prepare the branch--angle states of the :class:`.ABBGKQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import Dict, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import RYGate
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.bgk.angle_encoding import D2Q9AngleEncoding
from qlbm.tools.exceptions import CircuitException

ROTATION_TOLERANCE = 1e-12
"""Rotations with an angle smaller than this are omitted from the circuit."""


def append_multi_controlled_ry(
    circuit: QuantumCircuit,
    angle: float,
    control_qubits: Sequence[Qubit],
    control_values: Sequence[int],
    target_qubit: Qubit,
) -> None:
    r"""
    Appends an :math:`RY` rotation conditioned on one computational basis state.

    Qiskit's multi-controlled gates trigger on :math:`\ket{1}` on every control, so
    controls that should instead trigger on :math:`\ket{0}` are temporarily flipped
    with :math:`X` gates and flipped back afterwards. A control pattern of ``[0, 1]``
    therefore becomes ``X`` on the first control, a doubly-controlled :math:`RY`,
    and ``X`` on the first control again.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        The circuit to append to, modified in place.
    angle : float
        The rotation angle. Rotations below :attr:`ROTATION_TOLERANCE` are skipped.
    control_qubits : Sequence[qiskit.circuit.Qubit]
        The control qubits. An empty sequence gives an unconditional rotation.
    control_values : Sequence[int]
        The basis state of the controls that activates the rotation.
    target_qubit : qiskit.circuit.Qubit
        The qubit that the rotation acts on.

    Raises
    ------
    CircuitException
        If the controls and their values have different lengths,
        or if a control value is neither 0 nor 1.
    """
    if abs(angle) < ROTATION_TOLERANCE:
        return

    if not control_qubits:
        circuit.ry(angle, target_qubit)
        return

    if len(control_qubits) != len(control_values):
        raise CircuitException(
            f"Got {len(control_qubits)} control qubits but {len(control_values)} control values."
        )

    if any(value not in (0, 1) for value in control_values):
        raise CircuitException(
            f"Control values must be either 0 or 1, got {list(control_values)}."
        )

    zero_controls = [
        qubit for qubit, value in zip(control_qubits, control_values) if value == 0
    ]

    if zero_controls:
        circuit.x(zero_controls)

    circuit.append(
        RYGate(angle).control(len(control_qubits)),
        [*control_qubits, target_qubit],
    )

    if zero_controls:
        circuit.x(zero_controls)


class ABBranchStatePreparation(LBMPrimitive):
    r"""
    Prepares the branch superposition :math:`\sum_k \beta_k \ket{k}` of the angle encoding.

    The three branch qubits span eight basis states, but only the six branches that
    carry a feature of the equilibrium are populated; the remaining two amplitudes are
    zero. Since the branch amplitudes are real and non-negative, the state is prepared
    exactly by a binary tree of uniformly controlled :math:`RY` rotations: at every
    level, the rotation angle follows from the probability mass held by the two
    subtrees of the qubit being prepared.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab.collision import ABBranchStatePreparation

        ABBranchStatePreparation().draw("mpl")

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`encoding`
          - The :class:`.D2Q9AngleEncoding` supplying the branch amplitudes, by default a new instance.
        * - :attr:`logger`
          - The performance logger, by default ``getLogger("qlbm")``.
    """

    encoding: D2Q9AngleEncoding
    """The angle encoding whose branch amplitudes are prepared."""

    def __init__(
        self,
        encoding: D2Q9AngleEncoding | None = None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.encoding = D2Q9AngleEncoding() if encoding is None else encoding

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        num_qubits = self.encoding.NUM_BRANCH_QUBITS
        register = QuantumRegister(num_qubits, "branch")
        circuit = QuantumCircuit(register, name="branch_state_preparation")

        amplitudes = np.zeros(1 << num_qubits, dtype=float)
        amplitudes[: self.encoding.NUM_BRANCHES] = self.encoding.beta

        def subtree_probability(fixed_bits: Dict[int, int]) -> float:
            """Sums the probability of every basis state matching the given bits."""
            return float(
                sum(
                    amplitude**2
                    for index, amplitude in enumerate(amplitudes)
                    if all(
                        ((index >> bit) & 1) == value
                        for bit, value in fixed_bits.items()
                    )
                )
            )

        # Walk the binary tree from the most significant branch bit downwards, so that
        # every rotation is controlled on the bits that have already been prepared.
        for level in range(num_qubits):
            bit = num_qubits - 1 - level
            higher_bits = list(range(bit + 1, num_qubits))

            for pattern in range(1 << level):
                fixed = {
                    higher_bit: (pattern >> position) & 1
                    for position, higher_bit in enumerate(higher_bits)
                }

                probability_zero = subtree_probability({**fixed, bit: 0})
                probability_one = subtree_probability({**fixed, bit: 1})

                if probability_zero + probability_one < ROTATION_TOLERANCE:
                    continue

                # RY(angle) splits an amplitude into cos(angle / 2) and sin(angle / 2),
                # so this angle assigns the correct mass to the two subtrees.
                append_multi_controlled_ry(
                    circuit=circuit,
                    angle=2.0
                    * float(
                        np.arctan2(np.sqrt(probability_one), np.sqrt(probability_zero))
                    ),
                    control_qubits=[register[higher_bit] for higher_bit in higher_bits],
                    control_values=[fixed[higher_bit] for higher_bit in higher_bits],
                    target_qubit=register[bit],
                )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABBranchStatePreparation with encoding {self.encoding}]"


class ABBranchAngleEncoding(LBMPrimitive):
    r"""
    Encodes a velocity into the angle qubits of a prepared branch superposition.

    Applies the branch-dependent rotations :math:`RY(2 m_x^k \theta_x)` and
    :math:`RY(2 m_y^k \theta_y)` that give each branch its feature. The rotations
    leave the branch amplitudes untouched, so composing this primitive after
    :class:`.ABBranchStatePreparation` produces the branch--angle state
    :math:`\ket{\psi(\theta_x, \theta_y; \beta)}` of :meth:`.D2Q9AngleEncoding.branch_state`.

    The six branch multipliers of :attr:`.D2Q9AngleEncoding.BRANCH_MULTIPLIERS` collapse
    into the four controlled rotations of :attr:`CONTROL_PATTERNS`, because branches
    that share a rotation on one axis also share a control pattern.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab.collision import ABBranchAngleEncoding

        ABBranchAngleEncoding(0.4, -0.2).draw("mpl")

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`angle_x`
          - The angle :math:`\theta_x` that encodes the :math:`x` velocity component.
        * - :attr:`angle_y`
          - The angle :math:`\theta_y` that encodes the :math:`y` velocity component.
        * - :attr:`encoding`
          - The :class:`.D2Q9AngleEncoding` that the rotations belong to, by default a new instance.
        * - :attr:`logger`
          - The performance logger, by default ``getLogger("qlbm")``.
    """

    CONTROL_PATTERNS: Tuple[Tuple[str, int, Dict[int, int]], ...] = (
        ("x", 1, {2: 0, 0: 1}),
        ("y", 1, {2: 0, 1: 1}),
        ("x", 2, {2: 1, 0: 0}),
        ("y", 2, {2: 1, 0: 1}),
    )
    r"""The compact set of controlled rotations that reproduce all six branch features.

    Each entry is an ``(axis, multiplier, controls)`` triple that rotates the angle qubit
    of ``axis`` by :math:`2 \cdot \text{multiplier} \cdot \theta_{\text{axis}}` whenever
    the branch bits match ``controls``, which maps a branch bit index onto its required
    value. Four rotations suffice for six branches because branches that share a rotation
    on one axis also share a control pattern."""

    angle_x: float
    r"""The angle :math:`\theta_x`."""

    angle_y: float
    r"""The angle :math:`\theta_y`."""

    encoding: D2Q9AngleEncoding
    """The angle encoding that the rotations belong to."""

    def __init__(
        self,
        angle_x: float,
        angle_y: float,
        encoding: D2Q9AngleEncoding | None = None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.angle_x = float(angle_x)
        self.angle_y = float(angle_y)
        self.encoding = D2Q9AngleEncoding() if encoding is None else encoding

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        angle_register = QuantumRegister(2, "angle")
        branch_register = QuantumRegister(self.encoding.NUM_BRANCH_QUBITS, "branch")
        circuit = QuantumCircuit(
            angle_register, branch_register, name="branch_angle_encoding"
        )

        # The angle register is ordered [q_y, q_x] to match the little-endian layout
        # that the collision unitary expects.
        targets = {
            "x": (angle_register[1], self.angle_x),
            "y": (angle_register[0], self.angle_y),
        }

        for axis, multiplier, controls in self.CONTROL_PATTERNS:
            target_qubit, angle = targets[axis]

            append_multi_controlled_ry(
                circuit=circuit,
                angle=2.0 * multiplier * angle,
                control_qubits=[branch_register[bit] for bit in controls],
                control_values=list(controls.values()),
                target_qubit=target_qubit,
            )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABBranchAngleEncoding with angles ({self.angle_x}, {self.angle_y})]"


class ABAngleEncodedEquilibrium(LBMPrimitive):
    r"""
    Prepares the five-qubit branch--angle state that encodes one macroscopic velocity.

    This is the state preparation half of the angle-encoded collision: it composes
    :class:`.ABBranchStatePreparation` with :class:`.ABBranchAngleEncoding` to build

    .. math::
        \ket{\psi(\theta_x, \theta_y; \beta)} =
        \sum_{k=0}^{5} \beta_k \ket{k}_B \ket{\varphi_k(\theta_x, \theta_y)},

    the state that :class:`.ABBGKCollisionOperator` maps onto the equilibrium
    populations. The circuit is laid out on the qubit order
    ``[q_y, q_x, branch_0, branch_1, branch_2]``, which is exactly the order in which
    the collision unitary reinterprets the same five qubits as
    ``[velocity_0, ..., velocity_3, marker]``.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab.collision import ABAngleEncodedEquilibrium

        ABAngleEncodedEquilibrium(0.05, -0.02).draw("mpl")

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`velocity_x`
          - The :math:`x` velocity component to encode, bounded by the encoding.
        * - :attr:`velocity_y`
          - The :math:`y` velocity component to encode, bounded by the encoding.
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
    """The angle encoding used to prepare the state."""

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
        angle_x, angle_y = self.encoding.angles(self.velocity_x, self.velocity_y)

        register = QuantumRegister(self.encoding.NUM_COLLISION_QUBITS, "collision_io")
        circuit = QuantumCircuit(register, name="angle_encoded_equilibrium")

        branch_qubits = list(register[2:])

        circuit.compose(
            ABBranchStatePreparation(self.encoding, self.logger).circuit,
            qubits=branch_qubits,
            inplace=True,
        )

        circuit.compose(
            ABBranchAngleEncoding(
                float(angle_x), float(angle_y), self.encoding, self.logger
            ).circuit,
            qubits=list(register[:2]) + branch_qubits,
            inplace=True,
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABAngleEncodedEquilibrium with velocity ({self.velocity_x}, {self.velocity_y})]"
