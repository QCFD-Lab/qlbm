from logging import Logger, getLogger
from math import pi
from time import perf_counter_ns

import numpy as np
from qiskit import QuantumCircuit
from qiskit.synthesis import synth_qft_full as QFT
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.tools import bit_value


class ParameterizedPhaseShift(LBMPrimitive):
    r"""A primitive that applies the phase-shift as part of the :class:`.ParameterizedDraperAdder` used in :class:`.Comparator`\ s.

    The rotation applied is :math:`\pm \frac{\pi}{2^{n_q - 1 - j}}`, with :math:`j` the position of the qubit (indexed starting with 0).
    Unlike the regular :class:`.PhaseShift`, the parameterized version additionally adds a phase relative to the number supplied.
    For an in-depth mathematical explanation of the procedure, consult Sections 4 and 5.5 of :cite:t:`collisionless`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`num_qubits`        The number of qubits to perform the phase shift for.
    :attr:`positive`          Whether the phase shift is applied to increment (T)
                              or decrement (F) the position of the particles.
                              Defaults to ``False``.
    :attr:`num_to_add`        The specific number to add as part of the Draper Adder.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    :attr:`num_ctrl_qubits`   The number of qubits to control the PhaseShift.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components import ParameterizedPhaseShift

        # A phase shift of 5 qubits, adding the number 2
        ParameterizedPhaseShift(num_qubits=5, num_to_add=2, positive=True).draw("mpl")

    Including control qubits:

    .. plot::
        :include-source:

        from qlbm.components import ParameterizedPhaseShift

        # A phase shift of 5 qubits, controlled subtracting the number 1
        ParameterizedPhaseShift(num_qubits=5, num_to_add=1, positive=False, num_ctrl_qubits=3).draw("mpl")
    """

    num_qubits: int
    """The number of qubits the phase shift is performed on."""

    num_to_add: int
    """The number to add to the basis states encoded in the qubits."""

    positive: bool
    """Whether the operation is an addition or a subtraction."""

    num_ctrl_qubits: int
    """Optional additional qubits to control the operation on. If any, the control qubits trail the target qubits."""

    def __init__(
        self,
        num_qubits: int,
        num_to_add: int,
        positive: bool = False,
        num_ctrl_qubits: int = 0,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.num_qubits = num_qubits
        self.num_to_add = num_to_add
        self.positive = positive
        self.num_ctrl_qubits = num_ctrl_qubits

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits + self.num_ctrl_qubits)
        angles = np.zeros(self.num_qubits)

        for qubit_index in range(self.num_qubits):
            dig = bit_value(self.num_to_add, qubit_index)
            for i in range(self.num_qubits - qubit_index):
                # (2 * positive - 1) will flip the sign if positive is False
                # This effectively inverts the circuit
                angles[i] += (
                    (2 * self.positive - 1)
                    * dig
                    * pi
                    / (2 ** (self.num_qubits - qubit_index - i - 1))
                )

        for qubit_index in range(self.num_qubits):
            if self.num_ctrl_qubits == 0:
                circuit.p(angles[qubit_index], qubit_index)
            else:
                circuit.mcp(
                    angles[qubit_index],
                    list(
                        range(self.num_qubits, self.num_qubits + self.num_ctrl_qubits)
                    ),
                    qubit_index,
                )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ParameterizedPhaseShift of {self.num_qubits} qubits, num {self.num_to_add}, in direction {self.positive}, ctrl {self.num_ctrl_qubits}]"


class ParameterizedDraperAdder(LBMPrimitive):
    r"""A QFT-based incrementer used to perform streaming in the algorithms based on amplitude encodings.

    Incrementation and decerementation are performed as rotations on grid qubits
    that have been previously mapped to the Fourier basis.
    This happens by nesting a :class:`.ParameterizedPhaseShift` primitive
    between regular and inverse :math:`QFT`\ s.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`num_qubits`        Number of qubits of the circuit.
    :attr:`num_to_add`.       The number to add.
    :attr:`positive`          Whether to increment in in the positive (T) or negative (F) direction.
    :attr:`num_ctrl_qubits`   The number of qubits to control the PhaseShift.
    :attr:`logger`            The performance logger, by default getLogger("qlbm")
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components import ParameterizedDraperAdder

        ParameterizedDraperAdder(4, 1, True).draw("mpl")
    """

    num_qubits: int
    """The number of qubits the phase shift is performed on."""

    num_to_add: int
    """The number to add to the basis states encoded in the qubits."""

    positive: bool
    """Whether the operation is an addition or a subtraction."""

    num_ctrl_qubits: int
    """Optional additional qubits to control the operation on. If any, the control qubits trail the target qubits."""

    def __init__(
        self,
        num_qubits: int,
        num_to_add: int,
        positive: bool,
        num_ctrl_qubits: int = 0,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.num_qubits = num_qubits
        self.num_to_add = num_to_add
        self.positive = positive
        self.num_ctrl_qubits = num_ctrl_qubits

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits + self.num_ctrl_qubits)

        circuit.compose(
            QFT(self.num_qubits), inplace=True, qubits=list(range(self.num_qubits))
        )
        circuit.compose(
            ParameterizedPhaseShift(
                self.num_qubits,
                self.num_to_add,
                self.positive,
                num_ctrl_qubits=self.num_ctrl_qubits,
                logger=self.logger,
            ).circuit,
            inplace=True,
        )
        circuit.compose(
            QFT(self.num_qubits, inverse=True),
            inplace=True,
            qubits=list(range(self.num_qubits)),
        )

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive SimpleAdder on {self.num_qubits} qubits, on velocity {self.num_to_add}, in direction {self.positive}]"


class PhaseShift(LBMPrimitive):
    r"""
    A primitive that applies the phase-shift as part of the :class:`.ControlledIncrementer` used in the :class:`.MSStreamingOperator`.

    The rotation applied is :math:`\pm\frac{\pi}{2^{n_q - 1 - j}}`, with :math:`j` the position of the qubit (indexed starting with 0).
    For an in-depth mathematical explanation of the procedure, consult Section 4 of :cite:t:`collisionless`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`num_qubits`        The number of qubits to perform the phase shift for.
    :attr:`positive`          Whether the phase shift is applied to increment (T)
                              or decrement (F) the position of the particles.
                              Defaults to ``False``.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ms import PhaseShift

        # A phase shift of 5 qubits
        PhaseShift(num_qubits=5, positive=False).draw("mpl")
    """

    def __init__(
        self,
        num_qubits: int,
        positive: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.num_qubits = num_qubits
        self.positive = positive

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)

        for c, qubit_index in enumerate(range(self.num_qubits)):
            # (2 * positive - 1) will flip the sign if positive is False
            # This effectively inverts the circuit
            phase = (2 * self.positive - 1) * pi / (2 ** (self.num_qubits - 1 - c))
            circuit.p(phase, qubit_index)

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive PhaseShift of {self.num_qubits} qubits, in direction {self.positive}]"
