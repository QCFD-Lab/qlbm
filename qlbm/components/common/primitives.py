"""Common primitives used for multiple encodings."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, Tuple

import numpy as np
from numpy import pi
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, XGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import synth_qft_full as QFT
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.components.common.adders import ParameterizedDraperAdder
from qlbm.lattice import Lattice
from qlbm.tools.utils import get_qubits_to_invert


class EmptyPrimitive(LBMPrimitive):
    """
    Empty primitive used for effectively not specifying parts of the QLBM algorithm.

    Useful in situations where testing the end-to-end implementation of the algorithm
    where one part of the algorithm is left out or not yet implemented.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.MSLattice` or :class:`.SpaceTimeLattice` based on which the number of qubits is inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================
    """

    def __init__(
        self,
        lattice: Lattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        return self.lattice.circuit.copy()

    @override
    def __str__(self) -> str:
        return f"[Primitive EmptyPrimitive with lattice {self.lattice}]"


class MCSwap(LBMPrimitive):
    """
    Decomposition of a Multi-Controlled Swap Gate into 1 multi-controlled :math:`X` gate and 2 single-controlled :math:`X` gates.

    Decomposition taken from :cite:t:`mcswap`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`lattice`           The :class:`.Lattice` based on which the number of qubits is inferred.
    :attr:`control_qubits`    The qubits that control the swap gate.
    :attr:`target_qubits`     The two qubits to be swapped.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``.
    ========================= ======================================================================
    """

    def __init__(
        self,
        lattice: Lattice,
        control_qubits: List[int],
        target_qubits: Tuple[int, int],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.lattice = lattice
        self.control_qubits = control_qubits
        self.target_qubits = target_qubits

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        circuit.cx(self.target_qubits[1], self.target_qubits[0])
        circuit.compose(
            MCMTGate(
                XGate(), len(self.control_qubits) + 1, len(self.target_qubits) - 1
            ),
            qubits=self.control_qubits + list(self.target_qubits),
            inplace=True,
        )
        circuit.cx(self.target_qubits[1], self.target_qubits[0])

        return circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive MCSwap with lattice {self.lattice}]"


class HammingWeightAdder(LBMPrimitive):
    """
    QFT-based Hamming Weight adder.

    This primitive adds the hamming weight (number of 1s) in a given register :math:`x`
    to the binary-encoded value of a second register :math:`y`.
    """

    x_register_size: int
    """
    The size of the register encoding the hamming weight value to add.
    """

    y_register_size: int
    """
    The size of the register to which the hamming weight is added.
    """

    def __init__(
        self,
        x_register_size: int,
        y_register_size: int,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)
        self.x_register_size = x_register_size
        self.y_register_size = y_register_size

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.x_register_size + self.y_register_size)

        circuit.compose(
            QFT(self.y_register_size),
            inplace=True,
            qubits=list(
                range(self.x_register_size, self.x_register_size + self.y_register_size)
            ),
        )

        angles = np.zeros(self.y_register_size)
        for i in range(self.y_register_size):
            angles[i] = 2 * pi / (2 ** (self.y_register_size - i))

        for xi in range(self.x_register_size):
            for k, yi in enumerate(range(self.y_register_size)):
                circuit.cp(angles[k], xi, self.x_register_size + yi)

        circuit.compose(
            QFT(self.y_register_size, inverse=True),
            inplace=True,
            qubits=list(
                range(self.x_register_size, self.x_register_size + self.y_register_size)
            ),
        )

        return circuit

    @override
    def __str__(self):
        return f"[Primitive HWAdder with with register size {self.x_register_size} and {self.y_register_size}]"


class TruncatedQFT(LBMPrimitive):
    r"""Truncated Quantum Fourier Transform primitive used to create an equal magnitude superposition.

    For a superposition of the first :math:`k` basis states encoded in :math:`n` qubits,
    the operator consists of discrete fourier transform block of size :math:`k\times k`,
    padded with :math:`2^n - k` :math:`1`s on the main diagonal.
    The rationale and properties of this operator are described in :cite:`spacetime2`.
    This primitive is used in both amplitude-based and computational basis state encodings.
    In the :class:`ABInitialConditions`, it creates an equal magnitude superposition over the velocity space.
    In the :class:`EQCRedistribution`, the superposition is over all basis states with an equivalent mass and momenta.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.common import TruncatedQFT

        TruncatedQFT(4, 7).decompose(reps=2).draw("mpl")
    """

    num_qubits: int
    """The number of qubits the operator acts on."""

    dft_size: int
    """The size of the discrete Fourier transform block."""

    def __init__(
        self,
        num_qubits: int,
        dft_size: int,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)
        self.num_qubits = num_qubits
        self.dft_size = dft_size

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self):
        circuit = QuantumCircuit(self.num_qubits)

        QFT = np.array(
            [
                [
                    np.exp(2j * np.pi * i * j / self.dft_size) / np.sqrt(self.dft_size)
                    for j in range(self.dft_size)
                ]
                for i in range(self.dft_size)
            ]
        )

        U = np.eye(2**self.num_qubits, dtype=complex)
        U[: self.dft_size, : self.dft_size] = QFT
        op = Operator(U)
        assert op.is_unitary()

        circuit.append(op, list(range(self.num_qubits)))

        return circuit

    @override
    def __str__(self):
        return f"[Primitive TuncatedQFT({self.num_qubits}, {self.dft_size})]"


class UniformStatePrep(LBMPrimitive):
    r"""Efficient uniform state preparation primitive used to create an equal magnitude superposition over the first :math:`k` basis states.

    This is an implementation of Algorithm 1 described by :cite:t:`uniprep`.
    It is used to create an uniform magnitude superposition over arbitrary
    velocity states in :class:`.ABDiscreteUniformInitialConditions`.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.common import UniformStatePrep

        UniformStatePrep(4, 7).decompose(reps=2).draw("mpl")
    """

    num_qubits: int
    """The number of qubits the operator acts on."""

    num_states: int
    """The number of states to generate."""

    def __init__(
        self,
        num_qubits: int,
        num_states: int,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)
        self.num_qubits = num_qubits
        self.num_states = num_states

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self):
        circuit = QuantumCircuit(
            self.num_qubits, name=f"UniformStatePrep{self.num_states}"
        )

        # M = 1 : do nothing, stays in |0...0>
        if self.num_states == 1:
            return circuit

        # If M is a power of two, the solution is trivial: Hadamards on log2(M) qubits
        is_power_of_two = (self.num_states & (self.num_states - 1)) == 0
        if is_power_of_two:
            r = int(np.log2(self.num_states))
            for q in range(r):
                circuit.h(q)
            return circuit

        # --- General case: Algorithm 1 (Section 2.1 of the paper) ---

        # We only need n_eff = ceil(log2 M) active qubits; the rest stay in |0>
        n_eff = self._ceil_log2_M(self.num_states)
        if n_eff > self.num_qubits:
            raise ValueError("Internal error: n_eff > num_qubits.")

        # Binary decomposition: M = Σ_j 2^{l_j}, with 0 <= l0 < l1 < ... < lk
        bit_positions = [i for i in range(n_eff) if (self.num_states >> i) & 1]
        bit_positions.sort()
        l0 = bit_positions[0]
        k = len(bit_positions) - 1  # number of "higher" bits

        # Helper: safe acos for numerical stability
        def safe_acos(x: float) -> float:
            return np.acos(max(-1.0, min(1.0, x)))

        # Step 4: Apply X on qubits at positions l1, l2, ..., lk
        for j in range(1, len(bit_positions)):
            circuit.x(bit_positions[j])

        # Step 5: M0 = 2^{l0}
        M_prev = 2**l0  # This is M_0 in the paper

        # Step 6–7: If l0 > 0, apply H on qubits 0..(l0-1)
        if l0 > 0:
            for q in range(l0):
                circuit.h(q)

        # Step 8: Apply RY(theta0) on |q_{l1}>, theta0 = -2 arccos( sqrt(M0 / M) )
        l1 = bit_positions[1]
        theta0 = -2.0 * safe_acos(np.sqrt(M_prev / self.num_states))
        circuit.ry(theta0, l1)

        # Step 9: Controlled H on qubits i in [l0, l1) with open control on q_{l1} == |0>
        ctrl = l1
        circuit.x(ctrl)  # convert open control (on |0>) to normal control (on |1>)
        for i in range(l0, l1):
            circuit.ch(ctrl, i)
        circuit.x(ctrl)

        # Steps 10–13: For-loop over remaining bits
        for m in range(1, k):
            l_m = bit_positions[m]
            l_next = bit_positions[m + 1]

            # Step 11: Controlled RY(theta_m) on q_{l_{m+1}} with open control on q_{l_m} == |0>
            numerator = 2**l_m
            denominator = self.num_states - M_prev
            theta_m = -2.0 * safe_acos(np.sqrt(numerator / denominator))

            # open control on q_{l_m}
            ctrl = l_m
            target = l_next
            circuit.x(ctrl)
            circuit.cry(theta_m, ctrl, target)
            circuit.x(ctrl)

            # Step 12: Controlled H on qubits i in [l_m, l_{m+1}) with open control on q_{l_{m+1}} == |0>
            ctrl_next = l_next
            circuit.x(ctrl_next)
            for i in range(l_m, l_next):
                circuit.ch(ctrl_next, i)
            circuit.x(ctrl_next)

            # Step 13: M_m = M_{m-1} + 2^{l_m}
            M_prev += 2**l_m

        return circuit

    def _ceil_log2_M(self, M: int) -> int:
        """Minimal number of qubits n such that M <= 2**n."""
        if M <= 1:
            return 1
        # Power of two?
        if M & (M - 1) == 0:
            return int(np.log2(M))
        # Non power-of-two
        return M.bit_length()

    @override
    def __str__(self):
        return f"[Primitive UniformStatePrep({self.num_qubits}, {self.num_states})]"


class AdditionConversion(LBMPrimitive):
    num_qubits: int
    """The number of qubits the states are encoded in."""

    state_from: int
    """The starting state to convert."""

    state_to: int
    """The state to convert to."""

    def __init__(
        self,
        num_qubits: int,
        state_from: int,
        state_to: int,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)
        self.num_qubits = num_qubits
        self.state_from = state_from
        self.state_to = state_to

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self):
        circuit = QuantumCircuit(self.num_qubits + 1)

        state_setter_circ = StateSetter(
            self.num_qubits, self.state_from, self.logger
        ).circuit

        circuit.compose(
            state_setter_circ, qubits=list(range(self.num_qubits)), inplace=True
        )
        circuit.mcx(list(range(self.num_qubits)), self.num_qubits)
        circuit.compose(
            state_setter_circ, qubits=list(range(self.num_qubits)), inplace=True
        )

        circuit.compose(
            ParameterizedDraperAdder(
                self.num_qubits,
                abs(self.state_to - self.state_from),
                self.state_to > self.state_from,
                1,
                self.logger,
            ).circuit,
            inplace=True,
        )

        state_setter_circ = StateSetter(
            self.num_qubits, self.state_to, self.logger
        ).circuit

        circuit.compose(
            state_setter_circ, qubits=list(range(self.num_qubits)), inplace=True
        )
        circuit.mcx(list(range(self.num_qubits)), self.num_qubits)
        circuit.compose(
            state_setter_circ, qubits=list(range(self.num_qubits)), inplace=True
        )

        return circuit

    @override
    def __str__(self):
        return f"[Primitive AdditionConversion({self.num_qubits}, {self.state_from}, {self.state_to})]"


class StateSetter(LBMPrimitive):
    num_qubits: int
    """The number of qubits the state is encoded in."""

    state_to_set: int
    """The state to convert to :math:`\ket{1}^{\otimes n}`"""

    def __init__(
        self,
        num_qubits: int,
        state_to_set: int,
        logger: Logger = getLogger("qlbm"),
    ):
        super().__init__(logger)
        self.num_qubits = num_qubits
        self.state_to_set = state_to_set

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self):
        circuit = QuantumCircuit(self.num_qubits)

        qs = get_qubits_to_invert(self.state_to_set, self.num_qubits)

        if qs:
            circuit.x(qs)

        return circuit

    @override
    def __str__(self):
        return f"[Primitive StateSetter({self.num_qubits}, {self.state_to_set}]"
