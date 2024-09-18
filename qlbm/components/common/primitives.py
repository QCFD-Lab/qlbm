from enum import Enum
from logging import Logger, getLogger
from math import pi
from time import perf_counter_ns

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

from qlbm.components.base import LBMPrimitive
from qlbm.lattice import Lattice
from qlbm.tools import bit_value


class ComparatorMode(Enum):
    """
    Enumerator for the modes of quantum comparator circuits.
    """

    LT = (1,)
    LE = (2,)
    GT = (3,)
    GE = (4,)


class SpeedSensitiveAdder(LBMPrimitive):
    """
    A QFT-based incrementer used in the CQBM algorithm :cite:p:`collisionless`.


    ========================= ======================================================================
    Atribute                  Summary
    ========================= ======================================================================
    :attr:`num_qubits`        Number of qubits of the circuit.
    :attr:`speed`             The index of the speed to increment.
    :attr:`positive`          Whether to increment the particles travelling at this speed in the positive or negative direction.
    :attr:`logger`            The performance logger, by default getLogger("qlbm")
    ========================= ======================================================================

    """

    def __init__(
        self,
        num_qubits: int,
        speed: int,
        positive: bool,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.num_qubits = num_qubits
        self.speed = speed
        self.positive = positive

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)

        circuit.compose(QFT(self.num_qubits), inplace=True)
        circuit.compose(
            SpeedSensitivePhaseShift(
                self.num_qubits,
                self.speed,
                self.positive,
                logger=self.logger,
            ).circuit,
            inplace=True,
        )
        circuit.compose(QFT(self.num_qubits, inverse=True), inplace=True)

        return circuit

    def __str__(self) -> str:
        return f"[Primitive SimpleAdder on {self.num_qubits} qubits, on velocity {self.speed}, in direction {self.positive}]"


class Comparator(LBMPrimitive):
    """
    Quantum Comparator operator that compares two
    a quantum state of `n` qubits and an integer
    `k` with respect to a :class:`.ComparatorMode`.
    """

    def __init__(
        self,
        num_qubits: int,
        num_to_compare: int,
        mode: ComparatorMode,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.num_qubits = num_qubits
        self.num_to_compare = num_to_compare
        self.mode = mode

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self) -> QuantumCircuit:
        return self.__create_circuit(self.num_qubits, self.num_to_compare, self.mode)

    def __create_circuit(
        self, num_qubits: int, num_to_compare: int, mode: ComparatorMode
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(num_qubits)

        match mode:
            case ComparatorMode.LT:
                circuit.compose(
                    SpeedSensitiveAdder(
                        num_qubits, num_to_compare, positive=False, logger=self.logger
                    ).circuit,
                    inplace=True,
                )
                circuit.compose(
                    SpeedSensitiveAdder(
                        num_qubits - 1,
                        num_to_compare,
                        positive=True,
                        logger=self.logger,
                    ).circuit,
                    inplace=True,
                    qubits=range(num_qubits - 1),
                )
                return circuit
            case ComparatorMode.LE:
                if num_to_compare == 2 ** (num_qubits - 1) - 1:
                    return self.__create_circuit(num_qubits, 0, ComparatorMode.GE)

                return self.__create_circuit(
                    num_qubits, num_to_compare + 1, ComparatorMode.LT
                )
            case ComparatorMode.GT:
                if num_to_compare == 2 ** (num_qubits - 1) - 1:
                    return circuit
                else:
                    return self.__create_circuit(
                        num_qubits, num_to_compare + 1, ComparatorMode.GE
                    )
            case ComparatorMode.GE:
                circuit = self.__create_circuit(
                    num_qubits, num_to_compare, ComparatorMode.LT
                )
                circuit.x(num_qubits - 1)
                return circuit
            case _:
                raise ValueError("Invalid Comparator Mode")

    def __str__(self) -> str:
        return f"[Primitive Comparator of {self.num_qubits} and {self.num_to_compare}, mode={self.mode}]"


class PhaseShift(LBMPrimitive):
    """
    Applies the phase-shift required for controlled
    incrementation as part of the streaming operator
    of the CQBM algorithm :cite:p:`collisionless`.
    Used in the :class`.ControlledIncrementer`.
    """

    def __init__(
        self,
        num_qubits: int,
        positive: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        """
        Parameters
        ----------
        num_qubits : int
            The number of qubits of the circuit.
        positive : bool, optional
            Whether incrementation should be performed in the positive direction, by default False.
        logger : Logger, optional
            The performance logger, by default getLogger("qlbm")
        """
        super().__init__(logger)

        self.num_qubits = num_qubits
        self.positive = positive

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)

        for c, qubit_index in enumerate(range(self.num_qubits)):
            # (2 * positive - 1) will flip the sign if positive is False
            # This effectively inverts the circuit
            phase = (2 * self.positive - 1) * pi / (2 ** (self.num_qubits - 1 - c))
            circuit.p(phase, qubit_index)

        return circuit

    def __str__(self) -> str:
        return f"[Primitive PhaseShift of {self.num_qubits} qubits, in direction {self.positive}]"


class SpeedSensitivePhaseShift(LBMPrimitive):
    def __init__(
        self,
        num_qubits: int,
        speed: int,
        positive: bool = False,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.num_qubits = num_qubits
        self.speed = speed
        self.positive = positive

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)
        angles = np.zeros(self.num_qubits)

        for qubit_index in range(self.num_qubits):
            dig = bit_value(self.speed, qubit_index)
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
            circuit.p(angles[qubit_index], qubit_index)

        return circuit

    def __str__(self) -> str:
        return f"[Primitive SpeedSensitivePhaseShift of {self.num_qubits} qubits, speed {self.speed}, in direction {self.positive}]"


class EmptyPrimitive(LBMPrimitive):
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

    def create_circuit(self) -> QuantumCircuit:
        return QuantumCircuit(*self.lattice.registers)

    def __str__(self) -> str:
        return f"[Primitive EmptyPrimitive with lattice {self.lattice}]"
