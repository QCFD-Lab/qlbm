"""Quantum circuits that perform arithmetic comparison operations."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit.library import DraperQFTAdder
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.components.common.adders import ParameterizedDraperAdder
from qlbm.tools import ComparatorMode


class TwoRegisterComparator(LBMPrimitive):
    """
    Quantum comparator primitive that compares the states of 2 registers of ``num_qubits`` qubits a :class:`~qlbm.tools.ComparatorMode`.

    The generate circuit is of size ``2*num_qubits+1``, where the last qubit of the register holds the boolean result.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.common.comparators import TwoRegisterComparator
        from qlbm.tools import ComparatorMode

        # Compare two registers of size 4
        TwoRegisterComparator(num_qubits=4, mode=ComparatorMode.LT).draw("mpl")
    """

    def __init__(
        self,
        num_qubits: int,
        mode: ComparatorMode,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.num_qubits = num_qubits
        self.mode = mode

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(2 * self.num_qubits + 1)
        x_register = list(range(self.num_qubits))
        y_register = list(range(self.num_qubits, 2 * self.num_qubits))
        output_qubit = 2 * self.num_qubits

        match self.mode:
            case ComparatorMode.GT:
                self.__compose_gt(circuit, x_register, y_register, output_qubit)
            case ComparatorMode.LE:
                self.__compose_gt(circuit, x_register, y_register, output_qubit)
                circuit.x(output_qubit)
            case ComparatorMode.LT:
                self.__compose_gt(circuit, y_register, x_register, output_qubit)
            case ComparatorMode.GE:
                self.__compose_gt(circuit, y_register, x_register, output_qubit)
                circuit.x(output_qubit)
            case _:
                raise ValueError("Invalid Comparator Mode")

        return circuit

    def __compose_gt(
        self,
        circuit: QuantumCircuit,
        x_register: List[int],
        y_register: List[int],
        output_qubit: int,
    ) -> None:
        add_half = DraperQFTAdder(self.num_qubits, kind="half")
        add_fixed_inv = DraperQFTAdder(self.num_qubits, kind="fixed").inverse()

        circuit.x(y_register)
        circuit.compose(
            add_half,
            qubits=x_register + y_register + [output_qubit],
            inplace=True,
        )
        circuit.compose(
            add_fixed_inv,
            qubits=x_register + y_register,
            inplace=True,
        )
        circuit.x(y_register)

    @override
    def __str__(self) -> str:
        return f"[Primitive TwoRegisterComparator of {self.num_qubits} qubits, mode={self.mode}]"


class SingleRegisterComparator(LBMPrimitive):
    """
    Quantum comparator primitive that compares a quantum state of ``num_qubits`` qubits and an integer ``num_to_compare`` with respect to a :class:`~qlbm.tools.ComparatorMode`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`num_qubits`        Number of qubits encoding the integer to compare.
    :attr:`num_to_compare`    The integer to compare against.
    :attr:`mode`              The :class:`~qlbm.tools.ComparatorMode` used to compare the two numbers.
    :attr:`logger`            The performance logger, by default getLogger("qlbm")
    ========================= ======================================================================

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.common.comparators import SingleRegisterComparator
        from qlbm.tools.utils import ComparatorMode

        # On a 5 qubit register, compare the number 3
        SingleRegisterComparator(num_qubits=5,
                                num_to_compare=3,
                                mode=ComparatorMode.LT).draw("mpl")
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

    @override
    def create_circuit(self) -> QuantumCircuit:
        return self.__create_circuit(self.num_qubits, self.num_to_compare, self.mode)

    def __create_circuit(
        self, num_qubits: int, num_to_compare: int, mode: ComparatorMode
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(num_qubits)

        match mode:
            case ComparatorMode.LT:
                circuit.compose(
                    ParameterizedDraperAdder(
                        num_qubits, num_to_compare, positive=False, logger=self.logger
                    ).circuit,
                    inplace=True,
                )
                circuit.compose(
                    ParameterizedDraperAdder(
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

    @override
    def __str__(self) -> str:
        return f"[Primitive Comparator of {self.num_qubits} and {self.num_to_compare}, mode={self.mode}]"
