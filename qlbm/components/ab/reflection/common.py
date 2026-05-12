"""Common utilities for reflection in the :class:`.ABQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import List, Tuple

from qiskit import QuantumCircuit
from typing_extensions import override

from qlbm.components.ab.encodings import ABEncodingType
from qlbm.components.base import LBMPrimitive
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException


class ABBounceBackReflectionPermutation(LBMPrimitive):
    """
    Permutes velocity state to implement bounce-back reflection in the amplitude-based encoding for :math:`D_dQ_q` discretizations.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABEncodingType, ABBounceBackReflectionPermutation
        from qlbm.lattice import LatticeDiscretization

        ABBounceBackReflectionPermutation(4, LatticeDiscretization.D2Q9, ABEncodingType.AB).draw("mpl")

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
        return f"[Primitive ABBounceBackReflectionPermutation with {self.num_qubits} qubits on {self.discretization}]"


class ABSpecularReflectionPermutation(LBMPrimitive):
    """
    Permutes velocity state to implement bounce-back reflection in the amplitude-based encoding for :math:`D_dQ_q` discretizations.

    Example usage:

    .. plot::
        :include-source:

        from qlbm.components.ab import ABEncodingType, ABSpecularReflectionPermutation
        from qlbm.lattice import LatticeDiscretization

        ABSpecularReflectionPermutation(4, LatticeDiscretization.D2Q9, ABEncodingType.AB, (True, True)).draw("mpl")

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

    supported_reflection_types: List[str] = ["bounceback", "specular"]

    def __init__(
        self,
        num_qubits: int,
        discretization: LatticeDiscretization,
        encoding: ABEncodingType,
        reflect_in_dim: Tuple[bool, ...],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)

        self.num_qubits = num_qubits
        self.discretization = discretization
        self.encoding = encoding
        self.reflect_in_dim = reflect_in_dim

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
                if self.reflect_in_dim[0]:
                    circuit.swap(1, 3)
                    circuit.swap(5, 6)
                    circuit.swap(8, 7)
                if self.reflect_in_dim[1]:
                    circuit.swap(2, 4)
                    circuit.swap(5, 8)
                    circuit.swap(6, 7)

            case ABEncodingType.AB:
                if self.reflect_in_dim[0]:
                    # 1 <-> 3
                    circuit.x([0, 1])
                    circuit.mcx([0, 1, 3], 2)
                    circuit.x([0, 1])

                    # 5 <-> 6
                    circuit.x(0)
                    circuit.cx(3, 2)
                    circuit.mcx([0, 1, 2], 3)
                    circuit.cx(3, 2)
                    circuit.x(0)

                    # 8 <-> 7
                    circuit.cx(0, 1)
                    circuit.cx(0, 2)
                    circuit.cx(0, 3)
                    circuit.mcx([1, 2, 3], 0)
                    circuit.cx(0, 3)
                    circuit.cx(0, 2)
                    circuit.cx(0, 1)
                if self.reflect_in_dim[1]:

                    # 2 <-> 4
                    circuit.x([0, 3])
                    circuit.cx(1, 2)
                    circuit.mcx([0, 2, 3], 1)
                    circuit.cx(1, 2)
                    circuit.x([0, 3])

                    # 5 <-> 8
                    circuit.x(2)
                    circuit.cx(0, 1)
                    circuit.cx(0, 3)
                    circuit.mcx([1, 2, 3], 0)
                    circuit.cx(0, 3)
                    circuit.cx(0, 1)
                    circuit.x(2)

                    # 6 <-> 7
                    circuit.x(0)
                    circuit.mcx([0, 1, 2], 3)
                    circuit.x(0)
            case _:
                raise LatticeException(f"Unsupported lattice encoding: {self.encoding}")

        return circuit.reverse_bits() if self.encoding == ABEncodingType.AB else circuit

    @override
    def __str__(self) -> str:
        return f"[Primitive ABSpecularReflectionPermutation with {self.num_qubits} qubits on {self.discretization}, reflection {self.reflect_in_dim}]"
