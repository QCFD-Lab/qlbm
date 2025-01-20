"""Base classes for quantum primitives, operators, and algorithms."""

from abc import ABC, abstractmethod
from io import TextIOBase
from logging import Logger, getLogger

from qiskit import QuantumCircuit
from qiskit.qasm2 import dump as dump_qasm2
from qiskit.qasm3 import dump as dump_qasm3
from typing_extensions import override

from qlbm.lattice import CollisionlessLattice, Lattice
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice


class QuantumComponent(ABC):
    """
    Base class for all quantum circuits implementing QLBM functionality.

    This class wraps a :class:`qiskit.QuantumCircuit` object constructed
    through the parameters supplied to the constructor.
    The :meth:`create_circuit` is automatically called at construct time
    and its output is stored in the `circuit` attribute.
    All quantum components have an implementation of the :meth:`create_circuit` method
    which builds their specialized quantum circuits.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`circuit`           The :class:`.qiskit.QuantumCircuit` of the component.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``
    ========================= ======================================================================
    """

    circuit: QuantumCircuit
    logger: Logger

    def __init__(
        self,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__()
        self.logger = logger

    @abstractmethod
    def create_circuit(self) -> QuantumCircuit:
        """
        Creates the :class:`qiskit.QuantumCircuit` of this object.

        This method is called automatically at construction time for all quantum components.

        Returns
        -------
        QuantumCircuit
            The generated QuantumCircuit.
        """
        pass

    @override
    def __repr__(self) -> str:
        return self.circuit.__repr__()

    @abstractmethod
    def __str__(self) -> str:
        """
        The string representation of a quantum component.

        Returns
        -------
        str
            The string representation of a quantum component.
        """
        return self.circuit.__str__()

    def width(self) -> int:
        """
        Return the number of qubits plus clbits in the circuit.

        Returns
        -------
        int
            Width of circuit.
        """
        return self.circuit.width()

    def size(self) -> int:
        """
        Returns the total number of instructions (gates) in the circuit.

        Returns
        -------
        int
            The total number of gates in the circuit.
        """
        return self.circuit.size()

    def dump_qasm3(self, stream: TextIOBase) -> None:
        """
        Serialize to QASM3.

        Parameters
        ----------
        stream : TextIOBase
            The stream to output to.
        """
        return dump_qasm3(self.circuit, stream)

    def dump_qasm2(self, stream: TextIOBase) -> None:
        """
        Serialize to QASM2.

        Parameters
        ----------
        stream : TextIOBase
            The stream to output to.
        """
        return dump_qasm2(self.circuit, stream)

    def draw(self, output: str, filename: str | None = None):  # type: ignore
        """
        Draw the circuit to matplotlib, ASCII, or Latex representations.

        Parameters
        ----------
        output : str
            The format of the output. Use "text", "mpl", or "texsource", respectively.
        filename : str | None, optional
            The file to write the output to, by default None.
        """
        return self.circuit.draw(output=output, filename=filename)  # type: ignore


class LBMPrimitive(QuantumComponent):
    """
    Base class for all primitive-level quantum components.

    A primitive component is a small, isolated, and structurally parameterizable
    quantum circuit that can be reused throughout one or multiple algorithms.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`circuit`           The :class:`.qiskit.QuantumCircuit` of the primitive.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``
    ========================= ======================================================================
    """

    logger: Logger

    def __init__(
        self,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)


class LBMOperator(QuantumComponent):
    """
    Base class for all operator-level quantum components.

    An operator component implements a specific physical operation
    corresponding to the classical LBM (streaming, collision, etc.).
    Operators are inferred based on the structure of a :class:`.Lattice`
    object of an appropriate encoding.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`circuit`           The :class:`.qiskit.QuantumCircuit` of the operator.
    :attr:`lattice`           The :class:`.Lattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``
    ========================= ======================================================================
    """

    lattice: Lattice

    def __init__(
        self,
        lattice: Lattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice


class CQLBMOperator(LBMOperator):
    """
    Specialization of the :class:`.LBMOperator` operator class for the Collisionless Quantum Transport Method algorithm by :cite:t:`collisionless`.

    Specializaitons of this class infer their properties
    based on a :class:`.CollisionlessLattice`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`circuit`           The :class:`.qiskit.QuantumCircuit` of the operator.
    :attr:`lattice`           The :class:`.CollisionlessLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``
    ========================= ======================================================================
    """

    lattice: CollisionlessLattice

    def __init__(
        self,
        lattice: CollisionlessLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice


class SpaceTimeOperator(LBMOperator):
    """
    Specialization of the :class:`.LBMOperator` operator class for the Space-Time QBM algorithm by :cite:t:`spacetime`.

    Specializaitons of this class infer their properties
    based on a :class:`.SpaceTimeLattice`.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`circuit`           The :class:`.qiskit.QuantumCircuit` of the operator.
    :attr:`lattice`           The :class:`.SpaceTimeLattice` based on which the properties of the operator are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``
    ========================= ======================================================================
    """

    lattice: SpaceTimeLattice

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice = lattice


class LBMAlgorithm(QuantumComponent):
    """
    Base class for all end-to-end Quantum Boltzmann Methods.

    An end-to-end algorithm consists of a
    series of :class:`.LBMOperator` that perform
    the physical operations of the appropriate algorithm.

    ========================= ======================================================================
    Attribute                  Summary
    ========================= ======================================================================
    :attr:`circuit`           The :class:`.qiskit.QuantumCircuit` of the algorithm.
    :attr:`lattice`           The :class:`.Lattice` based on which the properties of the algorithm are inferred.
    :attr:`logger`            The performance logger, by default ``getLogger("qlbm")``
    ========================= ======================================================================
    """

    lattice: Lattice

    def __init__(
        self,
        lattice: Lattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice
