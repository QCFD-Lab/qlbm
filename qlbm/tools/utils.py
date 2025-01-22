"""General qlbm utilities."""

import re
from math import pi
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pytket.extensions.qiskit import qiskit_to_tk
from pytket.extensions.qulacs import tk_to_qulacs
from qiskit import QuantumCircuit as QiskitQC
from qiskit.qasm2 import dumps
from qulacs import QuantumCircuit as QulacsQC
from qulacs.converter import convert_QASM_to_qulacs_circuit


def create_directory_and_parents(directory: str) -> None:
    """
    Creates a given directory and all its parent directories.

    Parameters
    ----------
    directory : str
        The fully specified location of the directory.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def qiskit_circuit_to_qulacs(circuit: QiskitQC) -> QulacsQC:
    """Converts a Qiskit QuantumCircuit to a Qulacs QuantumCircuit.

    Conversion takes place by first converting the Qiskit circuit into
    its OpenQASM 2 representation. This representation is then parsed
    into a Qulacs circuit.

    Parameters
    ----------
    circuit (QiskitQC): The Qiskit QuantumCircuit to convert.

    Returns
    -------
    QulacsQC: The Qulacs counterpart to the QuantumCircuit.
    """
    formatted_qulacs_qasm_circuit = evaluate_qasm_rotation_string(dumps(circuit))
    return convert_QASM_to_qulacs_circuit(formatted_qulacs_qasm_circuit.split("\n"))


def flatten(xss):
    """
    Flattens nested lists by one level.

    Parameters
    ----------
    xss : List[List[Any]]
        A nested list.

    Returns
    -------
    List[Any]
        The input flattened to by one level.
    """
    return [x for xs in xss for x in xs]


def bit_value(num: int, position: int) -> int:
    """
    The value of the bit at a given position in a given integer.

    Parameters
    ----------
    num : int
        The number.
    position : int
        The position in the binary representation.

    Returns
    -------
    int
        The velut of the target bit.
    """
    return (num & (1 << position)) >> position


def evaluate_qasm_rotation_string(qasm_repr: str) -> str:
    """
    Evaluate the symbolic values in a given qasm representation numerically.

    Used a hacky way to convert between Qiskit and Qulacs representations.

    Parameters
    ----------
    qasm_repr : str
        A QASM string possibly containing symbolic values.

    Returns
    -------
    str
        The QASM string with symbolic values evaluated numerically.
    """
    return re.sub(
        r"(r[xyz]|p)(\([-]?[\d]*[*]?pi[/]?[\d]*\))",  # Replace all  (r[xyz]/p(...)) rotation matrix notations by evaluating occurrances of pi
        lambda m: f"{m.group(1)}({str(eval(m.group(2).replace('pi', str(pi))))})",  # Replace pi by the numeric value and evaluate
        qasm_repr,
    )


def get_circuit_properties(circuit: QiskitQC | QulacsQC) -> Tuple[str, int, int, int]:
    """Gets the static properties of a quantum circuit.

    Args:
        circuit (QiskitQC | QulacsQC): The circuit for which to compile the properties.

    Returns
    -------
        Tuple[str, int, int, int]: The circuit's platform, the number of qubits, depth, and number of gates of the circuit.
    """
    if isinstance(circuit, QiskitQC):
        return (
            "QISKIT",
            int(circuit.num_qubits),
            int(circuit.depth()),
            sum(circuit.count_ops().values()),
        )
    else:
        return (
            "QULACS",
            circuit.get_qubit_count(),
            circuit.calculate_depth(),
            circuit.get_gate_count(),
        )


def qiskit_to_qulacs(circuit: QiskitQC) -> QulacsQC:
    """
    Converts a Qiskit quantum circuit to a Qulacs quantum circuit equivalent using Tket.

    Parameters
    ----------
    circuit : QiskitQC
        An arbitrary Qiskit circuit.

    Returns
    -------
    QulacsQC
        The equivalent Qulacs circuit, if compatible.
    """
    return tk_to_qulacs(qiskit_to_tk(circuit))


def dimension_letter(dim: int) -> str:
    """Maps [0, 1, 2] to [x, y, z].

    Parameters
    ----------
    dim (int): The dimension to represent.

    Returns
    -------
        str: The letter associated with the dimension.
    """
    return chr(120 + dim)


def is_two_pow(num: int) -> bool:
    """Whether a number is a power of 2.

    Parameters
    ----------
    num (int): The number to evaluate.

    Returns
    -------
        bool: Whether the input is a power of 2.
    """
    # Powers of two have only one bit equal to 1.
    # Subtracting 1 flips all bits in this case.
    # The logical and between the two numbers is then 0.
    return (num & (num - 1) == 0) and num != 0


def get_time_series(
    num_discrete_velocities: int,
    max_allowed_iters: int = 10000,
    tolerance: float = 1e-6,
) -> List[List[int]]:
    """
    Compute a time series of the streaming velocities for a given number of discrete velocities.

    Parameters
    ----------
    num_discrete_velocities : int
        The number of discrete velocities in the time series.
    max_allowed_iters : int, optional
        The number of iterations before truncating the time series, by default 10000
    tolerance : float, optional
        The level of precision required to truncate the time sries, by default 1e-6

    Returns
    -------
    List[List[int]]
        The order in which velocities have to be streamed before all particles have reached discrete gridpoints exactly
    """
    # Used for debugging
    steps = 0

    # The order in which the velocities control the streaming incrementation
    speed_controls = []

    # Ignore the directional velocity qubit
    num_velocity_magnitudes = num_discrete_velocities // 2 + (
        num_discrete_velocities % 2
    )

    # Track the "progress" of each velocity towards the next grid point
    cfl_counter = np.zeros(num_velocity_magnitudes)

    # Evenly spaced velocity magnitudes
    velocity_magnitudes = np.arange(0.5, num_discrete_velocities / 2 + 0.5, 1)
    inverse_velocities = np.reciprocal(velocity_magnitudes)
    ones = np.ones(num_velocity_magnitudes)

    for _ in range(max_allowed_iters):
        # The "ground" covered by each velocity magnitude after this step
        distance_covered = np.multiply(ones - cfl_counter, inverse_velocities)

        # The minimum of this list dictates which velocity is closest to the next grid point
        min_velocity = np.argmin(distance_covered)

        # Update the progress of each velocity
        cfl_counter += distance_covered[min_velocity] * velocity_magnitudes

        # Get the indices of the velocities that have reached the next grid point
        streamed_velocity_indices = np.squeeze(
            np.argwhere(np.isclose(cfl_counter, 1.0, tolerance)), axis=1
        )

        # Reset the progress of the velocities streamed at this time step
        cfl_counter[streamed_velocity_indices] = 0

        # Track the controlled velocities
        speed_controls.append(streamed_velocity_indices.tolist())

        steps += 1

        # If all velocities have been streamed, break
        if (
            np.count_nonzero(
                np.isclose(cfl_counter, 1.0, tolerance)
                | np.isclose(cfl_counter, 0.0, tolerance)
            )
            == num_velocity_magnitudes
        ):
            break

    return speed_controls  # type: ignore


def get_qubits_to_invert(gridpoint_encoded: int, num_qubits: int) -> List[int]:
    r"""
    Returns the indices at which a given gridpoint encoded value has a 0. Inverting these indices results in a :math:`\ket{1}^{\otimes n}` state.

    Parameters
    ----------
    gridpoint_encoded : int
        The integer representation of the gridpoint.
    num_qubits : int
        The total nuimber of grid qubits.

    Returns
    -------
    List[int]
        The indices of the (qu)bits that have value 0.
    """
    return [i for i in range(num_qubits) if not bit_value(gridpoint_encoded, i)]
