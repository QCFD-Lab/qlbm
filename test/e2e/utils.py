"""Shared utilities for end-to-end tests."""

from typing import Dict, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def run_statevector(circuit: QuantumCircuit) -> Statevector:
    """Run a circuit on AerSimulator and return the final statevector.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to simulate. A ``save_statevector`` instruction is
        appended automatically.

    Returns
    -------
    Statevector
        The resulting statevector.
    """
    qc = circuit.copy()
    qc.save_statevector()
    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim, optimization_level=0)
    result = sim.run(tqc).result()
    return result.data(0)["statevector"]


def get_nonzero_amplitudes(
    sv: Statevector, threshold: float = 1e-8
) -> Dict[int, complex]:
    """Return a dict mapping basis-state index to amplitude for nonzero entries.

    Parameters
    ----------
    sv : Statevector
        The statevector to inspect.
    threshold : float
        Amplitude magnitude below which entries are treated as zero.

    Returns
    -------
    Dict[int, complex]
        Mapping from statevector index to complex amplitude.
    """
    data = np.array(sv)
    nonzero_idx = np.where(np.abs(data) > threshold)[0]
    return {int(idx): complex(data[idx]) for idx in nonzero_idx}


def decode_state(
    index: int, qubit_ranges: Dict[str, Tuple[int, int]]
) -> Dict[str, int]:
    """Decode a statevector index into named register values.

    Parameters
    ----------
    index : int
        The statevector basis-state index.
    qubit_ranges : Dict[str, Tuple[int, int]]
        Mapping from register name to ``(start_qubit, num_qubits)``.

    Returns
    -------
    Dict[str, int]
        Mapping from register name to the integer value stored in that register.
    """
    result: Dict[str, int] = {}
    for name, (start, size) in qubit_ranges.items():
        value = 0
        for i in range(size):
            if index & (1 << (start + i)):
                value |= 1 << i
        result[name] = value
    return result


def make_ab_qubit_layout(lattice) -> Dict[str, Tuple[int, int]]:
    """Build a qubit-layout dictionary for an ABLattice.

    Parameters
    ----------
    lattice : ABLattice
        The lattice whose register structure to describe.

    Returns
    -------
    Dict[str, Tuple[int, int]]
        Mapping ``{register_name: (start_qubit, size)}``.
    """
    dim_names = ["g_x", "g_y", "g_z"]
    layout: Dict[str, Tuple[int, int]] = {}
    for dim in range(lattice.num_dims):
        layout[dim_names[dim]] = (
            lattice.grid_index(dim)[0],
            len(lattice.grid_index(dim)),
        )
    layout["v"] = (lattice.velocity_index()[0], lattice.num_velocity_qubits)
    if lattice.num_comparator_qubits > 0:
        layout["a_c"] = (
            lattice.ancillae_comparator_index()[0],
            lattice.num_comparator_qubits,
        )
    layout["a_o"] = (
        lattice.ancillae_obstacle_index()[0],
        lattice.num_obstacle_qubits,
    )
    return layout


def make_ms_qubit_layout(lattice) -> Dict[str, Tuple[int, int]]:
    """Build a qubit-layout dictionary for an MSLattice.

    Parameters
    ----------
    lattice : MSLattice
        The lattice whose register structure to describe.

    Returns
    -------
    Dict[str, Tuple[int, int]]
        Mapping ``{register_name: (start_qubit, size)}``.
    """
    dim_names = ["g_x", "g_y", "g_z"]
    layout: Dict[str, Tuple[int, int]] = {}
    layout["a_v"] = (
        lattice.ancillae_velocity_index()[0],
        len(lattice.ancillae_velocity_index()),
    )
    layout["a_o"] = (
        lattice.ancillae_obstacle_index()[0],
        len(lattice.ancillae_obstacle_index()),
    )
    if lattice.ancillae_comparator_index():
        layout["a_c"] = (
            lattice.ancillae_comparator_index()[0],
            len(lattice.ancillae_comparator_index()),
        )
    for dim in range(lattice.num_dims):
        layout[dim_names[dim]] = (
            lattice.grid_index(dim)[0],
            len(lattice.grid_index(dim)),
        )
    for dim in range(lattice.num_dims):
        vi = lattice.velocity_index(dim)
        if vi:
            layout[f"v_{dim_names[dim][-1]}"] = (vi[0], len(vi))
    for dim in range(lattice.num_dims):
        layout[f"vd_{dim_names[dim][-1]}"] = (
            lattice.velocity_dir_index(dim)[0],
            len(lattice.velocity_dir_index(dim)),
        )
    return layout


def prepare_single_particle(
    lattice, grid_pos: Tuple[int, ...], velocity_channel: int
) -> QuantumCircuit:
    """Prepare a circuit with one particle at a specific position and velocity.

    Parameters
    ----------
    lattice : ABLattice | MSLattice
        The lattice that defines the register layout.
    grid_pos : Tuple[int, ...]
        Grid coordinates, one per dimension.
    velocity_channel : int
        Integer index of the velocity channel (binary-encoded into velocity qubits).

    Returns
    -------
    QuantumCircuit
        A circuit that prepares the desired initial state.
    """
    circuit = QuantumCircuit(*lattice.registers)
    for dim, pos in enumerate(grid_pos):
        for i in range(lattice.num_gridpoints[dim].bit_length()):
            if (pos >> i) & 1:
                circuit.x(lattice.grid_index(dim)[i])
    for i in range(lattice.num_velocity_qubits):
        if (velocity_channel >> i) & 1:
            circuit.x(lattice.velocity_index()[i])
    return circuit


def prepare_ms_particle(
    lattice,
    grid_pos: Tuple[int, ...],
    velocity_mag: Tuple[int, ...],
    velocity_dir: Tuple[int, ...],
) -> QuantumCircuit:
    """Prepare a circuit with one MS particle at a given position and velocity.

    Parameters
    ----------
    lattice : MSLattice
        The lattice that defines the register layout.
    grid_pos : Tuple[int, ...]
        Grid coordinates, one per dimension.
    velocity_mag : Tuple[int, ...]
        Velocity magnitude per dimension (binary-encoded).
    velocity_dir : Tuple[int, ...]
        Velocity direction per dimension (1 = positive, 0 = negative).

    Returns
    -------
    QuantumCircuit
        A circuit that prepares the desired initial state.
    """
    circuit = QuantumCircuit(*lattice.registers)
    for dim, pos in enumerate(grid_pos):
        for i in range(len(lattice.grid_index(dim))):
            if (pos >> i) & 1:
                circuit.x(lattice.grid_index(dim)[i])
    for dim in range(lattice.num_dims):
        for i in range(len(lattice.velocity_index(dim))):
            if (velocity_mag[dim] >> i) & 1:
                circuit.x(lattice.velocity_index(dim)[i])
        if velocity_dir[dim]:
            circuit.x(lattice.velocity_dir_index(dim)[0])
    return circuit
