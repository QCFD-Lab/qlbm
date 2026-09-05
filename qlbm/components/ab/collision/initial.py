r"""Quantum circuits used for setting the initial state in the :class:`.ABBGKQLBM` algorithm."""

from logging import Logger, getLogger
from time import perf_counter_ns

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Initialize
from typing_extensions import override

from qlbm.components.base import LBMPrimitive
from qlbm.lattice.lattices.ab_bgk_lattice import ABBGKLattice
from qlbm.tools.exceptions import CircuitException

AMPLITUDE_TOLERANCE = 1e-14
"""Amplitudes below this fraction of the largest one are rounded down to zero."""


class ABBGKInitialConditions(LBMPrimitive):
    r"""
    Initial conditions for the :class:`.ABBGKQLBM` algorithm.

    Encodes a macroscopic flow field, rather than a set of occupied velocity channels:
    every grid point receives the branch--angle state that matches its local velocity,
    weighted by its local density,

    .. math::
        \ket{\Psi} =
        \sum_{\mathbf{x}}
        \frac{\rho(\mathbf{x})}{\lVert \rho \rVert_2}
        \ket{\mathbf{x}}_G \ket{\psi(\theta_x(\mathbf{x}), \theta_y(\mathbf{x}); \beta)}.

    Applying :class:`.ABBGKCollisionOperator` to this state turns each local
    branch--angle state into the local equilibrium populations, which is what makes the
    collision act on the whole lattice with a single five-qubit gate. The density
    normalization is recorded in :attr:`density_norm` and is needed to restore the
    physical scale of the populations after measurement.

    .. note::
        The state is assembled classically and handed to Qiskit as an ``Initialize``
        instruction. Preparing an arbitrary flow field is a state-preparation problem in
        its own right, and is deliberately kept outside the scope of the algorithm being
        simulated here. This mirrors how ``qlbm`` resumes a simulation from a statevector
        in :class:`.IdentityReinitializer`, and leaves the collision, streaming, and
        reflection circuits as the only components under test.

    Example usage:

    .. testcode::

        import numpy as np

        from qlbm.components.ab.collision import ABBGKInitialConditions
        from qlbm.lattice import ABBGKLattice

        lattice = ABBGKLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        # A uniform flow along the positive x axis.
        initial_conditions = ABBGKInitialConditions(
            lattice,
            density=np.ones((4, 4)),
            velocity_x=np.full((4, 4), 0.05),
            velocity_y=np.zeros((4, 4)),
        )

        print(np.isclose(np.linalg.norm(initial_conditions.statevector), 1.0))

    .. testoutput::

        True

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`lattice`
          - The :class:`.ABBGKLattice` based on which the properties of the component are inferred.
        * - :attr:`density`
          - The density field, of shape ``(num_x_gridpoints, num_y_gridpoints)``.
        * - :attr:`velocity_x`
          - The :math:`x` velocity field, of the same shape as the density.
        * - :attr:`velocity_y`
          - The :math:`y` velocity field, of the same shape as the density.
        * - :attr:`logger`
          - The performance logger, by default ``getLogger("qlbm")``.
    """

    lattice: ABBGKLattice
    """The lattice to construct the component for."""

    density: np.typing.NDArray[np.float64]
    r"""The encoded density field :math:`\rho`."""

    velocity_x: np.typing.NDArray[np.float64]
    r"""The encoded :math:`x` velocity field."""

    velocity_y: np.typing.NDArray[np.float64]
    r"""The encoded :math:`y` velocity field."""

    density_norm: float
    r"""The Euclidean norm :math:`\lVert \rho \rVert_2` removed by the encoding.

    Amplitudes are normalized, so the absolute scale of the density has to be stored
    classically and reapplied when populations are read back out. Constructing this
    component publishes the norm to :attr:`.ABBGKLattice.density_norm`, which is where
    :class:`.ABBGKReinitializer` picks it up. See
    :meth:`.D2Q9AngleEncoding.populations_from_probabilities`."""

    statevector: np.typing.NDArray[np.complex128]
    """The prepared statevector, over all qubits of the lattice."""

    def __init__(
        self,
        lattice: ABBGKLattice,
        density: np.typing.NDArray[np.float64],
        velocity_x: np.typing.NDArray[np.float64],
        velocity_y: np.typing.NDArray[np.float64],
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(logger)
        self.lattice = lattice
        self.density = np.asarray(density, dtype=float)
        self.velocity_x = np.asarray(velocity_x, dtype=float)
        self.velocity_y = np.asarray(velocity_y, dtype=float)

        expected_shape = (
            lattice.num_gridpoints[0] + 1,
            lattice.num_gridpoints[1] + 1,
        )

        for name, field in (
            ("density", self.density),
            ("velocity_x", self.velocity_x),
            ("velocity_y", self.velocity_y),
        ):
            if field.shape != expected_shape:
                raise CircuitException(
                    f"The {name} field must have shape {expected_shape} to match the "
                    f"lattice, got {field.shape}."
                )

        if np.any(self.density < 0.0):
            raise CircuitException(
                "The density field must be non-negative, since the encoding stores its "
                "square root as an amplitude."
            )

        self.density_norm = float(np.linalg.norm(self.density.ravel()))

        if self.density_norm == 0.0:
            raise CircuitException("The density field must not be identically zero.")

        self.statevector = self.__create_statevector()

        # Publish the scale that was divided out, so that whoever decodes the state at
        # the end of the time step can restore it.
        lattice.density_norm = self.density_norm

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self) -> QuantumCircuit:
        circuit = self.lattice.circuit.copy()

        circuit.compose(
            Initialize(self.statevector, normalize=True),
            qubits=range(circuit.num_qubits),
            inplace=True,
        )

        return circuit

    def __create_statevector(self) -> np.typing.NDArray[np.complex128]:
        r"""
        Assembles the branch--angle statevector of the whole lattice.

        Every grid point contributes the 32 amplitudes of its own branch--angle state,
        scaled by its share of the density field. All other qubits, including the
        boundary-condition ancillae, are left in :math:`\ket{0}`.

        Returns
        -------
        numpy.typing.NDArray[numpy.complex128]
            The normalized statevector.
        """
        angle_x, angle_y = self.lattice.encoding.angles(
            self.velocity_x, self.velocity_y
        )

        statevector = np.zeros(1 << self.lattice.circuit.num_qubits, dtype=complex)
        grid_offsets = self.lattice.grid_basis_offsets()
        collision_offsets = self.lattice.collision_basis_offsets()

        for x in range(self.density.shape[0]):
            for y in range(self.density.shape[1]):
                statevector[int(grid_offsets[x, y]) | collision_offsets] = (
                    self.density[x, y] / self.density_norm
                ) * self.lattice.encoding.branch_state(
                    float(angle_x[x, y]), float(angle_y[x, y])
                )

        norm = float(np.linalg.norm(statevector))

        if not np.isclose(norm, 1.0, atol=1e-10):
            raise CircuitException(
                f"The prepared branch-angle state is not normalized, with norm {norm}."
            )

        # Branches that carry a product of two features can end up many orders of
        # magnitude below the rest of the state when both velocity components are close
        # to zero. Those amplitudes are below the resolution of the state as a whole, and
        # the dynamic range they introduce defeats the isometry decompositions that
        # generic state-preparation synthesis relies on.
        statevector[
            np.abs(statevector) < AMPLITUDE_TOLERANCE * np.max(np.abs(statevector))
        ] = 0.0

        statevector /= np.linalg.norm(statevector)

        return statevector

    @override
    def __str__(self) -> str:
        return f"[Primitive ABBGKInitialConditions with lattice {self.lattice}]"
