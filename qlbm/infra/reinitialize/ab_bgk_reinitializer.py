r""":class:`.ABBGKQLBM`-specific implementation of the :class:`.Reinitializer`."""

from logging import Logger, getLogger
from typing import Dict, Tuple

import numpy as np
from qiskit import QuantumCircuit as QiskitQC
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector
from qiskit.result import Counts
from qiskit_aer.backends.aer_simulator import AerBackend
from qulacs import QuantumCircuit as QulacsQC
from typing_extensions import override

from qlbm.components.ab.collision.initial import ABBGKInitialConditions
from qlbm.infra.compiler import CircuitCompiler
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.lattices.ab_bgk_lattice import ABBGKLattice
from qlbm.tools.exceptions import ExecutionException

from .base import Reinitializer


class ABBGKReinitializer(Reinitializer):
    r"""
    :class:`.ABBGKQLBM`-specific implementation of the :class:`.Reinitializer`.

    The angle-encoded collision consumes a branch--angle state and produces populations,
    so its output is not a valid input for the next time step. This reinitializer closes
    the loop by decoding the state and re-encoding the flow field that it describes:

    #. Read the probabilities of the marker-one basis states of every grid point.
    #. Undo the normalizations of the encoding to recover the populations,
       :math:`f_i = \alpha \lVert \rho \rVert_2 \sqrt{P_i}`.
    #. Take the macroscopic moments :math:`\rho` and :math:`\mathbf{u}` of those populations.
    #. Prepare fresh :class:`.ABBGKInitialConditions` from them.

    Step three is where the nonlinearity of the method lives: the equilibrium is a
    quadratic function of a velocity that is itself a ratio of moments, and recomputing
    it every step is what keeps the collision exact rather than linearized. The price is
    that the loop is hybrid, and that a copy of the statevector is required.

    Reading probabilities recovers :math:`\lvert f_i \rvert` rather than :math:`f_i`,
    which is valid while every population stays positive. The velocity bound of the
    angle encoding keeps simulations well inside that regime.

    Because amplitudes are normalized, the absolute scale of the density is tracked
    classically in :attr:`.ABBGKLattice.density_norm`, which
    :class:`.ABBGKInitialConditions` updates whenever a field is encoded. Velocities are
    independent of that scale, so an inaccurate norm shifts reported densities by a
    constant factor without changing the simulated dynamics at all.

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`lattice`
          - The :class:`.ABBGKLattice` of the simulated system.
        * - :attr:`compiler`
          - The compiler that converts the novel initial conditions circuits.
        * - :attr:`logger`
          - The performance logger, by default ``getLogger("qlbm")``.
        * - :attr:`solid_mask`
          - A Boolean mask marking solid gridpoints, by default inferred from the lattice geometry.
    """

    lattice: ABBGKLattice
    """The lattice of the simulated system."""

    solid_mask: np.typing.NDArray[np.bool_]
    """The gridpoints occupied by solid geometry, which hold no fluid."""

    populations: np.typing.NDArray[np.float64]
    """The populations decoded at the most recent reinitialization, of shape ``(nx, ny, 9)``."""

    steps_reinitialized: int
    """The number of time steps this reinitializer has already prepared a state for.

    A runner evaluates the initial time step without applying the algorithm, so the
    state handed over by that first call has not been collided yet and is passed
    through unchanged. Every later call decodes a genuine post-collision state."""

    diagnostics: Dict[str, float]
    r"""Sanity metrics of the most recent reinitialization.

    .. list-table:: Entries
        :widths: 35 55
        :header-rows: 1

        * - Key
          - Description
        * - ``physical_sector_probability``
          - The probability of measuring a physical population, the rest being held by the auxiliary sector.
        * - ``unused_velocity_probability``
          - Probability leaked into the seven marker-one states without a physical meaning. Should be zero.
        * - ``ancilla_probability``
          - Probability left on a boundary-condition ancilla. Should be zero.
        * - ``solid_population``
          - The largest population found inside solid geometry. Should be zero.
    """

    def __init__(
        self,
        lattice: ABBGKLattice,
        compiler: CircuitCompiler,
        logger: Logger = getLogger("qlbm"),
        solid_mask: np.typing.NDArray[np.bool_] | None = None,
    ) -> None:
        super().__init__(lattice, compiler, logger)
        self.lattice = lattice
        self.logger = logger

        self.solid_mask = (
            self.solid_mask_from_geometry(lattice) if solid_mask is None else solid_mask
        )
        self.populations = np.zeros(
            (
                lattice.num_gridpoints[0] + 1,
                lattice.num_gridpoints[1] + 1,
                lattice.encoding.NUM_POPULATIONS,
            )
        )
        self.diagnostics = {}
        self.steps_reinitialized = 0

        # The index tables only depend on the register layout, so they are built once.
        self.physical_indices = lattice.physical_basis_indices()
        self.unused_indices = lattice.unused_basis_indices()
        self.ancilla_mask = lattice.ancilla_basis_mask()

    @override
    def reinitialize(
        self,
        statevector: Statevector,
        counts: Counts,
        backend: AerBackend | None = None,
        optimization_level: int = 0,
    ) -> QiskitQC | QulacsQC:
        """
        Decodes the flow field held by the statevector and re-encodes it for the next time step.

        Parameters
        ----------
        statevector : qiskit.quantum_info.Statevector
            The statevector at the end of the simulated time step.
        counts : qiskit.result.Counts
            Ignored. Populations are read from amplitudes rather than sampled, since
            reconstructing a velocity field requires the joint grid and velocity
            distribution rather than the grid marginal.
        backend : qiskit_aer.backends.aer_simulator.AerBackend | None, optional
            The backend to compile the new initial conditions to, by default ``None``.
        optimization_level : int, optional
            The optimization level to pass to the circuit compiler, by default 0.

        Returns
        -------
        qiskit.QuantumCircuit | qulacs.QuantumCircuit
            The compiled initial conditions circuit of the next time step.
        """
        if self.steps_reinitialized == 0:
            # The time step circuit has not been applied yet, so the flow field already
            # encoded in this state is exactly the one the next step must start from.
            self.steps_reinitialized += 1
            circuit = self.lattice.circuit.copy()
            circuit.compose(
                Initialize(statevector, normalize=True),
                qubits=range(circuit.num_qubits),
                inplace=True,
            )

            return self.compiler.compile(
                circuit,
                backend=backend,
                optimization_level=optimization_level,
            )

        density, velocity_x, velocity_y = self.decode(statevector)
        self.steps_reinitialized += 1

        return self.compiler.compile(
            ABBGKInitialConditions(
                self.lattice,
                density,
                velocity_x,
                velocity_y,
                logger=self.logger,
            ),
            backend=backend,
            optimization_level=optimization_level,
        )

    def decode(
        self,
        statevector: Statevector,
    ) -> Tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ]:
        r"""
        Recovers the macroscopic flow field encoded in a statevector.

        Also refreshes :attr:`populations` and :attr:`diagnostics`, which makes this the
        entry point for inspecting a simulation without re-deriving the register layout.

        Parameters
        ----------
        statevector : qiskit.quantum_info.Statevector
            The statevector to decode.

        Returns
        -------
        Tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
            The density and the two velocity components, each of shape ``(nx, ny)``.
        """
        probabilities = np.abs(np.asarray(statevector.data)) ** 2

        populations = self.lattice.encoding.populations_from_probabilities(
            probabilities[self.physical_indices],
            self.lattice.density_norm,
        )

        self.diagnostics = {
            "physical_sector_probability": float(
                np.sum(probabilities[self.physical_indices])
            ),
            "unused_velocity_probability": float(
                np.sum(probabilities[self.unused_indices])
            ),
            "ancilla_probability": float(np.sum(probabilities[self.ancilla_mask])),
            "solid_population": float(
                np.max(np.abs(populations[self.solid_mask, :]), initial=0.0)
            ),
        }

        # Solid gridpoints hold no fluid. Exact reflection leaves them empty, so this
        # only discards floating-point residue before it reaches the moments.
        populations[self.solid_mask, :] = 0.0
        self.populations = populations

        return self.lattice.encoding.macroscopic(populations, self.solid_mask)

    @override
    def requires_statevector(self) -> bool:
        return True

    @override
    def requires_statevector_snapshots(self) -> bool:
        return True

    @staticmethod
    def solid_mask_from_geometry(
        lattice: ABBGKLattice,
    ) -> np.typing.NDArray[np.bool_]:
        r"""
        Builds the mask of solid gridpoints from the geometry of a lattice.

        Parameters
        ----------
        lattice : ABBGKLattice
            The lattice whose geometry to convert.

        Returns
        -------
        numpy.typing.NDArray[numpy.bool\_]
            The mask, of shape ``(nx, ny)``.

        Raises
        ------
        ExecutionException
            If the lattice contains a shape other than a :class:`.Block`, which is the
            only geometry that amplitude-based :math:`D_2Q_9` reflection supports.
        """
        mask = np.zeros(
            (lattice.num_gridpoints[0] + 1, lattice.num_gridpoints[1] + 1),
            dtype=bool,
        )

        for shape in lattice.shape_list:
            if not isinstance(shape, Block):
                raise ExecutionException(
                    f"Cannot infer a solid mask for shape {shape} of type {type(shape)}. "
                    "Pass an explicit solid_mask to the reinitializer."
                )

            (x_lower, x_upper), (y_lower, y_upper) = shape.bounds
            mask[x_lower : x_upper + 1, y_lower : y_upper + 1] = True

        return mask
