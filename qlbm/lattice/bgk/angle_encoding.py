r"""Angle encoding of the :math:`D_2Q_9` BGK equilibrium and its unitary collision operator."""

import json
from importlib import resources
from logging import Logger, getLogger
from typing import List, Tuple

import numpy as np

from qlbm.lattice.spacetime.properties_base import (
    LatticeDiscretization,
    LatticeDiscretizationProperties,
)
from qlbm.tools.exceptions import LatticeException

PARAMETER_DIRECTORY = "data"
"""The subdirectory of this module that holds the packaged encoding parameters."""

DEFAULT_PARAMETER_FILE = "d2q9_angle_encoding.json"
"""The name of the packaged file that holds the optimized branch amplitudes."""


class D2Q9AngleEncoding:
    r"""
    Classical model of the angle-encoded :math:`D_2Q_9` BGK collision at :math:`\tau=1`.

    This class holds all of the *classical* linear algebra behind the
    :class:`.ABBGKCollisionOperator`. It is deliberately free of Qiskit objects so that
    it can be used both to synthesize quantum circuits and to post-process
    measurement outcomes.

    Encoding
    ^^^^^^^^

    Macroscopic velocities are mapped onto two angles through

    .. math::
        u_x = U \sin \theta_x, \quad u_y = U \sin \theta_y,

    with :math:`U` the :attr:`max_velocity` bound of the encoding.
    Under this substitution, the normalized :math:`D_2Q_9` equilibrium is *exactly*
    linear in the six features

    .. math::
        \varphi(\theta_x, \theta_y) =
        \begin{bmatrix}
        1 &
        \sin\theta_x &
        \sin\theta_y &
        \cos 2\theta_x &
        \cos 2\theta_y &
        \sin\theta_x \sin\theta_y
        \end{bmatrix}^T,

    such that :math:`p^{eq} = C \varphi` for the :math:`9 \times 6` matrix :math:`C`
    returned by :meth:`feature_matrix`.

    Branch--angle states
    ^^^^^^^^^^^^^^^^^^^^

    The six features are produced in superposition on five qubits: three *branch*
    qubits label the feature, and two *angle* qubits carry the rotations.
    Branch :math:`k` applies :math:`RY(2 m_x^k \theta_x)` and :math:`RY(2 m_y^k \theta_y)`
    to the two angle qubits, with the integer multipliers listed in
    :attr:`BRANCH_MULTIPLIERS`. Weighting branch :math:`k` by an amplitude
    :math:`\beta_k` gives the *branch--angle state*

    .. math::
        \ket{\psi(\theta_x, \theta_y; \beta)} =
        \sum_{k=0}^{5} \beta_k \ket{k}_B \ket{\varphi_k(\theta_x, \theta_y)},

    constructed by :meth:`branch_state`. Because every :math:`\ket{\varphi_k}` is
    produced by rotations, the state is normalized whenever
    :math:`\sum_k \beta_k^2 = 1`.

    Unitary collision
    ^^^^^^^^^^^^^^^^^

    Each feature appears in the branch--angle state scaled by :math:`\beta_k`, so
    dividing the columns of :math:`C` by the matching branch amplitude yields a sparse
    :math:`9 \times 32` matrix :math:`\tilde{C}(\beta)` (see :meth:`sparse_collision_map`)
    with

    .. math::
        \tilde{C}(\beta) \ket{\psi(\theta_x, \theta_y; \beta)} = p^{eq}

    for every angle pair. Only the action of :math:`\tilde{C}` on states that the
    circuit can actually prepare matters, so the map is restricted to an orthonormal
    basis :math:`B_{in}(\beta)` of the reachable subspace (:meth:`reachable_subspace_basis`)
    and normalized by :math:`\alpha = \lVert \tilde{C} B_{in} \rVert_2`
    (:attr:`alpha`). Padding the restricted map with the auxiliary block
    :math:`J = \sqrt{I - P^T P}` makes it an isometry, and completing both bases
    with QR decompositions gives the real orthogonal :math:`32 \times 32` matrix
    returned by :meth:`collision_unitary`.

    Register layout
    ^^^^^^^^^^^^^^^

    The same five qubits are reinterpreted before and after the unitary.
    Qiskit's little-endian convention applies throughout, so the first qubit of the
    list below is the least significant statevector bit.

    .. list-table:: Interpretation of the five collision qubits
        :widths: 15 40 40
        :header-rows: 1

        * - Qubit
          - Before the unitary
          - After the unitary
        * - 0
          - Angle qubit :math:`q_y`
          - Velocity bit 0
        * - 1
          - Angle qubit :math:`q_x`
          - Velocity bit 1
        * - 2
          - Branch bit 0
          - Velocity bit 2
        * - 3
          - Branch bit 1
          - Velocity bit 3
        * - 4
          - Branch bit 2
          - Physical-sector marker

    The marker separates the nine physical :math:`D_2Q_9` amplitudes,
    which occupy statevector indices :math:`16, \ldots, 24`, from the auxiliary
    amplitudes required by unitarity, which occupy indices :math:`0, \ldots, 15`.
    Because the marker is one of the same five qubits, no additional ancilla is needed.

    Reading out populations
    ^^^^^^^^^^^^^^^^^^^^^^^

    A physical basis state carries probability
    :math:`f_i^2 / (\alpha^2 \lVert \rho \rVert_2^2)`, so populations are recovered as
    :math:`f_i = \alpha \lVert \rho \rVert_2 \sqrt{P_i}` (:meth:`populations_from_probabilities`).
    This reconstruction returns :math:`\lvert f_i \rvert`; it is therefore valid only
    while every equilibrium population stays positive, which holds throughout the
    low-Mach regime that the :math:`U`-bounded angle encoding is restricted to anyway.

    Example usage:

    .. testcode::

        import numpy as np

        from qlbm.lattice.bgk import D2Q9AngleEncoding

        encoding = D2Q9AngleEncoding()

        # The collision map reproduces the D2Q9 equilibrium exactly.
        state = encoding.branch_state(*encoding.angles(0.05, -0.02))
        populations = encoding.sparse_collision_map() @ state

        print(np.allclose(populations, encoding.equilibrium(1.0, 0.05, -0.02)))

    .. testoutput::

        True

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`beta`
          - The six branch amplitudes. Defaults to the optimized values shipped with ``qlbm``.
        * - :attr:`max_velocity`
          - The velocity bound :math:`U` of the angle encoding, by default ``0.1``.
        * - :attr:`logger`
          - The performance logger, by default ``getLogger("qlbm")``.
    """

    NUM_POPULATIONS: int = 9
    r"""The number of :math:`D_2Q_9` populations."""

    NUM_BRANCHES: int = 6
    """The number of active branches, one for each feature of the equilibrium."""

    NUM_BRANCH_QUBITS: int = 3
    """The number of qubits that label the branches."""

    NUM_VELOCITY_QUBITS: int = 4
    r"""The number of qubits that encode the :math:`D_2Q_9` velocity index."""

    NUM_COLLISION_QUBITS: int = 5
    """The number of qubits that the collision unitary acts on."""

    NUM_STATES: int = 32
    """The dimension of the collision Hilbert space."""

    NUM_VELOCITY_STATES: int = 16
    """The number of basis states available in each marker sector."""

    WEIGHTS: np.typing.NDArray[np.float64] = np.array(
        [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=float
    )
    r"""The :math:`D_2Q_9` lattice weights, in the velocity order used by ``qlbm``."""

    BRANCH_MULTIPLIERS: Tuple[Tuple[int, int], ...] = (
        (0, 0),  # branch 0: the |00> amplitude is 1
        (1, 0),  # branch 1: the q_x = |1> amplitude is sin(theta_x)
        (0, 1),  # branch 2: the q_y = |1> amplitude is sin(theta_y)
        (1, 1),  # branch 3: the |11> amplitude is sin(theta_x) sin(theta_y)
        (2, 0),  # branch 4: the |00> amplitude is cos(2 theta_x)
        (0, 2),  # branch 5: the |00> amplitude is cos(2 theta_y)
    )
    r"""The :math:`(m_x, m_y)` rotation multipliers of each branch.

    Branch :math:`k` applies :math:`RY(2 m_x^k \theta_x)` to :math:`q_x` and
    :math:`RY(2 m_y^k \theta_y)` to :math:`q_y`. Since
    :math:`RY(2a)\ket{0} = \cos(a)\ket{0} + \sin(a)\ket{1}`, integer multipliers
    are enough to generate all six features of the equilibrium."""

    FEATURE_SLOTS: Tuple[Tuple[int, int], ...] = (
        (0, 0),  # 1
        (1, 2),  # sin(theta_x)
        (2, 1),  # sin(theta_y)
        (4, 0),  # cos(2 theta_x)
        (5, 0),  # cos(2 theta_y)
        (3, 3),  # sin(theta_x) sin(theta_y)
    )
    r"""The ``(branch, angle)`` position of each feature within the branch--angle state.

    Entries follow the column order of :meth:`feature_matrix`. The two-qubit angle
    index runs over :math:`\ket{q_x q_y}`, so the global statevector index of a slot
    is ``4 * branch + angle``."""

    beta: np.typing.NDArray[np.float64]
    """The branch amplitudes, one per feature of the equilibrium."""

    max_velocity: float
    r"""The velocity bound :math:`U` such that :math:`u_x = U \sin \theta_x`."""

    alpha: float
    r"""The restricted normalization :math:`\lVert \tilde{C} B_{in} \rVert_2`.

    Physical output amplitudes are scaled down by this factor, so a collided state keeps
    probability :math:`\lVert p^{eq} \rVert^2 / \alpha^2` in the physical sector.
    Because :math:`\alpha` is the operator norm over the reachable subspace, that
    probability never exceeds one, and the smallest admissible :math:`\alpha` is the one
    that leaves the most probability there. The branch amplitudes shipped with ``qlbm``
    were optimized for exactly that."""

    reachable_subspace_rank: int
    """The dimension of the subspace spanned by all branch--angle states."""

    logger: Logger
    """The performance logger."""

    def __init__(
        self,
        beta: np.typing.NDArray[np.float64] | List[float] | None = None,
        max_velocity: float | None = None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        default_beta, default_max_velocity = self.load_parameters()

        self.beta = np.asarray(
            default_beta if beta is None else beta,
            dtype=float,
        )
        self.max_velocity = (
            default_max_velocity if max_velocity is None else float(max_velocity)
        )
        self.logger = logger

        if self.beta.shape != (self.NUM_BRANCHES,):
            raise LatticeException(
                f"The angle encoding requires {self.NUM_BRANCHES} branch amplitudes, got shape {self.beta.shape}."
            )

        if np.any(self.beta <= 0.0):
            raise LatticeException(
                "Every branch amplitude must be strictly positive, since the sparse "
                "collision map divides the equilibrium coefficients by it."
            )

        if not np.isclose(np.sum(self.beta**2), 1.0, atol=1e-10):
            raise LatticeException(
                f"The branch amplitudes must be normalized, got a squared norm of {np.sum(self.beta**2)}."
            )

        if self.max_velocity <= 0.0:
            raise LatticeException(
                f"The velocity bound of the angle encoding must be positive, got {self.max_velocity}."
            )

        self.velocities = LatticeDiscretizationProperties.get_velocity_vectors(
            LatticeDiscretization.D2Q9
        ).astype(float)

        self.__unitary, self.alpha, self.reachable_subspace_rank = (
            self.__build_collision_unitary()
        )

    @staticmethod
    def load_parameters(
        file_name: str = DEFAULT_PARAMETER_FILE,
    ) -> Tuple[np.typing.NDArray[np.float64], float]:
        r"""
        Reads pre-optimized branch amplitudes from a file packaged with ``qlbm``.

        The branch amplitudes are the solution of an offline optimization that
        minimizes :math:`lpha(eta)` and therefore maximizes the probability of
        measuring the physical sector. That optimization is not part of ``qlbm``;
        only its result is distributed here.

        Parameters
        ----------
        file_name : str, optional
            The name of the file within ``qlbm.lattice.bgk.data``,
            by default :attr:`DEFAULT_PARAMETER_FILE`.

        Returns
        -------
        Tuple[numpy.typing.NDArray[numpy.float64], float]
            The branch amplitudes and the velocity bound they were optimized for.
        """
        with (
            resources.files("qlbm.lattice.bgk")
            .joinpath(PARAMETER_DIRECTORY, file_name)
            .open("r") as parameter_file
        ):
            parameters = json.load(parameter_file)

        return (
            np.asarray(parameters["beta"], dtype=float),
            float(parameters["max_velocity"]),
        )

    def angles(
        self,
        velocity_x: np.typing.NDArray[np.float64] | float,
        velocity_y: np.typing.NDArray[np.float64] | float,
    ) -> Tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
        r"""
        Converts macroscopic velocities into the angles of the encoding.

        Inverts :math:`u_x = U \sin \theta_x` and :math:`u_y = U \sin \theta_y`.
        Both scalars and arrays of any shape are accepted.

        Parameters
        ----------
        velocity_x : numpy.typing.NDArray[numpy.float64] | float
            The :math:`x` velocity component, bounded by :attr:`max_velocity`.
        velocity_y : numpy.typing.NDArray[numpy.float64] | float
            The :math:`y` velocity component, bounded by :attr:`max_velocity`.

        Returns
        -------
        Tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
            The angles :math:`\theta_x` and :math:`\theta_y`.

        Raises
        ------
        LatticeException
            If either velocity component exceeds :attr:`max_velocity`.
        """
        velocity_x = np.asarray(velocity_x, dtype=float)
        velocity_y = np.asarray(velocity_y, dtype=float)

        largest_component = float(
            max(
                np.max(np.abs(velocity_x), initial=0.0),
                np.max(np.abs(velocity_y), initial=0.0),
            )
        )

        if largest_component > self.max_velocity * (1.0 + 1e-12):
            raise LatticeException(
                f"Velocity component {largest_component} exceeds the angle-encoding "
                f"bound U={self.max_velocity}. Reduce the flow speed or re-optimize "
                "the branch amplitudes for a larger bound."
            )

        return (
            np.arcsin(np.clip(velocity_x / self.max_velocity, -1.0, 1.0)),
            np.arcsin(np.clip(velocity_y / self.max_velocity, -1.0, 1.0)),
        )

    def equilibrium(
        self,
        density: np.typing.NDArray[np.float64] | float,
        velocity_x: np.typing.NDArray[np.float64] | float,
        velocity_y: np.typing.NDArray[np.float64] | float,
    ) -> np.typing.NDArray[np.float64]:
        r"""
        Evaluates the standard :math:`D_2Q_9` equilibrium populations.

        The populations follow the second-order expansion

        .. math::
            f_i^{eq} = \rho w_i
            \left(
                1 + 3 \mathbf{e}_i \cdot \mathbf{u}
                + \frac{9}{2} (\mathbf{e}_i \cdot \mathbf{u})^2
                - \frac{3}{2} \lvert \mathbf{u} \rvert ^2
            \right).

        Inputs may be scalars or arrays of a common shape; the discrete velocity index
        is appended as the trailing axis of the output.

        Parameters
        ----------
        density : numpy.typing.NDArray[numpy.float64] | float
            The macroscopic density :math:`\rho`.
        velocity_x : numpy.typing.NDArray[numpy.float64] | float
            The :math:`x` velocity component.
        velocity_y : numpy.typing.NDArray[numpy.float64] | float
            The :math:`y` velocity component.

        Returns
        -------
        numpy.typing.NDArray[numpy.float64]
            The equilibrium populations, of shape ``(..., 9)``.
        """
        density = np.asarray(density, dtype=float)
        velocity_x = np.asarray(velocity_x, dtype=float)
        velocity_y = np.asarray(velocity_y, dtype=float)

        velocity_projection = (
            velocity_x[..., None] * self.velocities[:, 0]
            + velocity_y[..., None] * self.velocities[:, 1]
        )
        squared_speed = velocity_x**2 + velocity_y**2

        return (
            density[..., None]
            * self.WEIGHTS
            * (
                1.0
                + 3.0 * velocity_projection
                + 4.5 * velocity_projection**2
                - 1.5 * squared_speed[..., None]
            )
        )

    def macroscopic(
        self,
        populations: np.typing.NDArray[np.float64],
        mask: np.typing.NDArray[np.bool_] | None = None,
    ) -> Tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ]:
        r"""
        Recovers the macroscopic moments of a population field.

        The moments are the usual :math:`\rho = \sum_i f_i` and
        :math:`\rho \mathbf{u} = \sum_i f_i \mathbf{e}_i`.

        Parameters
        ----------
        populations : numpy.typing.NDArray[numpy.float64]
            The population field, of shape ``(..., 9)``.
        mask : numpy.typing.NDArray[numpy.bool\_] | None, optional
            A Boolean mask of shape ``populations.shape[:-1]`` that marks solid nodes,
            by default ``None``. Masked nodes are excluded from the positivity check
            and are assigned a zero density and velocity.

        Returns
        -------
        Tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
            The density and the two velocity components.

        Raises
        ------
        LatticeException
            If the density of an unmasked node is not strictly positive.
        """
        density = np.sum(populations, axis=-1)
        fluid = (
            np.ones(np.shape(density), dtype=bool)
            if mask is None
            else ~np.asarray(mask, dtype=bool)
        )

        if np.any(fluid & (density <= 0.0)):
            raise LatticeException(
                "Encountered a non-positive density while computing macroscopic "
                f"moments: min(rho)={np.min(np.where(fluid, density, np.inf))}."
            )

        # Masked gridpoints hold no fluid, so their moments are zero rather than a
        # division by an empty density.
        safe_density = np.where(fluid, density, 1.0)

        return (
            np.where(fluid, density, 0.0),
            np.where(
                fluid,
                np.sum(populations * self.velocities[:, 0], -1) / safe_density,
                0.0,
            ),
            np.where(
                fluid,
                np.sum(populations * self.velocities[:, 1], -1) / safe_density,
                0.0,
            ),
        )

    def feature_matrix(self) -> np.typing.NDArray[np.float64]:
        r"""
        Builds the :math:`9 \times 6` matrix :math:`C` with :math:`p^{eq} = C \varphi`.

        Substituting :math:`u_x = U \sin\theta_x` and :math:`u_y = U \sin\theta_y` into
        the equilibrium and rewriting :math:`\sin^2 \theta = (1 - \cos 2\theta) / 2`
        makes the expansion exactly linear in the six features of the encoding.

        Returns
        -------
        numpy.typing.NDArray[numpy.float64]
            The coefficient matrix, with columns ordered as
            :math:`[1, \sin\theta_x, \sin\theta_y, \cos 2\theta_x, \cos 2\theta_y, \sin\theta_x \sin\theta_y]`.
        """
        velocity_x = self.velocities[:, 0]
        velocity_y = self.velocities[:, 1]
        bound = self.max_velocity

        coefficients = np.zeros((self.NUM_POPULATIONS, self.NUM_BRANCHES), dtype=float)

        # The constant feature absorbs the angle-independent halves of sin^2.
        coefficients[:, 0] = self.WEIGHTS * (
            1.0 + (3.0 * bound**2 / 4.0) * (3.0 * (velocity_x**2 + velocity_y**2) - 2.0)
        )

        # The first-order terms 3 e . u.
        coefficients[:, 1] = self.WEIGHTS * (3.0 * bound * velocity_x)
        coefficients[:, 2] = self.WEIGHTS * (3.0 * bound * velocity_y)

        # The oscillating halves of sin^2.
        coefficients[:, 3] = self.WEIGHTS * (
            -3.0 * bound**2 / 4.0 * (3.0 * velocity_x**2 - 1.0)
        )
        coefficients[:, 4] = self.WEIGHTS * (
            -3.0 * bound**2 / 4.0 * (3.0 * velocity_y**2 - 1.0)
        )

        # The cross term of (e . u)^2.
        coefficients[:, 5] = self.WEIGHTS * (9.0 * bound**2 * velocity_x * velocity_y)

        return coefficients

    def branch_state(
        self,
        angle_x: float,
        angle_y: float,
    ) -> np.typing.NDArray[np.float64]:
        r"""
        Constructs the 32-entry branch--angle state :math:`\ket{\psi(\theta_x, \theta_y; \beta)}`.

        Each branch contributes four consecutive amplitudes, namely the two-qubit
        angle state :math:`\ket{q_x} \otimes \ket{q_y}` scaled by its branch amplitude.
        This is the statevector that :class:`.ABAngleEncodedEquilibrium` prepares.

        Parameters
        ----------
        angle_x : float
            The angle :math:`\theta_x`.
        angle_y : float
            The angle :math:`\theta_y`.

        Returns
        -------
        numpy.typing.NDArray[numpy.float64]
            The normalized branch--angle statevector.
        """
        state = np.zeros(self.NUM_STATES, dtype=float)

        for branch, (multiplier_x, multiplier_y) in enumerate(self.BRANCH_MULTIPLIERS):
            angle_state = np.kron(
                [np.cos(multiplier_x * angle_x), np.sin(multiplier_x * angle_x)],
                [np.cos(multiplier_y * angle_y), np.sin(multiplier_y * angle_y)],
            )
            state[4 * branch : 4 * branch + 4] = self.beta[branch] * angle_state

        return state

    def sparse_collision_map(self) -> np.typing.NDArray[np.float64]:
        r"""
        Builds the sparse :math:`9 \times 32` map :math:`\tilde{C}(\beta)`.

        Feature :math:`j` appears in the branch--angle state scaled by the amplitude of
        the branch that carries it, so the matching column of :math:`C` is divided by
        that amplitude. Every column that corresponds to an unused amplitude is zero.
        The result satisfies
        :math:`\tilde{C} \ket{\psi(\theta_x, \theta_y; \beta)} = p^{eq}` exactly.

        Returns
        -------
        numpy.typing.NDArray[numpy.float64]
            The sparse collision map.
        """
        coefficients = self.feature_matrix()
        collision_map = np.zeros((self.NUM_POPULATIONS, self.NUM_STATES), dtype=float)

        for feature, (branch, angle) in enumerate(self.FEATURE_SLOTS):
            collision_map[:, 4 * branch + angle] = (
                coefficients[:, feature] / self.beta[branch]
            )

        return collision_map

    def reachable_subspace_basis(
        self,
        samples_per_angle: int = 5,
        tolerance: float = 1e-10,
    ) -> np.typing.NDArray[np.float64]:
        r"""
        Computes an orthonormal basis :math:`B_{in}(\beta)` of the reachable subspace.

        The circuit cannot prepare an arbitrary vector: it only reaches states of the
        form :math:`\ket{\psi(\theta_x, \theta_y; \beta)}`. Their span is recovered
        numerically by sampling a grid of angles and taking the left singular vectors
        that belong to the non-negligible singular values. A :math:`5 \times 5` grid
        robustly recovers the rank-13 subspace of this encoding.

        Parameters
        ----------
        samples_per_angle : int, optional
            The number of samples along each angle, by default 5.
        tolerance : float, optional
            The relative singular-value cutoff used to determine the rank, by default ``1e-10``.

        Returns
        -------
        numpy.typing.NDArray[numpy.float64]
            The basis, of shape ``(32, rank)``.
        """
        angles = np.linspace(-np.pi / 2, np.pi / 2, samples_per_angle)

        samples = np.column_stack(
            [
                self.branch_state(angle_x, angle_y)
                for angle_x in angles
                for angle_y in angles
            ]
        )

        left_vectors, singular_values, _ = np.linalg.svd(samples, full_matrices=False)
        rank = int(np.sum(singular_values > tolerance * singular_values[0]))

        return left_vectors[:, :rank]

    def collision_unitary(self) -> np.typing.NDArray[np.float64]:
        r"""
        Returns the real orthogonal :math:`32 \times 32` collision matrix.

        The matrix is built once at construction time and is reproducible: the QR
        completions that fix its action outside the reachable subspace are seeded
        deterministically. Since simulations only ever feed it branch--angle states,
        the arbitrary part of the completion never affects results.

        Returns
        -------
        numpy.typing.NDArray[numpy.float64]
            The collision unitary, acting on ``[velocity_0, ..., velocity_3, marker]``.
        """
        return self.__unitary

    def physical_state_indices(self) -> np.typing.NDArray[np.int64]:
        r"""
        Returns the statevector indices that hold the nine physical populations.

        These are the marker-one states whose velocity index is a valid
        :math:`D_2Q_9` label, that is, indices :math:`16, \ldots, 24`.

        Returns
        -------
        numpy.typing.NDArray[numpy.int64]
            The nine physical indices, ordered by velocity.
        """
        return np.arange(
            self.NUM_VELOCITY_STATES,
            self.NUM_VELOCITY_STATES + self.NUM_POPULATIONS,
            dtype=np.int64,
        )

    def unused_state_indices(self) -> np.typing.NDArray[np.int64]:
        r"""
        Returns the marker-one statevector indices that carry no physical meaning.

        Four velocity qubits span sixteen labels but :math:`D_2Q_9` only uses nine of
        them, so indices :math:`25, \ldots, 31` must stay empty. Leakage into them is a
        reliable indicator that the collision, streaming, or reflection circuits have
        been composed onto the wrong qubits.

        Returns
        -------
        numpy.typing.NDArray[numpy.int64]
            The seven unused indices.
        """
        return np.arange(
            self.NUM_VELOCITY_STATES + self.NUM_POPULATIONS,
            2 * self.NUM_VELOCITY_STATES,
            dtype=np.int64,
        )

    def populations_from_probabilities(
        self,
        probabilities: np.typing.NDArray[np.float64],
        density_norm: float,
    ) -> np.typing.NDArray[np.float64]:
        r"""
        Reconstructs populations from physical-sector probabilities.

        The encoding stores :math:`f_i / (\alpha \lVert \rho \rVert_2)` as an amplitude,
        so the populations follow from
        :math:`f_i = \alpha \lVert \rho \rVert_2 \sqrt{P_i}`.

        .. warning::
            Probabilities only determine :math:`\lvert f_i \rvert`. The reconstruction
            is therefore valid while all populations are positive, which holds in the
            low-Mach regime the angle encoding is bounded to. A signed population field
            would require phase-sensitive measurements.

        Parameters
        ----------
        probabilities : numpy.typing.NDArray[numpy.float64]
            The probabilities of the physical basis states, of shape ``(..., 9)``.
        density_norm : float
            The Euclidean norm :math:`\lVert \rho \rVert_2` of the density field that
            was encoded, which restores the scale removed at state-preparation time.

        Returns
        -------
        numpy.typing.NDArray[numpy.float64]
            The reconstructed populations, of shape ``(..., 9)``.
        """
        return self.alpha * density_norm * np.sqrt(np.clip(probabilities, 0.0, None))

    def flow_field_from_branch_amplitudes(
        self,
        amplitudes: np.typing.NDArray[np.float64],
        density_norm: float,
    ) -> Tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ]:
        r"""
        Recovers the flow field encoded in branch--angle states, before the collision.

        The state prepared by :class:`.ABBGKInitialConditions` carries the macroscopic
        field in three of its amplitudes, since the branch that holds the constant
        feature is unrotated and the two first-order branches hold one sine each:

        .. math::
            a_0 = \frac{\rho}{\lVert \rho \rVert_2} \beta_0, \quad
            a_6 = \frac{\rho}{\lVert \rho \rVert_2} \beta_1 \sin \theta_x, \quad
            a_9 = \frac{\rho}{\lVert \rho \rVert_2} \beta_2 \sin \theta_y.

        Dividing the last two by the first removes the density and leaves the angles.
        This is what makes the un-evolved initial state readable: unlike every later
        state, it holds branch--angle amplitudes rather than populations.

        .. note::
            Passing the square roots of measured probabilities in place of amplitudes
            recovers the density and the magnitudes :math:`\lvert u_x \rvert` and
            :math:`\lvert u_y \rvert`, but not their signs. The speed
            :math:`\lvert \mathbf{u} \rvert` is unaffected.

        Parameters
        ----------
        amplitudes : numpy.typing.NDArray[numpy.float64]
            The branch--angle amplitudes of each grid point, of shape ``(..., 32)``.
        density_norm : float
            The Euclidean norm of the density field that was encoded.

        Returns
        -------
        Tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
            The density and the two velocity components.
        """
        constant_slot, sine_x_slot, sine_y_slot = (
            4 * branch + angle for branch, angle in self.FEATURE_SLOTS[:3]
        )

        # The unrotated branch holds the density share of the grid point.
        density_share = amplitudes[..., constant_slot] / self.beta[0]
        occupied = density_share != 0.0
        safe_share = np.where(occupied, density_share, 1.0)

        sine_x = np.where(
            occupied, amplitudes[..., sine_x_slot] / (self.beta[1] * safe_share), 0.0
        )
        sine_y = np.where(
            occupied, amplitudes[..., sine_y_slot] / (self.beta[2] * safe_share), 0.0
        )

        return (
            density_norm * density_share,
            self.max_velocity * np.clip(sine_x, -1.0, 1.0),
            self.max_velocity * np.clip(sine_y, -1.0, 1.0),
        )

    def __build_collision_unitary(
        self,
    ) -> Tuple[np.typing.NDArray[np.float64], float, int]:
        """
        Constructs the collision unitary, its normalization, and the reachable rank.

        Returns
        -------
        Tuple[numpy.typing.NDArray[numpy.float64], float, int]
            The unitary, the normalization ``alpha``, and the reachable-subspace rank.
        """
        input_basis = self.reachable_subspace_basis()
        rank = input_basis.shape[1]

        if rank > self.NUM_VELOCITY_STATES:
            raise LatticeException(
                f"The reachable subspace has rank {rank}, which exceeds the "
                f"{self.NUM_VELOCITY_STATES} auxiliary states available in the marker-zero sector."
            )

        # Restrict the exact collision map to the states the circuit can prepare and
        # normalize it so that it becomes contractive.
        restricted_map = self.sparse_collision_map() @ input_basis
        alpha = float(np.linalg.norm(restricted_map, ord=2))
        physical_block = restricted_map / alpha

        # Pad the physical block with J = sqrt(I - P^T P) so that the two blocks
        # together form an isometry.
        residual = np.eye(rank) - physical_block.T @ physical_block
        eigenvalues, eigenvectors = np.linalg.eigh(residual)

        if eigenvalues.min() < -1e-10:
            raise LatticeException(
                "The residual I - P^T P is not positive semidefinite, with a minimum "
                f"eigenvalue of {eigenvalues.min()}. The collision map was not normalized correctly."
            )

        auxiliary_block = (
            eigenvectors
            @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
            @ eigenvectors.T
        )

        # The auxiliary block lives in the marker-zero sector and the physical block
        # in the first nine states of the marker-one sector.
        output_basis = np.zeros((self.NUM_STATES, rank), dtype=float)
        output_basis[:rank, :] = auxiliary_block
        output_basis[self.physical_state_indices(), :] = physical_block

        isometry_error = float(
            np.linalg.norm(output_basis.T @ output_basis - np.eye(rank))
        )
        if isometry_error > 1e-8:
            raise LatticeException(
                f"The prescribed collision output is not an isometry, with error {isometry_error}."
            )

        unitary = (
            self.__complete_basis(output_basis, 2)
            @ self.__complete_basis(input_basis, 1).T
        )

        return unitary, alpha, rank

    def __complete_basis(
        self,
        partial_basis: np.typing.NDArray[np.float64],
        random_seed: int,
    ) -> np.typing.NDArray[np.float64]:
        """
        Extends orthonormal columns to a complete orthonormal basis of the collision space.

        Random columns are appended to the prescribed ones and orthonormalized with a
        QR decomposition. Because the prescribed columns come first and are already
        orthonormal, QR reproduces them up to a sign, which is then corrected.

        Parameters
        ----------
        partial_basis : numpy.typing.NDArray[numpy.float64]
            The columns to preserve, of shape ``(32, rank)``.
        random_seed : int
            The seed of the generator that draws the padding columns. Fixing it keeps
            the collision unitary reproducible across runs.

        Returns
        -------
        numpy.typing.NDArray[numpy.float64]
            A ``(32, 32)`` orthonormal basis whose leading columns are ``partial_basis``.
        """
        num_prescribed = partial_basis.shape[1]
        generator = np.random.default_rng(random_seed)

        basis, _ = np.linalg.qr(
            np.column_stack(
                (
                    partial_basis,
                    generator.standard_normal((self.NUM_STATES, self.NUM_STATES)),
                )
            )
        )

        signs = np.sign(np.sum(basis[:, :num_prescribed] * partial_basis, axis=0))
        signs[signs == 0.0] = 1.0
        basis[:, :num_prescribed] *= signs

        return basis[:, : self.NUM_STATES]

    def __str__(self) -> str:
        """
        The string representation of the encoding.

        Returns
        -------
        str
            A summary of the velocity bound, the normalization, and the branch amplitudes.
        """
        return f"[D2Q9AngleEncoding with U={self.max_velocity}, alpha={self.alpha}, beta={self.beta.tolist()}]"

    def __repr__(self) -> str:
        """
        The string representation of the encoding.

        Returns
        -------
        str
            The same summary as :meth:`__str__`.
        """
        return self.__str__()
