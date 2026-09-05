r"""Implementation of the Amplitude-Based lattice specialized for the angle-encoded BGK collision."""

from __future__ import annotations

from logging import Logger, getLogger
from typing import TYPE_CHECKING, List

import numpy as np
from typing_extensions import override

if TYPE_CHECKING:
    from qlbm.infra.compiler import CircuitCompiler
    from qlbm.infra.reinitialize.base import Reinitializer
    from qlbm.infra.result.base import QBMResult

from qlbm.lattice.bgk.angle_encoding import D2Q9AngleEncoding
from qlbm.lattice.spacetime.properties_base import LatticeDiscretization
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import basis_state_offsets

from .ab_lattice import ABLattice


class ABBGKLattice(ABLattice):
    r"""
    Implementation of the :class:`.ABLattice` specialized for the :class:`.ABBGKQLBM` algorithm.

    The collisionless :class:`.ABQLBM` streams populations that are already stored as
    amplitudes, so it needs no additional structure. Adding a collision does: the
    :math:`\tau=1` BGK equilibrium is a contraction, not a unitary, so it can only be
    realized by pushing the lost norm into an auxiliary sector. This lattice reserves
    exactly one marker qubit for that purpose:

    .. list-table:: Meaning of the marker qubit
        :widths: 25 60
        :header-rows: 1

        * - Marker state
          - Sector
        * - :math:`\ket{1}`
          - The physical sector, holding the nine :math:`D_2Q_9` populations of every grid point.
        * - :math:`\ket{0}`
          - The auxiliary sector, holding the amplitudes that make the collision unitary.

    Everything that acts on populations must therefore be restricted to the physical
    sector, which :class:`.ABBGKQLBM` does by supplying the marker qubit as an
    additional control to streaming and reflection.

    Because the marker is reserved, this lattice cannot additionally use markers to
    label parallel geometries: unlike :class:`.ABLattice`, it accepts a single geometry
    only. The register layout is otherwise identical to :class:`.ABLattice`, and the
    marker qubit is placed in the same :attr:`marker_register`.

    A lattice is constructed from the same specification as any other
    :math:`D_2Q_9` amplitude-based lattice:

    .. code-block:: json

        {
            "lattice": {
                "dim": {
                    "x": 16,
                    "y": 16
                },
                "velocities": "d2q9"
            },
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [7, 8],
                    "y": [7, 8],
                    "boundary": "bounceback"
                }
            ]
        }

    The register setup can be visualized by constructing a lattice object:

    .. plot::
        :include-source:

        from qlbm.lattice import ABBGKLattice

        ABBGKLattice(
            {
                "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "d2q9"},
                "geometry": [],
            }
        ).circuit.draw("mpl")

    .. list-table:: Constructor Attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`lattice_data`
          - The lattice specification, either a path to a JSON file or a dictionary.
        * - :attr:`encoding`
          - The :class:`.D2Q9AngleEncoding` that parameterizes the collision, by default a new instance.
        * - :attr:`logger`
          - The performance logger, by default ``getLogger("qlbm")``.
    """

    encoding: D2Q9AngleEncoding
    """The angle encoding that parameterizes the collision of this lattice."""

    density_norm: float
    r"""The Euclidean norm of the density field currently encoded on this lattice.

    Quantum states are normalized, so the absolute scale of the density has to be kept
    classically. :class:`.ABBGKInitialConditions` updates this attribute whenever it
    encodes a field, and :class:`.ABBGKReinitializer` reads it back to restore the scale
    of the populations it decodes. Velocities are ratios of moments and are therefore
    independent of it."""

    def __init__(
        self,
        lattice_data,
        encoding: D2Q9AngleEncoding | None = None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice_data, logger)

        if self.discretization != LatticeDiscretization.D2Q9:
            raise LatticeException(
                f"The angle-encoded BGK collision is only implemented for D2Q9, got {self.discretization}."
            )

        self.encoding = (
            D2Q9AngleEncoding(logger=logger) if encoding is None else encoding
        )
        self.density_norm = 1.0

        # Overrides the geometry-derived marker logic of the ABLattice: the single
        # marker of this lattice separates physical from auxiliary amplitudes.
        self.set_num_marker_qubits(1)

    @override
    def set_geometries(self, geometries):
        """
        Updates the geometry of the lattice, which must remain a single configuration.

        Parameters
        ----------
        geometries : List
            A list holding exactly one geometry.

        Raises
        ------
        LatticeException
            If more than one geometry is supplied. The marker qubit that
            :class:`.ABLattice` would use to label parallel geometries is reserved here
            for the physical sector of the collision.
        """
        if len(geometries) > 1:
            raise LatticeException(
                "The ABBGKLattice reserves its marker qubit for the physical sector of "
                "the collision, and therefore cannot simulate multiple geometries in "
                f"parallel. Got {len(geometries)} geometries; use an ABLattice instead."
            )

        super().set_geometries(geometries)

        # The base implementation re-derives the marker count from the geometry.
        self.set_num_marker_qubits(1)

    def collision_index(self) -> List[int]:
        r"""
        Get the indices of the qubits that the collision unitary acts on.

        The collision reinterprets the same five qubits before and after it is applied.
        Beforehand they hold the two angle qubits and the three branch qubits of the
        encoding; afterwards they hold the :math:`D_2Q_9` velocity label and the marker.
        The order below is the little-endian order that
        :class:`.D2Q9AngleEncoding` assumes.

        Returns
        -------
        List[int]
            The four velocity qubit indices followed by the marker qubit index.
        """
        return self.velocity_index() + self.marker_index()

    def grid_basis_offsets(self) -> np.typing.NDArray[np.int64]:
        """
        Get the statevector index contribution of every grid coordinate.

        Combining an entry of this table with the offsets of the collision qubits
        through a bitwise ``or`` gives the index of a complete basis state, which makes
        it possible to write and read the state of one grid point at a time.

        Returns
        -------
        numpy.typing.NDArray[numpy.int64]
            The grid offsets, of shape ``(num_x_gridpoints, num_y_gridpoints)``.
        """
        return (
            basis_state_offsets(
                np.arange(self.num_gridpoints[0] + 1), self.grid_index(0)
            )[:, None]
            | basis_state_offsets(
                np.arange(self.num_gridpoints[1] + 1), self.grid_index(1)
            )[None, :]
        )

    def collision_basis_offsets(self) -> np.typing.NDArray[np.int64]:
        """
        Get the statevector index contribution of every state of the collision qubits.

        The 32 entries are ordered exactly like the branch--angle statevectors of
        :meth:`.D2Q9AngleEncoding.branch_state`, so a local state is embedded into the
        lattice statevector by adding a grid offset to this table.

        Returns
        -------
        numpy.typing.NDArray[numpy.int64]
            The collision offsets, of shape ``(32,)``.
        """
        return basis_state_offsets(
            np.arange(self.encoding.NUM_STATES), self.collision_index()
        )

    def collision_basis_indices(self) -> np.typing.NDArray[np.int64]:
        """
        Get the statevector indices of every collision state of every grid point.

        Where :meth:`physical_basis_indices` selects the nine population states, this
        returns all 32, in the order that :meth:`.D2Q9AngleEncoding.branch_state` uses.
        It is what makes an un-evolved branch--angle state readable.

        Returns
        -------
        numpy.typing.NDArray[numpy.int64]
            The indices, of shape ``(num_x_gridpoints, num_y_gridpoints, 32)``.
        """
        return self.__marked_velocity_indices(
            np.arange(self.encoding.NUM_STATES, dtype=np.int64)
        )

    def physical_basis_indices(self) -> np.typing.NDArray[np.int64]:
        r"""
        Get the statevector indices that hold the physical populations of every grid point.

        These are the marker-one states whose velocity register holds a valid
        :math:`D_2Q_9` label.

        Returns
        -------
        numpy.typing.NDArray[numpy.int64]
            The indices, of shape ``(num_x_gridpoints, num_y_gridpoints, 9)``.
        """
        return self.__marked_velocity_indices(self.encoding.physical_state_indices())

    def unused_basis_indices(self) -> np.typing.NDArray[np.int64]:
        r"""
        Get the statevector indices of the marker-one states without a physical meaning.

        Four velocity qubits span sixteen labels while :math:`D_2Q_9` only uses nine, so
        these indices must remain empty throughout a simulation. Probability mass
        appearing here is a reliable sign that a component has been composed onto the
        wrong qubits.

        Returns
        -------
        numpy.typing.NDArray[numpy.int64]
            The indices, of shape ``(num_x_gridpoints, num_y_gridpoints, 7)``.
        """
        return self.__marked_velocity_indices(self.encoding.unused_state_indices())

    def ancilla_basis_mask(self) -> np.typing.NDArray[np.bool_]:
        r"""
        Get a mask over the statevector that selects states with an occupied ancilla.

        The comparator and obstacle ancillae that boundary conditions rely on are
        uncomputed before the end of a time step, so every state selected by this mask
        must carry zero amplitude. This makes the mask a cheap correctness check on the
        reflection operators.

        Returns
        -------
        numpy.typing.NDArray[numpy.bool\_]
            The mask, of shape ``(2 ** num_qubits,)``.
        """
        interface_qubits = set(
            self.grid_index() + self.velocity_index() + self.marker_index()
        )
        ancilla_bitmask = sum(
            1 << qubit
            for qubit in range(self.circuit.num_qubits)
            if qubit not in interface_qubits
        )

        return (
            np.arange(1 << self.circuit.num_qubits, dtype=np.int64) & ancilla_bitmask
        ) != 0

    def __marked_velocity_indices(
        self,
        collision_states: np.typing.NDArray[np.int64],
    ) -> np.typing.NDArray[np.int64]:
        """
        Combines grid offsets with the offsets of a subset of the collision states.

        Parameters
        ----------
        collision_states : numpy.typing.NDArray[numpy.int64]
            The local collision states to index, as values of :meth:`collision_index`.

        Returns
        -------
        numpy.typing.NDArray[numpy.int64]
            The indices, of shape ``(num_x_gridpoints, num_y_gridpoints, len(collision_states))``.
        """
        return (
            self.grid_basis_offsets()[:, :, None]
            | basis_state_offsets(collision_states, self.collision_index())[
                None, None, :
            ]
        )

    @override
    def create_result(self, output_directory: str, output_file_name: str) -> QBMResult:
        from qlbm.infra.result.ab_bgk_result import ABBGKResult

        return ABBGKResult(self, output_directory, output_file_name)

    @override
    def create_reinitializer(
        self,
        compiler: CircuitCompiler,
        logger: Logger = getLogger("qlbm"),
    ) -> Reinitializer:
        from qlbm.infra.reinitialize.ab_bgk_reinitializer import ABBGKReinitializer

        return ABBGKReinitializer(self, compiler, logger)

    @override
    def logger_name(self) -> str:
        return f"abbgklattice-{self.num_dims}d-{'x'.join(str(n + 1) for n in self.num_gridpoints)}-{len(self.shape_list)}-obstacle"

    @override
    def __str__(self) -> str:
        return f"[ABBGKLattice with {[n + 1 for n in self.num_gridpoints]} gridpoints, {self.discretization.name} velocities, and {len(self.shape_list)} obstacles]"
