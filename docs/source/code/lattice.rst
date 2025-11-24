.. _lattice:

Lattices and Geometry
=====================

This page contains documentation about :ref:`lattices` and :ref:`geometry` classes of ``qlbm``.
Lattices and geometry go hand-in-hand in that they do not themselves contain
quantum components, but instead provide a convenient interface
for accessing the information that determines the structure and composition of quantum components.
``qlbm`` supports the following kinds of lattices:

#. Amplitude-Based (AB) lattices. These are the most common encdoings in QLBM literature. All AB lattices compress the grid into logarithmically many qubits. 
    #. :class:`.AmpltiudeLattice` is the abstract base class for all amplitude-based lattices.
    #. :class:`.ABLattice` is the "standard" amplitude-based lattice, where both the grid and the velocities are logarithmically compressed. It supports only :math:`D_dQ_q` discretization.
    #. :class:`.MSLattice` is the multi-speed lattice for the algorithm described in :cite:t:`collisionless`. It is the same as the :class:`.ABLattice`, except it supports different velocity discretizations.
    #. :class:`.OHLattice` is the amplitude-based lattice where the grid is logarithmically compressed, but the :math:`D_dQ_q` velocities are not. It assigns one basis state per discrete velocity.

#. LGA lattices. These rely on the computational basis state encoding (CBSE) and are used for QLGA algorithms.
    #. :class:`.SpaceTimeLattice` is the realization of the space-time data encoding described in :cite:`spacetime` and :cite:`spacetime2`. It uses an expanded CBSE to accomodate multiple time steps.
    #. :class:`.LQLGALattice` is the entirely uncompressed CBSE, encoding all velocity channels in the system.

.. _lattices:

Lattices
----------------------------------

Lattices are the backbone of ``qlbm`` quantum components.
While they do not contain quantum circuits themselves, lattices encode the information that
makes implementing and extending QLBM quantum circuits seamless.
Concretely, each :class:`.Lattice` fulfills the following functionality:

#. Infers the number of qubits required to construct a quantum circuit based on a user specification.
#. Warn the user of invalid or ill-formed specifications.
#. Group the qubits into separate quantum registers according to their functionality.
#. Provide convenient indexing methods methods to access individual (or groups of) qubits based on their purpose.
#. Encode additional information required for the automatic assembly of large quantum circuits.

.. autoclass:: qlbm.lattice.lattices.ms_lattice.MSLattice
    :members:

.. autoclass:: qlbm.lattice.lattices.ab_lattice.ABLattice
    :members:

.. autoclass:: qlbm.lattice.lattices.oh_lattice.OHLattice
    :members:

.. autoclass:: qlbm.lattice.lattices.spacetime_lattice.SpaceTimeLattice
    :members:

.. autoclass:: qlbm.lattice.lattices.lqlga_lattice.LQLGALattice
    :members:

.. _geometry:

Geometry
----------------------------------

Processing obstacle geometry into quantum circuits is a tedious and error-prone task when performed manually.
To alleviate this challenge, ``qlbm`` provides :class:`.Block` and :class:`.Circle` classes that
parse the geometry information supplied as part of the :class:`.Lattice` specification
into information that parameterized the construction of quantum circuits.
This includes the position of the obstacle within the grid and its boundary conditions.
In addition, these shapes contain triangulation methods that
allow them to be exported as ``stl`` files and visualized in Paraview.

Each shape contains snippets of information that determine how
individual components of reflection behave.
To make the generation of this circuits more manageable, we
segment the block information into different categories of edge cases, which are also broken down by algorithm.

The :class:`.CQLBM` algorithm uses the following data structures:

#. :class:`.DimensionalReflectionData` models the isolated, one-dimensional features of a fixed point on the grid.
#. :class:`.ReflectionPoint` models the 2D or 3D information of a fixed point in space.
#. :class:`.ReflectionWall` models the 2D or 3D information of the wall of the obstacle.
#. :class:`.ReflectionResetEdge` models the 3D information of an edge along the walls of an obstacle.

The :class:`.SpaceTimeQLBM` algorithm on makes use of the following:

#. :class:`.SpaceTimePWReflectionData` models the reflection data of a single grid point of the lattice.
#. :class:`.SpaceTimeVolumetricReflectionData` models the reflection data of a contiguous volume in space.
#. :class:`.SpaceTimeDiagonalReflectionData` of diagonals in 2D.

.. note::
    :class:`.CQLBM` and :class:`.SpaceTimeQLBM` support different kinds of geometry.
    For :class:`.CQLBM`, geometry objects can only be 2D or 3D cuboids, and they must be placed
    at least two grid points apart for consistent behavior.
    :class:`.SpaceTimeQLBM` supports 2D rectangles of arbitrary lengths, as well as circles.

.. autoclass:: qlbm.lattice.geometry.Block
    :members:

.. autoclass:: qlbm.lattice.geometry.Circle
    :members:

.. autoclass:: qlbm.lattice.geometry.DimensionalReflectionData

.. autoclass:: qlbm.lattice.geometry.ReflectionPoint

.. autoclass:: qlbm.lattice.geometry.ReflectionWall

.. autoclass:: qlbm.lattice.geometry.ReflectionResetEdge

.. autoclass:: qlbm.lattice.geometry.SpaceTimePWReflectionData

.. autoclass:: qlbm.lattice.geometry.SpaceTimeVolumetricReflectionData

.. autoclass:: qlbm.lattice.geometry.SpaceTimeDiagonalReflectionData