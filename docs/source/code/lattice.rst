.. _lattice:

Lattices and Geometry
=====================

This page contains documentation about :ref:`lattices` and :ref:`geometry` classes of ``qlbm``.
Lattices and geometry go hand-in-hand in that they do not themselves contain
quantum components, but instead provide a convenient interface
for accessing the information that determines the structure and composition of quantum components.

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

.. autoclass:: qlbm.lattice.lattices.collisionless_lattice.CollisionlessLattice
    :members:

.. autoclass:: qlbm.lattice.lattices.spacetime_lattice.SpaceTimeLattice
    :members:

.. _geometry:

Geometry
----------------------------------

Processing obstacle geometry into quantum circuits is a tedious and error-prone task when performed manually.
To alleviate this challenge, ``qlbm`` provides a :class:`.Block` class that
parses the geometry information supplied as part of the :class:`.Lattice` specification
into information that parameterized the construction of quantum circuits.
This includes the position of the obstacle within the grid and its boundary conditions.
In addition, :class:`Block`\ s contain triangulation methods that
allow them to be exported as ``stl`` files and visualized in Paraview.

Each block contains snippets of information that determine how
individual components of reflection behave.
To make the generation of this circuits more manageable, we
segment the block information into different categories of edge cases:

#. :class:`.DimensionalReflectionData` models the isolated, one-dimensional features of a fixed point on the grid.
#. :class:`.ReflectionPoint` models the 2D or 3D information of a fixed point in space.
#. :class:`.ReflectionWall` models the 2D or 3D information of the wall of the obstacle.
#. :class:`.ReflectionResetEdge` models the 3D information of an edge along the walls of an obstacle.

.. note::
    At the moment, only the :class:`.CQLBM` algorithm supports geometry.
    Geometry objects can only be 2D or 3D cuboids, and they must be placed
    at least two grid points apart for consistent behavior.

.. autoclass:: qlbm.lattice.blocks.Block
    :members:

.. autoclass:: qlbm.lattice.blocks.DimensionalReflectionData

.. autoclass:: qlbm.lattice.blocks.ReflectionPoint

.. autoclass:: qlbm.lattice.blocks.ReflectionWall

.. autoclass:: qlbm.lattice.blocks.ReflectionResetEdge