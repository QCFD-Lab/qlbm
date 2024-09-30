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

.. autoclass:: qlbm.lattice.CollisionlessLattice
    :members:

.. autoclass:: qlbm.lattice.SpaceTimeLattice
    :members:

.. _geometry:

Geometry
----------------------------------
