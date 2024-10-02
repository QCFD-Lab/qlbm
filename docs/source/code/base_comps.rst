.. _base_components:

====================================
Base Classes
====================================

For extendability and modularity purposes, ``qlbm`` provides several base classes
with interfaces that aim to make the development of novel QLBM methods easier.
This page documents the base classes that have to do with the register
setup and quantum circuit generation.
The architecture of the inheritance hierarchy is two-fold:
vertically based on specificity (i.e., more specialized classes for specific QLBMs)
and horizontally based on methods (i.e., different implementations for different QLBMs).

.. _lattice_bases:

Lattice Base
----------------------------------

.. autoclass:: qlbm.lattice.lattices.base.Lattice
    :members:

.. _components_bases:

Components Base
----------------------------------

.. autoclass:: qlbm.components.base.QuantumComponent
    :members:

.. autoclass:: qlbm.components.base.LBMPrimitive

.. autoclass:: qlbm.components.base.LBMOperator

.. autoclass:: qlbm.components.base.CQLBMOperator

.. autoclass:: qlbm.components.base.SpaceTimeOperator

.. autoclass:: qlbm.components.base.LBMAlgorithm