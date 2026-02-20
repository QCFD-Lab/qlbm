.. _misc_components:

====================================
Common and Misc Circuits
====================================

Circuits that are used throughout different algorithms,
have niche use cases, or are not encoding-specific.

This page documents components that are shared throughout different
encodings and different stages of algorithms.

Comparators
----------------------------------

.. autoclass:: qlbm.components.common.SingleRegisterComparator

.. autoclass:: qlbm.components.common.TwoRegisterComparator

.. autoclass:: qlbm.tools.ComparatorMode

Arithmetic
----------------------------------

.. autoclass:: qlbm.components.common.ParameterizedDraperAdder

.. autoclass:: qlbm.components.common.ParameterizedPhaseShift


Miscellaneous
----------------------------------

.. autoclass:: qlbm.components.common.EmptyPrimitive

.. autoclass:: qlbm.components.common.MCSwap

.. autoclass:: qlbm.components.common.HammingWeightAdder

.. autoclass:: qlbm.components.common.TruncatedQFT

.. autoclass:: qlbm.components.common.UniformStatePrep

.. autoclass:: qlbm.components.common.AdditionConversion

.. autoclass:: qlbm.components.common.StateSetter