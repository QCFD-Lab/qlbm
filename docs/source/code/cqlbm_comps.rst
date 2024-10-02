.. _cqlbm_components:

====================================
Collisionless Circuits
====================================

This page contains documentation about the quantum circuits that make up the
**C**\ ollisionless **Q**\ uantum **L**\ attice **B**\ oltzmann **M**\ ethod (CQLBM)
first described in :cite:p:`collisionless` and later expanded in :cite:p:`qmem`.
At its core, the CQLBM algorithm manipulates the particle probability distribution
in an amplitude-based encoding of the quantum state.
This happens in several distinct steps:

#. Initial conditions prepare the starting state of the probability distribution function.
#. :ref:`cqlbm_streaming` circuits increment or decrement the position of particles in physical space through QFT-based streaming.
#. :ref:`cqlbm_reflection` circuits apply boundary conditions that affect particles that come in contact with solid obstacles. Reflection places those particles back in the appropriate position of the fluid domain.
#. Measurement operations extract information out of the quantum state, which can later be post-processed classically.

This page documents the individual components that make up the CQLBM algorithm.
Subsections follow a top-down approach, where end-to-end operators are introduced first,
before being broken down into their constituent parts.

.. _e2e:

End-to-end algorithms
----------------------------------

.. autoclass:: qlbm.components.collisionless.cqlbm.CQLBM

.. _cqlbm_streaming:

Streaming
----------------------------------

.. autoclass:: qlbm.components.collisionless.streaming.CollisionlessStreamingOperator

.. autoclass:: qlbm.components.collisionless.streaming.StreamingAncillaPreparation

.. autoclass:: qlbm.components.collisionless.streaming.ControlledIncrementer

.. autoclass:: qlbm.components.collisionless.primitives.SpeedSensitiveAdder

.. autoclass:: qlbm.components.collisionless.streaming.SpeedSensitivePhaseShift

.. autoclass:: qlbm.components.collisionless.streaming.PhaseShift

.. _cqlbm_reflection:

Reflection
----------------------------------

.. autoclass:: qlbm.components.collisionless.bounceback_reflection.BounceBackReflectionOperator
    :members:

.. autoclass:: qlbm.components.collisionless.specular_reflection.SpecularReflectionOperator
    :members:

.. autoclass:: qlbm.components.collisionless.bounceback_reflection.BounceBackWallComparator

.. autoclass:: qlbm.components.collisionless.specular_reflection.SpecularWallComparator

.. autoclass:: qlbm.components.collisionless.primitives.EdgeComparator

.. autoclass:: qlbm.components.collisionless.primitives.Comparator

.. autoclass:: qlbm.components.collisionless.primitives.ComparatorMode

.. _cqlbm_others:

Others
-----------------------------------

.. autoclass:: qlbm.components.collisionless.primitives.CollisionlessInitialConditions

.. autoclass:: qlbm.components.collisionless.primitives.CollisionlessInitialConditions3DSlim

.. autoclass:: qlbm.components.collisionless.primitives.GridMeasurement