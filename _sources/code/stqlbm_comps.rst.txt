.. _stqlbm_components:

====================================
Space-Time Circuits
====================================

This page contains documentation about the quantum circuits that make up the
**S**\ pace-\ **T**\ ime **Q**\ uantum **L**\ attice **B**\ oltzmann **M**\ ethod (STQLBM)
described in :cite:p:`spacetime`.
At its core, the Space-Time QLBM uses an extended computational basis state
encoding that that circumvents the non-locality of the streaming
step by including additional information from neighboring grid points.
This happens in several distinct steps:

#. Initial conditions prepare the starting state of the probability distribution function.
#. :ref:`stqlbm_streaming` moves the position of particles to neighboring points according to their velocity.
#. :ref:`stqlbm_collision` locally changes the velocity profile of particles positioned at the same position in space.
#. Measurement operations extract information out of the quantum state, which can later be post-processed classically.

This page documents the individual components that make up the STQLBM algorithm.
Subsections follow a top-down approach, where end-to-end operators are introduced first,
before being broken down into their constituent parts.

.. warning::
    The STQBLM algorithm is a based on typical :math:`D_dQ_q` discretizations.
    The current implementation only supports :math:`D_2Q_4` for one time step.
    This is work in progress.
    Multiple steps are possible through ``qlbm``\ 's reinitialization mechanism.

.. _stqlbm_e2e:

End-to-end algorithms
----------------------------------

.. autoclass:: qlbm.components.spacetime.spacetime.SpaceTimeQLBM

.. _stqlbm_streaming:

Streaming
----------------------------------

.. autoclass:: qlbm.components.spacetime.streaming.SpaceTimeStreamingOperator

.. _stqlbm_collision:

Collision
----------------------------------

.. autoclass:: qlbm.components.spacetime.collision.SpaceTimeCollisionOperator


.. _stqlbm_others:

Others
-----------------------------------

.. autoclass:: qlbm.components.spacetime.initial_conditions.SpaceTimeInitialConditions

.. autoclass:: qlbm.components.spacetime.measurement.SpaceTimeGridVelocityMeasurement