.. _lqlga_components:

====================================
LQLGA Circuits
====================================

.. testcode::
    :hide:

    from qlbm.components import (
        LQLGA,
        LQGLAInitialConditions,
        LQLGAGridVelocityMeasurement,
        LQLGAReflectionOperator,
        LQLGAStreamingOperator,
    )
    from qlbm.lattice import LQLGALattice
    print("ok")

.. testoutput::
    :hide:

    ok


This page contains documentation about the quantum circuits that make up the
**L**\ inear **Q**\ uantum **L**\ attice **G**\ as **A**\ utomata (LQLGA).
For a more in-depth depth description of the LQLGA algorithm,
we suggest :cite:p:`lqlga1`, :cite:p:`lqlga2`, and :cite:p:`spacetime2`.
The LQGLA encodes a lattice of :math:`N_g` gridpoints with :math:`q` discrete velocities
each into :math:`N_g \cdot q` qubits.
The time-evolution of the system consists of the following steps:

#. :ref:`lqlga_initial` prepare the starting state of the flow field.
#. :ref:`lqlga_streaming` move particles across gridpoints according to the velocity discretization.
#. :ref:`lqlga_reflection` circuits apply boundary conditions that affect particles that come in contact with solid obstacles. Reflection places those particles back in the appropriate position of the fluid domain.
#. :ref:`lqlga_collision` operators create superposed local configurations of velocity profiles.
#. :ref:`lqlga_measurement` operations extract information out of the quantum state, which can later be post-processed classically.

This page documents the individual components that make up the CQLBM algorithm.
Subsections follow a top-down approach, where end-to-end operators are introduced first,
before being broken down into their constituent parts.

.. _lqlga_e2e:

End-to-end algorithms
----------------------------------

.. autoclass:: qlbm.components.LQLGA

.. _lqlga_initial:

Initial Conditions
----------------------------------

.. autoclass:: qlbm.components.LQGLAInitialConditions

.. _lqlga_streaming:

Streaming
----------------------------------

.. autoclass:: qlbm.components.LQLGAStreamingOperator

.. _lqlga_reflection:

Reflection
----------------------------------

.. autoclass:: qlbm.components.LQLGAReflectionOperator

.. _lqlga_collision:

Collision
-----------------------------------

.. autoclass:: qlbm.components.GenericLQLGACollisionOperator

.. _lqlga_measurement:

Measurement
----------------------------------

.. autoclass:: qlbm.components.LQLGAGridVelocityMeasurement