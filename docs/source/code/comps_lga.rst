.. _qlga_components:

====================================
QLGA Circuits
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
    from qlbm.components import (
        CQLBM,
        MSStreamingOperator,
        ControlledIncrementer,
        SpecularReflectionOperator,
        SpeedSensitivePhaseShift,
    )
    from qlbm.lattice import MSLattice, LQLGALattice
    print("ok")

.. testoutput::
    :hide:

    ok


This page contains documentation about the quantum circuits that make up the 
**L**\ attice **G**\ as **A**\ utomata (LGA) algorithms of ``qlbm``.
This includes two aglorithms:

#. **S**\ pace-\ **T**\ ime **Q**\ uantum **L**\ attice **B**\ oltzmann **M**\ ethod (STQLBM), described in :cite:p:`spacetime` and extended in :cite:p:`spacetime2`.
#. **L**\ inear **Q**\ uantum **L**\ attice **G**\ as **A**\ utomata (LQLGA), :cite:p:`lqlga1`, :cite:p:`lqlga2`.

At its core, the Space-Time QLBM uses an extended computational basis state
encoding that that circumvents the non-locality of the streaming
step by including additional information from neighboring grid points.
This happens in several distinct steps:
The LQGLA encodes a lattice of :math:`N_g` gridpoints with :math:`q` discrete velocities
each into :math:`N_g \cdot q` qubits.
LQLGA can be seen as the "limit" of the extended computational basis state encoding that is the Space-Time encoding.
For both algorithms, time-evolution of the system consists of the following steps:

#. :ref:`lqlga_initial` prepare the starting state of the flow field.
#. :ref:`lqlga_streaming` move particles across gridpoints according to the velocity discretization.
#. :ref:`lqlga_reflection` circuits apply boundary conditions that affect particles that come in contact with solid obstacles. Reflection places those particles back in the appropriate position of the fluid domain.
#. :ref:`lqlga_collision` operators create superposed local configurations of velocity profiles.
#. :ref:`lqlga_measurement` operations extract information out of the quantum state, which can later be post-processed classically.

This page documents the individual components that make up the CQLBM algorithm.
Subsections follow a top-down approach, where end-to-end operators are introduced first,
before being broken down into their constituent parts.

.. warning::
    STQLBM and LQLGA are a based on typical :math:`D_dQ_q` discretizations.
    The current implementation only supports :math:`D_1Q_2`, :math:`D_1Q_3`, and :math:`D_2Q_4` for one time step
    with inexact restarts through ``qlbm``\ 's reinitialization mechanism.
    LQLGA only supports :math:`D_1Q_3`.

.. note::
    Need to work with a different discretization or want to work together? Reach out at ``qcfd-ewi@tudelft.nl``.

.. _lqlga_e2e:

End-to-end algorithms
----------------------------------

.. autoclass:: qlbm.components.spacetime.spacetime.SpaceTimeQLBM

.. autoclass:: qlbm.components.LQLGA

.. _lqlga_initial:

Initial Conditions
----------------------------------

.. autoclass:: qlbm.components.spacetime.initial.PointWiseSpaceTimeInitialConditions

.. autoclass:: qlbm.components.LQGLAInitialConditions

.. _lqlga_streaming:

Streaming
----------------------------------

.. autoclass:: qlbm.components.spacetime.streaming.SpaceTimeStreamingOperator

.. autoclass:: qlbm.components.LQLGAStreamingOperator

.. _lqlga_reflection:

Reflection
----------------------------------

.. autoclass:: qlbm.components.LQLGAReflectionOperator

.. _lqlga_collision:

Collision
-----------------------------------

The collision module contains collision operators and adjacent logic classes.
The former implements the circuits that perform collision in computational basis state encodings,
while the latter contains useful abstractions that circuits build on top of.
Collision in LGA algorithms is based on the concept of equivalence classes
described in Section 4 of :cite:p:`spacetime2`, and follows a permute-redistribute-unpermute (PRP) approach.
All components of this module may be used for different variations of the Computational Basis State Encoding (CBSE)
of the velocity register.
The components of this module consist of:

.. autoclass:: qlbm.components.spacetime.collision.GenericSpaceTimeCollisionOperator

.. autoclass:: qlbm.components.spacetime.collision.SpaceTimeD2Q4CollisionOperator

.. autoclass:: qlbm.components.GenericLQLGACollisionOperator

.. autoclass:: qlbm.components.common.EQCPermutation

.. autoclass:: qlbm.components.common.EQCRedistribution

.. autoclass:: qlbm.components.common.EQCCollisionOperator

.. autoclass:: qlbm.lattice.spacetime.properties_base.LatticeDiscretization

.. autoclass:: qlbm.lattice.spacetime.properties_base.LatticeDiscretizationProperties

.. autoclass:: qlbm.lattice.eqc.EquivalenceClass

.. autoclass:: qlbm.lattice.eqc.EquivalenceClassGenerator


.. _lqlga_measurement:

Measurement
----------------------------------

.. autoclass:: qlbm.components.spacetime.measurement.SpaceTimeGridVelocityMeasurement

.. autoclass:: qlbm.components.LQLGAGridVelocityMeasurement