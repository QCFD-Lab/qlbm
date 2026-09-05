.. _amplitude_components:

====================================
Amplitude-Based Circuits
====================================

.. testcode::
    :hide:

    from qlbm.components import (
        CQLBM,
        MSQLBM,
        ABBGKQLBM,
        MSStreamingOperator,
        ControlledIncrementer,
        SpecularReflectionOperator,
        ParameterizedPhaseShift,
    )
    from qlbm.lattice import ABBGKLattice, MSLattice
    print("ok")

.. testoutput::
    :hide:

    ok


This page documents the components that are used in algorithms
that use the **A** mplitude **B** ased (AB) Encoding.
At the moment, this includes three algorithms:

#. The "regular" Amplitude-Based Collisionless QLBM: ABQLBM,
#. The Multi-Speed (MS) Collisionless QLBM: MSQLBM,
#. The Amplitude-Based QLBM with an angle-encoded BGK collision: ABBGKQLBM.

:class:`.ABQLBM` uses :class:`.ABLattice`\ s and :class:`.OHLattice`\ s, while :class:`.MSQLBM` is built from :class:`.MSLattice`\ s.
The general interface of :class:`.CQLBM` is built from all 3 lattice instances and delegates to the appropriate implementation automatically.

The first two algorithms are instances of the Collisionless QLBM (:class:`.CQLBM`), also known as the
Quantum Transport Method (QTM).
:class:`.ABBGKQLBM` extends the first with a collision, and is documented in :ref:`abbgk`.
Both algorithms compress the grid and the number of discrete velocities
into :math:`N_g\cdot N_v \mapsto \lceil \log_2 N_g \rceil + \lceil \log_2 N_v \rceil` qubits.
The amplitude of each basis state is directly related to the populations in the classical LBM discretization.
The MSQLBM is a generalization of the ABQLBM. 
The implementation of the algorithms was first described in :cite:p:`collisionless` and later expanded in :cite:p:`qmem`.

At its core, the CQLBM algorithm manipulates the particle probability distribution
in an amplitude-based encoding of the quantum state.
The :ref:`cqlbm_e2e` can be broken down into several distinct steps:

#. :ref:`cqlbm_initial` conditions prepare the starting state of the probability distribution function.
#. :ref:`cqlbm_streaming` circuits increment or decrement the position of particles in physical space through QFT-based streaming.
#. :ref:`cqlbm_reflection` circuits apply boundary conditions that affect particles that come in contact with solid obstacles. Reflection places those particles back in the appropriate position of the fluid domain.
#. :ref:`cqlbm_measurement` operations extract information out of the quantum state, which can later be post-processed classically.

This page documents the individual components that make up the CQLBM algorithm.
Subsections follow a top-down approach, where end-to-end operators are introduced first,
before being broken down into their constituent parts.

.. _cqlbm_e2e:

End-to-end algorithms
----------------------------------

.. autoclass:: qlbm.components.CQLBM

.. autoclass:: qlbm.components.ms.MSQLBM

.. autoclass:: qlbm.components.ab.ABQLBM

.. _cqlbm_initial:

Initial Conditions
-----------------------------------

.. autoclass:: qlbm.components.ms.primitives.MSInitialConditions

.. autoclass:: qlbm.components.ms.primitives.MSInitialConditions3DSlim

.. autoclass:: qlbm.components.ab.initial.ABDiscreteUniformInitialConditions

.. autoclass:: qlbm.components.ab.initial.ABParallelDiscreteUniformInitialConditions

.. autoclass:: qlbm.components.ab.initial.ABInitialConditions

.. _cqlbm_streaming:

Streaming
----------------------------------

.. autoclass:: qlbm.components.ms.streaming.MSStreamingOperator

.. autoclass:: qlbm.components.ms.streaming.StreamingAncillaPreparation

.. autoclass:: qlbm.components.ms.streaming.ControlledIncrementer

.. autoclass:: qlbm.components.ms.streaming.PhaseShift

.. autoclass:: qlbm.components.ab.streaming.ABStreamingOperator

.. _cqlbm_reflection:

Reflection
----------------------------------

.. autoclass:: qlbm.components.ms.bounceback_reflection.BounceBackReflectionOperator
    :members:

.. autoclass:: qlbm.components.ms.specular_reflection.SpecularReflectionOperator
    :members:

.. autoclass:: qlbm.components.ms.bounceback_reflection.BounceBackWallComparator

.. autoclass:: qlbm.components.ms.specular_reflection.SpecularWallComparator

.. autoclass:: qlbm.components.ms.primitives.EdgeComparator


.. note::
    The amplitude-based QLBM supports two kinds of reflection operators: segment-wise (or standard) and zone-agnostic.
    The two methods are physically equivalent, but their implementation differs algorithmically.
    For multi-geometry cases, only the segment-wise implementation is currently supported.


.. autoclass:: qlbm.components.ab.reflection.ABReflectionOperator

.. autoclass:: qlbm.components.ab.reflection.ABBounceBackReflectionOperator

.. autoclass:: qlbm.components.ab.reflection.ABSpecularReflectionOperator

.. autoclass:: qlbm.components.ab.reflection.ABZoneAgnosticReflectionOperator 

.. autoclass:: qlbm.components.ab.reflection.ABZoneAgnosticReflectionOracle

.. _cqlbm_measurement:

Measurement
-----------------------------------

.. autoclass:: qlbm.components.ms.primitives.GridMeasurement

.. autoclass:: qlbm.components.ab.measurement.ABGridMeasurement

.. _abbgk:

Angle-Encoded BGK Collision
----------------------------------

The collisionless algorithms above stream populations that are already stored as
amplitudes. :class:`.ABBGKQLBM` adds a :math:`\tau=1` BGK collision to every time step,
so a single circuit performs a complete LBM cycle.

The construction rests on a change of variables. Writing the macroscopic velocity as
:math:`u_x = U \sin\theta_x` and :math:`u_y = U \sin\theta_y` makes the
:math:`D_2Q_9` equilibrium *exactly linear* in six trigonometric features of the two
angles. Those features are prepared in superposition on five qubits, three labelling the
feature and two carrying the rotations, and a single :math:`32 \times 32` unitary maps
the resulting state onto the equilibrium populations. The classical linear algebra behind
that unitary lives in :class:`.D2Q9AngleEncoding`; see :ref:`collision_models`.

A collision contracts the state, so the unitary needs somewhere to put the norm it
removes. :class:`.ABBGKLattice` reserves one marker qubit for that purpose: its
:math:`\ket{1}` sector holds the physical populations and its :math:`\ket{0}` sector the
auxiliary amplitudes. Streaming and reflection are therefore controlled on the marker,
which :class:`.ABBGKQLBM` arranges by passing it to :class:`.ABStreamingOperator` and
:class:`.ABReflectionOperator`.

.. note::
    The collision maps a branch-angle state onto populations, and not back again, so a
    time step cannot be applied twice in a row. Each step starts from a freshly encoded
    flow field, which :class:`.ABBGKReinitializer` derives from the state at the end of
    the previous one. Recomputing the equilibrium from the moments at every step is what
    keeps the nonlinearity of the collision exact rather than linearized, and is why
    simulations of this algorithm require statevector snapshots.

.. autoclass:: qlbm.components.ab.bgk.ABBGKQLBM

.. autoclass:: qlbm.components.ab.collision.bgk_collision.ABBGKCollisionOperator

.. autoclass:: qlbm.components.ab.collision.bgk_collision.ABLocalBGKCollision

.. autoclass:: qlbm.components.ab.collision.initial.ABBGKInitialConditions

.. autoclass:: qlbm.components.ab.collision.angle_encoding.ABAngleEncodedEquilibrium

.. autoclass:: qlbm.components.ab.collision.angle_encoding.ABBranchStatePreparation

.. autoclass:: qlbm.components.ab.collision.angle_encoding.ABBranchAngleEncoding

.. autofunction:: qlbm.components.ab.collision.angle_encoding.append_multi_controlled_ry

.. autoclass:: qlbm.components.ab.collision.measurement.ABBGKMeasurement
