.. _amplitude_components:

====================================
Amplitude-Based Circuits
====================================

.. testcode::
    :hide:

    from qlbm.components import (
        CQLBM,
        MSQLBM,
        MSStreamingOperator,
        ControlledIncrementer,
        SpecularReflectionOperator,
        SpeedSensitivePhaseShift,
    )
    from qlbm.lattice import MSLattice
    print("ok")

.. testoutput::
    :hide:

    ok


This page documents the components that are used in algorithms
that use the **A** mplitude **B** ased (AB) Encoding.
At the moment, this includes two algorihtms:

#. The "regular" Amplitude-Based Collisionless QLBM: ABQLBM,
#. The Multi-Speed (MS) Collisionless QLBM: MSQLBM.

Both algorithms are instances of the Collisionless QLBM (CQLBM), also known as the
Quantum Transport Method (QTM).
Both algorithms compress the grid and the vnumber of discrete velocities
into :math:`N_g\cdot N_v \mapsto \lceil \log_2 N_g \rceil + \lceil \log_2 N_v \rceil` qubits.
The amplitude of each basis state is directly related to the populations in the classical LBM discretization.
The MSQLBM is a generalization of the ABQLBM. 
The implementation of the algorithms was first described in :cite:p:`collisionless` and later expanded in :cite:p:`qmem`.

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

.. autoclass:: qlbm.components.CQLBM

.. autoclass:: qlbm.components.ms.MSQLBM

.. autoclass:: qlbm.components.ab.ABQLBM

.. _cqlbm_streaming:

Streaming
----------------------------------

MSQLBM
**********************************

.. autoclass:: qlbm.components.ms.streaming.MSStreamingOperator

.. autoclass:: qlbm.components.ms.streaming.StreamingAncillaPreparation

.. autoclass:: qlbm.components.ms.streaming.ControlledIncrementer

.. autoclass:: qlbm.components.ms.primitives.SpeedSensitiveAdder

.. autoclass:: qlbm.components.ms.streaming.SpeedSensitivePhaseShift

.. autoclass:: qlbm.components.ms.streaming.PhaseShift

ABQLBM
**********************************

.. autoclass:: qlbm.components.ab.streaming.ABStreamingOperator

.. _cqlbm_reflection:

Reflection
----------------------------------

MSQLBM Reflection
**********************************

.. autoclass:: qlbm.components.ms.bounceback_reflection.BounceBackReflectionOperator
    :members:

.. autoclass:: qlbm.components.ms.specular_reflection.SpecularReflectionOperator
    :members:

.. autoclass:: qlbm.components.ms.bounceback_reflection.BounceBackWallComparator

.. autoclass:: qlbm.components.ms.specular_reflection.SpecularWallComparator

.. autoclass:: qlbm.components.ms.primitives.EdgeComparator

.. autoclass:: qlbm.components.ms.primitives.Comparator

.. autoclass:: qlbm.components.ms.primitives.ComparatorMode

ABQLBM Reflection
**********************************

.. autoclass:: qlbm.components.ab.reflection.ABReflectionOperator

.. _cqlbm_others:

Initial Conditions
-----------------------------------

MSQLBM Initial Conditions
**********************************

.. autoclass:: qlbm.components.ms.primitives.MSInitialConditions

.. autoclass:: qlbm.components.ms.primitives.MSInitialConditions3DSlim

ABQLBM Initial Conditions
**********************************

.. autoclass:: qlbm.components.ab.initial.ABInitialConditions

Measurement
-----------------------------------

MSQLBM Measurement
**********************************

.. autoclass:: qlbm.components.ms.primitives.GridMeasurement

ABQLBM Measurement
**********************************

.. autoclass:: qlbm.components.ab.measurement.ABGridMeasurement