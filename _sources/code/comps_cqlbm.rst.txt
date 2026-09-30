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
        ParameterizedPhaseShift,
    )
    from qlbm.lattice import MSLattice
    print("ok")

.. testoutput::
    :hide:

    ok


This page documents the components that are used in algorithms
that use the **A** mplitude **B** ased (AB) Encoding.
At the moment, this includes two algorithms:

#. The "regular" Amplitude-Based Collisionless QLBM: ABQLBM,
#. The Multi-Speed (MS) Collisionless QLBM: MSQLBM.

:class:`.ABQLBM` uses :class:`.ABLattice`\ s and :class:`.OHLattice`\ s, while :class:`.MSQLBM` is built from :class:`.MSLattice`\ s.
The general interface of :class:`.CQLBM` is built from all 3 lattice instances and delegates to the appropriate implementation automatically.

Both algorithms are instances of the Collisionless QLBM (:class:`.CQLBM`), also known as the
Quantum Transport Method (QTM).
All amplitude-based lattices compress the grid into logarithmically many
qubits. :class:`.ABLattice` also compresses the velocity index, whereas
:class:`.OHLattice` assigns one qubit to each velocity channel. For example,
excluding ancillae, a :math:`16 \times 8` D2Q9 lattice uses 7 grid qubits and
either 4 velocity qubits with :class:`.ABLattice` (11 total) or 9 velocity
qubits with :class:`.OHLattice` (16 total).
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

.. warning::

    Constructing an :class:`.ABLattice` for a :math:`D_dQ_q` discretization
    does not imply that every AB operator supports that discretization. Current
    operator support is:

    .. list-table::
        :header-rows: 1
        :widths: 40 30

        * - Component
          - Supported discretizations
        * - :class:`.ABDiscreteUniformInitialConditions`
          - Generic :math:`D_dQ_q`
        * - :class:`.ABStreamingOperator`
          - D1Q3 with :class:`.ABLattice`; D2Q9 with :class:`.ABLattice` or
            :class:`.OHLattice`
        * - AB reflection operators
          - D2Q9 with :class:`.ABLattice` or :class:`.OHLattice`
        * - Complete :class:`.ABQLBM`
          - D2Q9 with :class:`.ABLattice` or :class:`.OHLattice`

.. _cqlbm_e2e:

End-to-end algorithms
----------------------------------

.. autoclass:: qlbm.components.CQLBM

.. autoclass:: qlbm.components.ms.MSQLBM

.. autoclass:: qlbm.components.ab.ABQLBM

.. _cqlbm_initial:

Initial Conditions
-----------------------------------

Several AB components identify velocities by their zero-based channel index.
For D2Q9, the convention is:

.. list-table:: D2Q9 velocity channels
    :header-rows: 1
    :widths: 15 25 35

    * - Index
      - Velocity vector
      - Direction
    * - 0
      - ``(0, 0)``
      - Rest
    * - 1
      - ``(+1, 0)``
      - Positive x
    * - 2
      - ``(0, +1)``
      - Positive y
    * - 3
      - ``(-1, 0)``
      - Negative x
    * - 4
      - ``(0, -1)``
      - Negative y
    * - 5
      - ``(+1, +1)``
      - Positive x, positive y
    * - 6
      - ``(-1, +1)``
      - Negative x, positive y
    * - 7
      - ``(-1, -1)``
      - Negative x, negative y
    * - 8
      - ``(+1, -1)``
      - Positive x, negative y

With :class:`.ABLattice`, the velocity register stores this index in binary.
With :class:`.OHLattice`, channel ``i`` is represented by a 1 on velocity
qubit ``i`` and 0 on every other velocity qubit.

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
