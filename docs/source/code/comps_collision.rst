.. _comps_collision:

====================================
Collision Logic Classes
====================================

.. testcode::
    :hide:

    from qlbm.components import (
        EQCPermutation,
        EQCRedistribution,
    )
    from qlbm.lattice.spacetime.properties_base import (
        LatticeDiscretization,
        LatticeDiscretizationProperties,
    )

    from qlbm.lattice.eqc import (
        EquivalenceClass,
        EquivalenceClassGenerator,
    )
    print("ok")

.. testoutput::
    :hide:

    ok


This page contain collision-related components shared
between the :ref:`stqlbm_components` and :ref:`lqlga_components` algorithms.
Collision in these algorithms is based on the concept of equivalence classes
described in Section 4 of :cite:p:`spacetime2`, and follows a PRP (permute-redistribute-unpermute) approach.
All components of this module may be used for different variations of the Computational Basis State Encoding (CBSE)
of the velocity register.
The components of this module consist of:

#. :ref:`collision_logic` classes that provide information about the lattice discretization and equivalence classes.
#. :ref:`collision_components` that implement the small parts of the collision operator.

.. _collision_logic:

Collision Logic Classes
-----------------------------------

.. autoclass:: qlbm.lattice.spacetime.properties_base.LatticeDiscretization

.. autoclass:: qlbm.lattice.spacetime.properties_base.LatticeDiscretizationProperties

.. autoclass:: qlbm.lattice.eqc.EquivalenceClass

.. autoclass:: qlbm.lattice.eqc.EquivalenceClassGenerator

.. _collision_components:

Collision Components
-----------------------------------

.. autoclass:: qlbm.components.common.EQCPermutation

.. autoclass:: qlbm.components.common.EQCRedistribution

.. autoclass:: qlbm.components.common.EQCCollisionOperator

