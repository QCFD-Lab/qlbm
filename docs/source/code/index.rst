.. _internal_docs:

Internal Documentation
================================

``qlbm`` is made up of 4 main modules.
Together, the :ref:`base_components`, :ref:`amplitude_components`, and :ref:`stqlbm_components`
module handle the parameterized creation of quantum circuits that compose QBMs.
The :ref:`lattice` module parses external information into quantum
registers and provides uniform interfaces for underlying algorithms.
The :ref:`infra` module integrates the quantum components
with Tket, Qiskit, and Qulacs transpilers and runners.
The :ref:`tools` module contains miscellaneous utilities.

.. toctree::

   lattice
   comps_base
   comps_cqlbm
   comps_collision
   comps_stqbm
   comps_lqlga
   infra
   tools

