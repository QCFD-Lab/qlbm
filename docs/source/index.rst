``qlbm`` documentation
======================

``qlbm`` aims to be a quantum computational fluid dynamics (QCFD) solver for fault-tolerant
quantum computers running on a heterogeneous quantum-classical high-performance computer (QHPC).
On this website, you can find the :ref:`internal_docs` of the source code components that make up ``qlbm``.
``qlbm`` is made up of 4 main modules.

Together, the :ref:`base_components`, :ref:`cqlbm_components`, and :ref:`stqlbm_components`
module handle the parameterized creation of quantum circuits that compose QBMs.
The :ref:`lattice` module parses external information into quantum
registers and provides uniform interfaces for underlying algorithms.
The :ref:`infra` module integrates the quantum components
with Tket, Qiskit, and Qulacs transpilers and runners.
The :ref:`tools` module contains miscellaneous utilities.


.. toctree::
   :caption: Code documentation
   :maxdepth: 2

   code/index

References
-----------------------------------

.. bibliography::