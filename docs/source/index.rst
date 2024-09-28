``qlbm`` documentation
======================

``qlbm`` aims to be a quantum computational fluid dynamics (QCFD) solver for fault-tolerant
quantum computers running on a heterogeneous quantum-classical high-performance computer (QHPC).

Module Documentation
====================

On this website, you can find the documentation of the source code components that make up `qlbm`.
The documentation follows the same structure as the source code.
``qlbm`` is made up of 4 main modules, listed below.
The :ref:`cqlbm_components` module handles the parameterized creation
of quantum circuits that compose QBMs.
The :ref:`lattice` module parses external information into quantum
register and provides uniform interfaces for underlying algorithms.
The :ref:`infra` module integrates the quantum components
with Tket, Qiskit, and Qulacs transpilers and runners.
The :ref:`tools` module contains miscellaneous utilities.


.. toctree::
   :caption: Code documentation
   :maxdepth: 2

   code/index