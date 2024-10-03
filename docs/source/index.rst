``qlbm`` documentation
======================

The long-term goal of ``qlbm`` is to be a quantum computational fluid dynamics (QCFD) solver for fault-tolerant
quantum computers running on a heterogeneous quantum-classical high-performance computer (QHPC).
Currently, the primary aim of ``qlbm`` is to accelerate
and improve the research surrounding  **Q**\ uantum **L**\ attice **B**\ oltzmann **M**\ ethods (QLBMs).
On this website, you can find the :ref:`internal_docs` of the source code components that make up ``qlbm``.

``qlbm`` is made up of 4 main modules.
Together, the :ref:`base_components`, :ref:`cqlbm_components`, and :ref:`stqlbm_components`
module handle the parameterized creation of quantum circuits that compose QBMs.
The :ref:`lattice` module parses external information into quantum
registers and provides uniform interfaces for underlying algorithms.
The :ref:`infra` module integrates the quantum components
with Tket, Qiskit, and Qulacs transpilers and runners.
The :ref:`tools` module contains miscellaneous utilities.

``qlbm`` currently supports two algorithms:

#. The **C**\ ollisionless **QLBM** (CQLBM) first described in :cite:p:`collisionless` and later expanded in :cite:p:`qmem`.

#. **S**\ pace-\ **T**\ ime **QLBM** (STQLBM) described in :cite:p:`spacetime`.

.. card:: Internal Documentation
    :link: internal_docs
    :link-type: ref
    
    :fas:`book;sd-text-primary` Detailed documentation of ``qlbm``.

.. card:: Tutorials
    :link: tutorials
    :link-type: ref
    
    :fas:`flask;sd-text-primary` Hands-on examples.


.. toctree::
   :caption: Code documentation
   :maxdepth: 2

   code/index
   examples/index

References
-----------------------------------

.. bibliography::