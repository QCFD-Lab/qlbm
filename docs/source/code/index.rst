.. _internal_docs:

Internal Documentation
================================

``qlbm`` is made up of 4 main modules.
Together, the :ref:`base_components`, :ref:`amplitude_components`, :ref:`qlga_components`, and :ref:`misc_components`
module handle the parameterized creation of quantum circuits that compose QBMs.
The :ref:`lattice` module parses external information into quantum
registers and provides uniform interfaces for underlying algorithms.
The :ref:`infra` module integrates the quantum components
with Tket, Qiskit, and Qulacs transpilers and runners.
The :ref:`tools` module contains miscellaneous utilities.

.. rst-class:: center-align-col

+----------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+
|                            | Encodings                                                                                                                                                                    |
+----------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+
|                            | Amplitude Encodings                                                                                    |  Computational Basis State Encoding                                 |
+============================+==================================+==================================+==================================+==================================+==================================+
|                            | Ampl. Based                      | One-Hot                          | Multi-Speed                      | Space-Time                       | Linear                           |
+----------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+
| Algorithm                  | QLBM                                                                                                   |QLGA                                                                 |
+----------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+
| Reference                  | N/A                              | :cite:`erio`                     | :cite:`collisionless`            | :cite:`spacetime`                | :cite:`lqlga1, spacetime2`       |
+----------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+
| Discretization             | :math:`D_dQ_q`                                                      | MS :cite:`collisionless`         | :math:`D_dQ_q`                                                      |
+----------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+
| Implementation             | :math:`D_2Q_9`                                                      | 2D, 3D, :math:`\geq 4` speeds    | :math:`D_1Q_2`, :math:`D_1Q_3`, :math:`D_2Q_4`                      |
+----------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+
| Required Qubits            | :math:`{O}(\log(qN_g))`          | :math:`{O}(\log(N_g)+q)`         | :math:`{O}(\log(qN_g))`          | :math:`{O}(\log(N_g)+N_t^d)`     | :math:`qN_g`                     |
+----------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+----------------------------------+



.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   lattice
   comps_base
   comps_cqlbm
   comps_lga
   comps_other
   infra
   tools
