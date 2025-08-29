from qlbm.components.collisionless import Comparator, ComparatorMode

# On a 5 qubit register, compare the number 3
Comparator(num_qubits=5,
           num_to_compare=3,
           mode=ComparatorMode.LT).draw("mpl")