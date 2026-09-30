from qlbm.components.common import HammingWeightAdder

# Add the Hamming weight of a 3-qubit register onto a 5-qubit register
HammingWeightAdder(3, 5).draw("mpl")