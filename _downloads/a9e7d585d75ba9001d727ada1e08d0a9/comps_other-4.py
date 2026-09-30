from qlbm.components.common.adders import ParameterizedPhaseShift

# A phase shift of 5 qubits, adding the number 2
ParameterizedPhaseShift(num_qubits=5, num_to_add=2, positive=True).draw("mpl")