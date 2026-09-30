from qlbm.components.common.adders import ParameterizedPhaseShift

# A phase shift of 5 qubits, controlled subtracting the number 1
ParameterizedPhaseShift(num_qubits=5, num_to_add=1, positive=False, num_ctrl_qubits=3).draw("mpl")