from qlbm.components.collisionless import SpeedSensitivePhaseShift

# A phase shift of 5 qubits, controlled on speed index 2
SpeedSensitivePhaseShift(num_qubits=5, speed=2, positive=True).draw("mpl")