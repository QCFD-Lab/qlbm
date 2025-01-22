from qiskit_aer import AerSimulator

# Compiler the circuit to the qiskit AerSimulator
compiled_circuit = compiler.compile(component, backend=AerSimulator())