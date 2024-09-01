# from os import listdir
from os.path import exists

from qiskit.qasm3 import load as load_qasm3

from qlbm.tools.utils import get_circuit_properties

if __name__ == "__main__":
    qasm_circuit_dir = "demos/openqasm/circuits"

    if not exists(qasm_circuit_dir):
        raise RuntimeError(f'Directory "{qasm_circuit_dir}" does not exist.')

    # all_circuits = listdir(qasm_circuit_dir)

    # circuit_files_qasm2, circuit_files_qasm3 = [], []

    # for circuit_file_name in sorted(all_circuits):
    #     relative_path = f"{qasm_circuit_dir}/{circuit_file_name}"
    #     # if circuit_file_name.endswith(".qasm2"):
    #     #     circuit_files_qasm2.append(load_qasm2(relative_path))
    #     print(relative_path)
    #     if circuit_file_name.endswith(".qasm3"):
    #         circuit_files_qasm3.append(load_qasm3(relative_path))

    # Alternatively, read in a single circuit
    specific_circuit = load_qasm3("demos/openqasm/circuits/2d-16x16-2-obstacle.qasm2")
    print(get_circuit_properties(specific_circuit))
    # simulator = QasmSimulator()
    # simulator.run(specific_circuit)
