import argparse
from typing import List

import pandas as pd

parser = argparse.ArgumentParser(
    prog="QLBM Log File Analaysis",
    description="Parses qlb.log files into csv format for data analysis",
    epilog="Text at the bottom of help",
)

parser.add_argument(
    "-f",
    "--file",
    dest="filename",
    type=str,
)

parser.add_argument("-o", "--output", dest="output_filename", default="data", type=str)

args = parser.parse_args()

compiler_lines: List[str] = []
simulation_lines: List[str] = []
circuit_properties_lines: List[str] = []
compiler_info_line = None
simulation_start_line = None
simulation_end_line = None

with open(args.filename, "r") as f:
    for line in f.readlines():
        if "Compilation took" in line:
            compiler_lines.append(line)
        elif "Simulation start" in line:
            if simulation_start_line is not None:
                raise RuntimeError(
                    "Ill-formatted log file: multiple simulation start lines!"
                )
            simulation_start_line = line
        elif "Entire simulation" in line:
            if simulation_end_line is not None:
                raise RuntimeError(
                    "Ill-formatted log file: multiple simulation end lines!"
                )
            simulation_end_line = line
        elif "Simulation of" in line:
            simulation_lines.append(line)
        elif "Main circuit" in line:
            circuit_properties_lines.append(line)
        elif "[Compiler" in line and compiler_info_line is None:
            compiler_info_line = line


if compiler_info_line is None:
    raise RuntimeError("Ill-formated log file: no compiler info line!")

if simulation_start_line is None:
    raise RuntimeError("Ill-formatted log file: no simulation start line!")

if simulation_end_line is None:
    raise RuntimeError("Ill-formatted log file: no simulation end line!")

if len(simulation_lines) != len(circuit_properties_lines):
    raise RuntimeError(
        f"Ill-formatted log file: {len(simulation_lines)} simulation lines and {len(circuit_properties_lines)} circuit property lines!"
    )

compiler_platform = compiler_info_line.split("INFO: ")[-1].split()[2]
compiler_target = compiler_info_line.split()[-1][:-1]
compiler_time = sum(float(line.split()[-2]) for line in compiler_lines)


shots_per_step = int(simulation_start_line.split("num_shots=")[1].split(",")[0])
backend = simulation_start_line.split("on backend ")[1].split()[0]
total_steps = int(simulation_start_line.split("num_steps=")[1].split(",")[0])
snapshots = bool(simulation_start_line.split("snapshots=")[1].split(",")[0])
sv_sampling = bool(
    simulation_start_line.split("statevector_sampling=")[1].split(".")[0]
)
total_time = float(simulation_end_line.split()[-1])

simulation_platform = simulation_start_line.split("Simulation start: ")[1].split()[0]

steps = []
step_simulation_time = []
circuit_qubits = []
circuit_depth = []
circuit_gates = []


for sl, cl in zip(simulation_lines, circuit_properties_lines):
    step = sl.split("Simulation of ")[1].split()[0]
    simulation_time = sl.split()[-1]
    qc_step = cl.split("for step ")[1].split()[0]

    if qc_step != step:
        raise RuntimeError("Misaligned steps in log file!")

    qc_properties = cl.split("has properties ")[-1].strip("\n")[1:-1].split(", ")

    steps.append(int(step))
    step_simulation_time.append(float(simulation_time))
    circuit_qubits.append(int(qc_properties[1]))
    circuit_depth.append(int(qc_properties[2]))
    circuit_gates.append(int(qc_properties[3]))

pd.DataFrame(
    {
        "Compiler Platform": [compiler_platform],
        "Compiler Target": [compiler_target],
        "Compiler Time": [compiler_time],
        "Backend": [backend],
        "Total Steps": [total_steps],
        "Snapshot Simulation": [snapshots],
        "Statevector Sampling": [sv_sampling],
        "Simulation Duration": [total_time],
    },
).to_csv(f"{args.output_filename}_extra.csv", index=False)

pd.DataFrame(
    {
        "Step": steps,
        "Simulation Duration": step_simulation_time,
        "Qubits": circuit_qubits,
        "Depth": circuit_depth,
        "Gates": circuit_gates,
        "Shots": [shots_per_step] * len(steps),
    },
).to_csv(f"{args.output_filename}_simulation.csv", index=False)
