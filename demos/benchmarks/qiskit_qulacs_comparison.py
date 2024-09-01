import logging.config
from logging import Logger, getLogger
from typing import List

from pytket.extensions.qulacs import QulacsBackend as TketQulacsBackend
from qiskit_aer import StatevectorSimulator

from qlbm.components import (
    CQLBM,
    CollisionlessInitialConditions,
    EmptyPrimitive,
    GridMeasurement,
)
from qlbm.infra import CircuitCompiler, CircuitRunner
from qlbm.lattice import CollisionlessLattice
from qlbm.tools.utils import create_directory_and_parents, get_circuit_properties


def benchmark(
    cfgs: List[str],
    num_steps: int,
    num_shots: int,
    statevector_snapshots: bool,
    statevector_sampling: bool,
    logger: Logger,
    dummy_logger: Logger,
) -> None:
    # Declare the compiler and running platform
    qiskit_compiler = CircuitCompiler("TKET", "QISKIT", logger=dummy_logger)
    qulacs_compiler = CircuitCompiler("TKET", "QULACS", logger=dummy_logger)

    # Declare the Qiskit backend to run on
    qiskit_backend = StatevectorSimulator()
    qulacs_backend = TketQulacsBackend()

    for count, lattice_file in enumerate(cfgs):
        logger.info(f"Combination #{count + 1} of {len(cfgs)}")

        lattice_name = lattice_file.split("/")[-1].split(".")[0].replace("_", "-")
        lattice = CollisionlessLattice(lattice_file, logger=dummy_logger)
        logger.info(f"Lattice={lattice_file}, num_qubits={lattice.num_total_qubits}")

        # Compile initial condition circuit
        initial_conditions = CollisionlessInitialConditions(lattice, logger=dummy_logger)
        algorithm = CQLBM(lattice, logger=dummy_logger)
        postprocessing = EmptyPrimitive(lattice, logger=dummy_logger)
        measurement = GridMeasurement(lattice, logger=dummy_logger)

        logger.info("Executing QULACS...")
        # Compile the circuits, except for GridMeasurement
        algorithm = qulacs_compiler.compile(algorithm, qulacs_backend, 0)
        initial_conditions = qulacs_compiler.compile(
            initial_conditions, qulacs_backend, 0
        )
        postprocessing = qiskit_compiler.compile(
            postprocessing, qulacs_backend, 0
        )  # Compile with qiskit!
        logger.info(f"Final circuit properties: {get_circuit_properties(algorithm)}")

        # Declare the output directory
        output_dir = f"qlbm-output/qiskit-qulacs-comparison/qulacs-{lattice_name}-{statevector_snapshots}-{statevector_snapshots}"
        create_directory_and_parents(output_dir)

        qulacs_runner = CircuitRunner(
            lattice,
            output_dir,
            "step",
        )

        qulacs_runner.run(
            initial_conditions,
            algorithm,
            postprocessing,
            measurement,
            num_steps,  # Number of time steps
            num_shots,  # Number of shots per time step
            statevector_snapshots,  # Enable snapshots
            statevector_sampling,  # Enable SV sampling
        )

        logger.info("Executing QISKIT...")
        # Compile initial condition circuit
        initial_conditions = CollisionlessInitialConditions(lattice, logger=dummy_logger)
        algorithm = CQLBM(lattice, logger=dummy_logger)
        postprocessing = EmptyPrimitive(lattice, logger=dummy_logger)
        measurement = GridMeasurement(lattice, logger=dummy_logger)

        # Compile the circuits, except for GridMeasurement
        algorithm = qiskit_compiler.compile(algorithm, qulacs_backend, 0)
        initial_conditions = qiskit_compiler.compile(
            initial_conditions, qulacs_backend, 0
        )
        postprocessing = qiskit_compiler.compile(postprocessing, qulacs_backend, 0)
        logger.info(f"Final circuit properties: {get_circuit_properties(algorithm)}")

        # Declare the output directory
        output_dir = f"qlbm-output/qiskit-qulacs-comparison/qiskit-{lattice_name}-{statevector_snapshots}-{statevector_snapshots}"
        create_directory_and_parents(output_dir)

        qiskit_runner = CircuitRunner(
            lattice,
            output_dir,
            "step",
            backend=qiskit_backend,
        )

        qiskit_runner.run(
            initial_conditions,
            algorithm,
            postprocessing,
            measurement,
            num_steps,  # Number of time steps
            num_shots,  # Number of shots per time step
            statevector_snapshots,  # Enable snapshots
            statevector_sampling,  # Enable SV sampling
        )


if __name__ == "__main__":
    num_shots = 2**13
    num_steps = 3

    cfgs = [
        # "demos/lattices/2d_8x8_1_obstacle.json",
        # "demos/lattices/2d_8x8_2_obstacle.json",
        "demos/lattices/2d_16x16_1_obstacle.json",
        # "demos/lattices/2d_16x16_2_obstacle.json",
        # "demos/lattices/2d_16x16_3_obstacle.json",
        # "demos/lattices/2d_32x32_1_obstacle.json",
        # "demos/lattices/2d_32x32_2_obstacle.json",
        # "demos/lattices/2d_32x32_3_obstacle.json",
    ]

    dummy_logger = getLogger("dummy")

    # By logging at this point we ignore the output of circuit creation
    logging.config.fileConfig(
        "demos/performance_analysis/qiskit_qulacs_comparison_logging.conf"
    )
    logger = getLogger("qlbm")
    logger.info("Session: snapshots=True, sampling=True")
    benchmark(cfgs, num_steps, num_shots, True, True, logger, dummy_logger)

    logger.info("Session: snapshots=False, sampling=True")
    benchmark(cfgs, num_steps, num_shots, False, True, logger, dummy_logger)
