import logging.config
from logging import Logger, getLogger
from typing import List

from qiskit_aer import AerSimulator

from qlbm.components import (
    CQLBM,
    CollisionlessInitialConditions,
    EmptyPrimitive,
    GridMeasurement,
)
from qlbm.infra import QiskitRunner, SimulationConfig
from qlbm.lattice import CollisionlessLattice
from qlbm.tools.utils import create_directory_and_parents, get_circuit_properties


def benchmark(
    lattice_files: List[str],
    num_shots: int,
    num_steps: int,
    statevector_snapshots: bool,
    statevector_sampling: bool,
    logger: Logger,
    dummy_logger: Logger,
    root_output_dir: str,
    target_platform="QISKIT",
    compiler_platform="QISKIT",
    execution_backend=AerSimulator(method="statevector"),
    sampling_backend=AerSimulator(method="statevector"),
    num_repetitions: int = 5,
) -> None:
    for rep in range(num_repetitions):
        logger.info(f"Repetition #{rep} of {num_repetitions}")
        for count, lattice_file in enumerate(lattice_files):
            logger.info(f"Combination #{count + 1} of {len(lattice_files)}")

            lattice_name = lattice_file.split("/")[-1].split(".")[0].replace("_", "-")
            lattice = CollisionlessLattice(lattice_file, logger=dummy_logger)
            logger.info(
                f"Lattice={lattice_file}, num_qubits={lattice.num_total_qubits}"
            )

            # Compile initial condition circuit
            output_dir = (
                f"{root_output_dir}/qiskit-{lattice_name}-{statevector_snapshots}"
            )

            create_directory_and_parents(output_dir)
            lattice = CollisionlessLattice(lattice_file)

            cfg = SimulationConfig(
                initial_conditions=CollisionlessInitialConditions(
                    lattice, logger=dummy_logger
                ),
                algorithm=CQLBM(lattice, logger=dummy_logger),
                postprocessing=EmptyPrimitive(lattice, logger=dummy_logger),
                measurement=GridMeasurement(lattice, logger=dummy_logger),
                target_platform=target_platform,
                compiler_platform=compiler_platform,
                optimization_level=0,
                statevector_sampling=statevector_sampling,
                execution_backend=execution_backend,
                sampling_backend=sampling_backend,
                logger=logger,
            )

            cfg.prepare_for_simulation()

            logger.info(
                f"Final circuit properties: {get_circuit_properties(cfg.algorithm)}"
            )
            # Create a runner object to simulate the circuit
            runner = QiskitRunner(cfg, lattice, logger=logger)

            # Simulate the circuits using both snapshots and sampling
            runner.run(
                num_steps,  # Number of time steps
                num_shots,  # Number of shots per time step
                output_dir,
                statevector_snapshots=statevector_snapshots,
            )


if __name__ == "__main__":
    NUM_SHOTS = 2**14
    NUM_STEPS = 5
    ROOT_OUTPUT_DIR = "qlbm-output/benchmark-statevector-snapshots"

    create_directory_and_parents(ROOT_OUTPUT_DIR)

    lattice_files = [
        # "demos/lattices/2d_8x8_1_obstacle.json",
        # "demos/lattices/2d_8x8_2_obstacle.json",
        "demos/lattices/2d_16x16_1_obstacle.json",
        "demos/lattices/2d_16x16_2_obstacle.json",
        "demos/lattices/2d_16x16_3_obstacle.json",
        # "demos/lattices/2d_32x32_1_obstacle.json",
        # "demos/lattices/2d_32x32_2_obstacle.json",
        # "demos/lattices/2d_32x32_3_obstacle.json",
    ]

    dummy_logger = getLogger("dummy")

    # By logging at this point we ignore the output of circuit creation
    logging.config.fileConfig("demos/benchmarks/statevector_snapshots.conf")
    logger = getLogger("qlbm")
    logger.info("Session: snapshots=True")
    benchmark(
        lattice_files,
        NUM_SHOTS,
        NUM_STEPS,
        True,
        True,
        logger,
        dummy_logger,
        ROOT_OUTPUT_DIR,
        num_repetitions=5,
    )

    logger.info("Session: snapshots=False")
    benchmark(
        lattice_files,
        NUM_SHOTS,
        NUM_STEPS,
        False,
        False,
        logger,
        dummy_logger,
        ROOT_OUTPUT_DIR,
        num_repetitions=5,
    )
