import logging.config
from logging import Logger, getLogger
from typing import List

from pytket.extensions.qulacs import QulacsBackend as TketQulacsBackend

from qlbm.components import (
    CQLBM,
)
from qlbm.infra import CircuitCompiler
from qlbm.lattice import CollisionlessLattice
from qlbm.tools.utils import create_directory_and_parents


def benchmark(
    lattice_files: List[str],
    logger: Logger,
    dummy_logger: Logger,
    compiler_platform: List[str],
    target_platform: List[str],
    optimization_levels: List[int],
    backend: TketQulacsBackend | None,
    num_repetitions: int = 5,
) -> None:
    for rep in range(num_repetitions):
        logger.info(f"Repetition #{rep} of {num_repetitions}")
        for count, lattice_file in enumerate(lattice_files):
            for opt_count, optimization_level in enumerate(optimization_levels):
                logger.info(
                    f"Combination #{(count * len(lattice_files)) + opt_count + 1} of {len(lattice_files)*len(optimization_levels)}"
                )

                lattice_name = (
                    lattice_file.split("/")[-1].split(".")[0].replace("_", "-")
                )
                logger.info(f"Lattice: {lattice_name}; opt={optimization_level}")

                lattice = CollisionlessLattice(lattice_file, logger=dummy_logger)
                logger.info(
                    f"Lattice={lattice_file}, num_qubits={lattice.num_total_qubits}"
                )

                lattice = CollisionlessLattice(lattice_file)

                algorithm = CQLBM(lattice, logger=dummy_logger)
                compiler = CircuitCompiler(
                    compiler_platform, target_platform, logger=logger
                )

                compiler.compile(
                    compile_object=algorithm,
                    backend=backend,
                    optimization_level=optimization_level,
                )


if __name__ == "__main__":
    NUM_SHOTS = 2**14
    NUM_STEPS = 5
    ROOT_OUTPUT_DIR = "qlbm-output/benchmark-compilers"

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
    logging.config.fileConfig("demos/benchmarks/compilers.conf")
    logger = getLogger("qlbm")
    logger.info("Session: QISKIT")
    benchmark(
        lattice_files,
        logger,
        dummy_logger,
        "QISKIT",
        "QULACS",
        [0, 1, 2],
        None,
        num_repetitions=1,
    )

    logger.info("Session: TKET")
    benchmark(
        lattice_files,
        logger,
        dummy_logger,
        "TKET",
        "QULACS",
        [0, 1, 2],
        TketQulacsBackend(),
        num_repetitions=1,
    )
