
import pytest
from qiskit_aer import AerSimulator

from qlbm.components.collisionless import (
    CQLBM,
    CollisionlessInitialConditions,
    GridMeasurement,
)
from qlbm.components.common import EmptyPrimitive
from qlbm.components.spacetime import (
    SpaceTimeGridVelocityMeasurement,
    SpaceTimeInitialConditions,
    SpaceTimeQLBM,
)
from qlbm.infra.runner import QiskitRunner
from qlbm.infra.runner.simulation_config import SimulationConfig
from qlbm.lattice import CollisionlessLattice
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice

OUTPUT_DIR = "test/artifacts"


@pytest.fixture
def collisionless_circuits():
    lattice = CollisionlessLattice("test/resources/symmetric_2d_1_obstacle.json")

    return {
        "initial_conditions": CollisionlessInitialConditions(lattice),
        "algorithm": CQLBM(lattice),
        "postprocessing": EmptyPrimitive(lattice),
        "measurement": GridMeasurement(lattice),
        "lattice": lattice,
    }


@pytest.fixture
def spacetime_circuits():
    lattice = SpaceTimeLattice(1, "test/resources/symmetric_2d_1_obstacle_q4.json")

    return {
        "initial_conditions": SpaceTimeInitialConditions(lattice),
        "algorithm": SpaceTimeQLBM(lattice),
        "postprocessing": EmptyPrimitive(lattice),
        "measurement": SpaceTimeGridVelocityMeasurement(lattice),
        "lattice": lattice,
    }


@pytest.mark.parametrize("statevector_sampling", [True, False])
def test_collisionless_qiskit_execution(
    collisionless_circuits,
    statevector_sampling,
):
    cfg = SimulationConfig(
        initial_conditions=collisionless_circuits["initial_conditions"],
        algorithm=collisionless_circuits["algorithm"],
        postprocessing=collisionless_circuits["postprocessing"],
        measurement=collisionless_circuits["measurement"],
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=0,
        execution_backend=AerSimulator(method="statevector"),
        sampling_backend=AerSimulator(method="statevector")
        if statevector_sampling
        else None,
        statevector_sampling=statevector_sampling,
    )

    cfg.validate()
    cfg.prepare_for_simulation()

    runner = QiskitRunner(cfg, collisionless_circuits["lattice"])

    # Simulate the circuits using both snapshots and sampling
    runner.run(
        2,  # Number of time steps
        2048,  # Number of shots per time step
        f"{OUTPUT_DIR}/collisionless-sampling-{int(statevector_sampling)}",
        statevector_snapshots=True,
    )


@pytest.mark.parametrize("statevector_sampling", [True, False])
def test_spacetime_qiskit_execution(
    spacetime_circuits,
    statevector_sampling,
):
    cfg = SimulationConfig(
        initial_conditions=spacetime_circuits["initial_conditions"],
        algorithm=spacetime_circuits["algorithm"],
        postprocessing=spacetime_circuits["postprocessing"],
        measurement=spacetime_circuits["measurement"],
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=0,
        execution_backend=AerSimulator(method="statevector"),
        sampling_backend=AerSimulator(method="statevector")
        if statevector_sampling
        else None,
        statevector_sampling=statevector_sampling,
    )

    cfg.validate()
    cfg.prepare_for_simulation()

    runner = QiskitRunner(cfg, spacetime_circuits["lattice"])

    # Simulate the circuits using both snapshots and sampling
    runner.run(
        2,  # Number of time steps
        512,  # Number of shots per time step
        f"{OUTPUT_DIR}/spacetime-sampling-{int(statevector_sampling)}",
        statevector_snapshots=True,
    )
