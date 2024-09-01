from qiskit_aer import AerSimulator

from qlbm.components import (
    CQLBM,
    CollisionlessInitialConditions,
    EmptyPrimitive,
    GridMeasurement,
)
from qlbm.infra import QulacsRunner, SimulationConfig
from qlbm.lattice import CollisionlessLattice
from qlbm.tools.utils import create_directory_and_parents

if __name__ == "__main__":
    # Number of shots to simulate for each timestep when running the circuit
    num_shots = 2**10
    # Number of shots to simulate
    num_steps = 20

    # Example with mixed boundary conditions
    lattice_file = "demos/lattices/2d_8x8_0_obstacle.json"
    lattice_name = lattice_file.split("/")[-1].split(".")[0].replace("_", "-")
    output_dir = f"qlbm-output/collisionless-{lattice_name}-qulacs"

    create_directory_and_parents(output_dir)
    lattice = CollisionlessLattice(lattice_file)

    cfg = SimulationConfig(
        initial_conditions=CollisionlessInitialConditions(lattice),
        algorithm=CQLBM(lattice),
        postprocessing=EmptyPrimitive(lattice),
        measurement=GridMeasurement(lattice),
        target_platform="QULACS",
        compiler_platform="TKET",
        optimization_level=0,
        statevector_sampling=True,
        execution_backend=None,
        sampling_backend=AerSimulator(method="statevector"),
    )

    cfg.prepare_for_simulation()
    # Create a runner object to simulate the circuit
    runner = QulacsRunner(
        cfg,
        lattice,
    )

    # Simulate the circuits using both snapshots and sampling
    runner.run(
        num_steps,  # Number of time steps
        num_shots,  # Number of shots per time step
        output_dir,
        statevector_snapshots=True,
    )
