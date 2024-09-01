from qiskit_aer import AerSimulator

from qlbm.components import EmptyPrimitive
from qlbm.components.spacetime import (
    SpaceTimeGridVelocityMeasurement,
    SpaceTimeInitialConditions,
    SpaceTimeQLBM,
)
from qlbm.infra import QiskitRunner, SimulationConfig
from qlbm.lattice import SpaceTimeLattice
from qlbm.tools.utils import create_directory_and_parents

if __name__ == "__main__":
    # Number of shots to simulate for each timestep when running the circuit
    num_shots = 2**10
    # Number of shots to simulate
    num_steps = 5

    # Example with mixed boundary conditions
    lattice_file = "demos/lattices/2d_4x8_0_obstacle_q4.json"
    lattice_name = lattice_file.split("/")[-1].split(".")[0].replace("_", "-")
    output_dir = f"qlbm-output/spacetime-{lattice_name}-qiskit"

    create_directory_and_parents(output_dir)
    lattice = SpaceTimeLattice(1, lattice_file)

    space_time_circuit = SpaceTimeQLBM(lattice)
    space_time_circuit.draw('mpl', 'spacetime.pdf')

    cfg = SimulationConfig(
        initial_conditions=SpaceTimeInitialConditions(
            lattice, grid_data=[((1, 5), (True, True, True, True))]
        ),
        algorithm=SpaceTimeQLBM(lattice),
        postprocessing=EmptyPrimitive(lattice),
        measurement=SpaceTimeGridVelocityMeasurement(lattice),
        target_platform="QISKIT",
        compiler_platform="QISKIT",
        optimization_level=0,
        statevector_sampling=False,
        execution_backend=AerSimulator(method="statevector"),
        sampling_backend=AerSimulator(method="statevector"),
    )

    cfg.prepare_for_simulation()
    # Create a runner object to simulate the circuit
    runner = QiskitRunner(
        cfg,
        lattice,
    )

    # Simulate the circuits using both snapshots and sampling
    runner.run(
        num_steps,  # Number of time steps
        num_shots,  # Number of shots per time step
        output_dir,
        statevector_snapshots=False,
    )
