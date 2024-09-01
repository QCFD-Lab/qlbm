from qlbm.components.spacetime import (
    SpaceTimeInitialConditions,
    SpaceTimeQLBM,
    SpaceTimeStreamingOperator,
)
from qlbm.lattice import SpaceTimeLattice
from qlbm.tools.utils import create_directory_and_parents

if __name__ == "__main__":
    root_directory = "qlbm-output/visualization_spacetime_components"
    create_directory_and_parents(root_directory)
    # This is the lattice based on which we build operators and algorithms
    example_lattice = SpaceTimeLattice(1, "demos/lattices/2d_16x16_0_obstacle_q4.json")

    initial_conditions = SpaceTimeInitialConditions(example_lattice)

    # Also through Qiskit's Matplotlib interface
    initial_conditions.draw("mpl", f"{root_directory}/initial_conditions.pdf")

    SpaceTimeStreamingOperator(example_lattice, 1).draw(
        "mpl", f"{root_directory}/streaming_operator.pdf"
    )

    SpaceTimeQLBM(example_lattice).draw(
        "mpl", f"{root_directory}/stqlbm.pdf"
    )
