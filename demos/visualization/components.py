from qlbm.components import (
    CQLBM,
    CollisionlessStreamingOperator,
    ControlledIncrementer,
    SpecularReflectionOperator,
    SpeedSensitivePhaseShift,
)
from qlbm.lattice import CollisionlessLattice
from qlbm.tools.utils import create_directory_and_parents

if __name__ == "__main__":
    root_directory = "qlbm-output/visualization_components"
    create_directory_and_parents(root_directory)
    # This is the lattice based on which we build operators and algorithms
    example_lattice = CollisionlessLattice("demos/lattices/2d_8x8_1_obstacle.json")

    speed_shift_primitive: SpeedSensitivePhaseShift = SpeedSensitivePhaseShift(
        4, 2, True
    )

    # You can draw circuits in Qiskit's ASCII art format
    speed_shift_primitive.draw("text", f"{root_directory}/phase_shift.txt")

    # Also through Qiskit's Matplotlib interface
    speed_shift_primitive.draw("mpl", f"{root_directory}/phase_shift.pdf")

    # Can also export directly to Latex source
    speed_shift_primitive.draw("latex_source", f"{root_directory}/phase_shift.tex")

    # All primitives can be drawn to the same interface
    ControlledIncrementer(example_lattice, reflection=False).draw(
        "mpl", f"{root_directory}/controlled_incrementer.pdf"
    )

    # All operators can be drawn the same way
    CollisionlessStreamingOperator(example_lattice, [0, 2, 3]).draw(
        "mpl", f"{root_directory}/streaming.pdf"
    )
    SpecularReflectionOperator(
        example_lattice, example_lattice.blocks["bounceback"]
    ).draw("mpl", f"{root_directory}/specular_reflection.pdf")

    # As can entire algorithms
    CQLBM(example_lattice).draw("mpl", f"{root_directory}/collisionless_lbm.pdf")
