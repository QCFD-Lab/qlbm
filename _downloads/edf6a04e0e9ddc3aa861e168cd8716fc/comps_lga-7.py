from qlbm.components.lqlga import LQLGAReflectionOperator
from qlbm.lattice import LQLGALattice

lattice = LQLGALattice(
    {
        "lattice": {
            "dim": {"x": 7},
            "velocities": "D1Q3",
        },
        "geometry": [{"shape": "cuboid", "x": [3, 5], "boundary": "bounceback"}],
    },
)
reflection_operator = LQLGAReflectionOperator(
    lattice, shapes=lattice.shapes["bounceback"]
)
reflection_operator.draw("mpl")