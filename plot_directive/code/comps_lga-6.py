from qlbm.components.lqlga import LQLGAStreamingOperator
from qlbm.lattice import LQLGALattice

lattice = LQLGALattice(
    {
        "lattice": {
            "dim": {"x": 4},
            "velocities": "D1Q3",
        },
        "geometry": [],
    },
)
streaming_operator = LQLGAStreamingOperator(lattice)
streaming_operator.draw("mpl")