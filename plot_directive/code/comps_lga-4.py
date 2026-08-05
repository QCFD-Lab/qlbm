from qlbm.lattice import LQLGALattice
from qlbm.components.lqlga import LQGLAInitialConditions

lattice = LQLGALattice(
    {
        "lattice": {
            "dim": {"x": 4},
            "velocities": "D1Q3",
        },
        "geometry": [],
    },
)
initial_conditions = LQGLAInitialConditions(lattice, [(tuple([2]), (True, True, True))])
initial_conditions.draw("mpl")