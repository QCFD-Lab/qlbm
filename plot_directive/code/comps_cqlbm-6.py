from qlbm.components.ab import ABInitialConditions
from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
        "geometry": [],
    }
)

ABInitialConditions(lattice).circuit.decompose(reps=2).draw("mpl")