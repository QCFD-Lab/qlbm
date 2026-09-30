from qlbm.components.lqlga import LQLGAGridVelocityMeasurement
from qlbm.lattice import LQLGALattice

lattice = LQLGALattice(
    {
        "lattice": {
            "dim": {"x": 5},
            "velocities": "D1Q3",
        },
        "geometry": [],
    },
)

LQLGAGridVelocityMeasurement(lattice=lattice).draw("mpl")