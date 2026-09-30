from qlbm.components.lqlga import LQLGA
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

LQLGA(lattice=lattice).draw("mpl")