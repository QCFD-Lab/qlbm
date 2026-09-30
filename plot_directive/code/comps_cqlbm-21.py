from qlbm.components.ab import ABZoneAgnosticReflectionOperator
from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
        "geometry": [
            {
                "shape": "cuboid",
                "x": [1, 3],
                "y": [1, 3],
                "boundary": "bounceback",
            }
        ],
    }
)

ABZoneAgnosticReflectionOperator(lattice).draw("mpl")