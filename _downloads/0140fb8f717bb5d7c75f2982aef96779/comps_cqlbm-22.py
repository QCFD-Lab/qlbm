from qlbm.components.ab.reflection import ABZoneAgnosticReflectionOracle
from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {"dim": {"x": 4, "y": 16}, "velocities": "d2q9"},
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

ABZoneAgnosticReflectionOracle(lattice, shape=lattice.shapes["bounceback"][0]).draw("mpl")