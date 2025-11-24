from qlbm.components.ms import MSInitialConditions
from qlbm.lattice import MSLattice

# Build an example lattice
lattice = MSLattice({
    "lattice": {
        "dim": {
                "x": 8,
                "y": 8
            },
            "velocities": {
                "x": 4,
                "y": 4
        }
    },
    "geometry": [
        {
            "shape": "cuboid",
            "x": [5, 6],
            "y": [1, 2],
            "boundary": "specular"
        }
    ]
})

# Draw the initial conditions circuit
MSInitialConditions(lattice).draw("mpl")