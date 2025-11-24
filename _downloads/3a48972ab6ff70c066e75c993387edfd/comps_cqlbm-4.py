from qlbm.components.ms import MSInitialConditions3DSlim
from qlbm.lattice import MSLattice

# Build an example lattice
lattice = MSLattice({
    "lattice": {
        "dim": {
        "x": 8,
        "y": 8,
        "z": 8
        },
        "velocities": {
        "x": 4,
        "y": 4,
        "z": 4
        }
    },
    "geometry": []
})

# Draw the initial conditions circuit
MSInitialConditions3DSlim(lattice).draw("mpl")