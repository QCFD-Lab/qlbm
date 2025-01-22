from qlbm.components.collisionless import CollisionlessInitialConditions3DSlim
from qlbm.lattice import CollisionlessLattice

# Build an example lattice
lattice = CollisionlessLattice({
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
CollisionlessInitialConditions3DSlim(lattice).draw("mpl")