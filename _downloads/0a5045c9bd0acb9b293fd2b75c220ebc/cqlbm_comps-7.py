from qlbm.components.collisionless import BounceBackReflectionOperator
from qlbm.lattice import CollisionlessLattice

# Build an example lattice
lattice = CollisionlessLattice(
    {
        "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
        "geometry": [{"x": [5, 6], "y": [1, 2], "boundary": "bounceback"}],
    }
)

BounceBackReflectionOperator(lattice=lattice, blocks=lattice.block_list)