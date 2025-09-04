from qlbm.components.collisionless import CollisionlessStreamingOperator
from qlbm.lattice import CollisionlessLattice

# Build an example lattice
lattice = CollisionlessLattice(
    {
        "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
        "geometry": [],
    }
)

# Streaming the velocity with index 2
CollisionlessStreamingOperator(lattice=lattice, velocities=[2]).draw("mpl")