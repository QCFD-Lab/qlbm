from qlbm.components.collisionless import EdgeComparator
from qlbm.lattice import CollisionlessLattice

# Build an example lattice
lattice = CollisionlessLattice(
    {
        "lattice": {
            "dim": {"x": 8, "y": 8, "z": 8},
            "velocities": {"x": 4, "y": 4, "z": 4},
        },
        "geometry": [{"x": [2, 5], "y": [2, 5], "z": [2, 5], "boundary": "specular"}],
    }
)

# Draw the edge comparator circuit for one specific corner edge
EdgeComparator(lattice, lattice.block_list[0].corner_edges_3d[0]).draw("mpl")