from qlbm.components.ms import BounceBackWallComparator
from qlbm.lattice import MSLattice

# Build an example lattice
lattice = MSLattice(
    {
        "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
        "geometry": [{"shape":"cuboid", "x": [5, 6], "y": [1, 2], "boundary": "bounceback"}],
    }
)

# Comparing on the indices of the inside x-wall on the lower-bound of the obstacle
BounceBackWallComparator(
    lattice=lattice, wall=lattice.shape_list[0].walls_inside[0][0]
).draw("mpl")