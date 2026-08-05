from qlbm.components.spacetime.initial import PointWiseSpaceTimeInitialConditions
from qlbm.lattice import SpaceTimeLattice

# Build an example lattice
lattice = SpaceTimeLattice(
    num_timesteps=1,
    lattice_data={
        "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "D2Q4"},
        "geometry": [],
    },
)

# Draw the initial conditions for two particles at (3, 7), traveling in the +y and -y directions
PointWiseSpaceTimeInitialConditions(lattice=lattice, grid_data=[((3, 7), (False, True, False, True))]).draw("mpl")