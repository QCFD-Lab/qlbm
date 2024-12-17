from qlbm.components.spacetime import SpaceTimeInitialConditions
from qlbm.lattice import SpaceTimeLattice

# Build an example lattice
lattice = SpaceTimeLattice(
    num_timesteps=1,
    lattice_data={
        "lattice": {"dim": {"x": 4, "y": 8}, "velocities": {"x": 2, "y": 2}},
        "geometry": [],
    },
)

# Draw the initial conditions for two particles at (3, 7), traveling in the +y and -y directions
SpaceTimeInitialConditions(lattice=lattice, grid_data=[((3, 7), (False, True, False, True))]).draw("mpl")