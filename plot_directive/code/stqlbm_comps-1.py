from qlbm.components.spacetime import SpaceTimeQLBM
from qlbm.lattice import SpaceTimeLattice

# Build an example lattice
lattice = SpaceTimeLattice(
    num_timesteps=1,
    lattice_data={
        "lattice": {"dim": {"x": 4, "y": 8}, "velocities": {"x": 2, "y": 2}},
        "geometry": [],
    },
)

# Draw the end-to-end algorithm for 1 time step
SpaceTimeQLBM(lattice=lattice).draw("mpl")