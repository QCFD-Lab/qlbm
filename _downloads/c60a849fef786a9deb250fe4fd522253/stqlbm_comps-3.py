from qlbm.components.spacetime import SpaceTimeCollisionOperator
from qlbm.lattice import SpaceTimeLattice

# Build an example lattice
lattice = SpaceTimeLattice(
    num_timesteps=1,
    lattice_data={
        "lattice": {"dim": {"x": 4, "y": 8}, "velocities": {"x": 2, "y": 2}},
        "geometry": [],
    },
)

# Draw the collision operator for 1 time step
SpaceTimeCollisionOperator(lattice=lattice, timestep=1).draw("mpl")