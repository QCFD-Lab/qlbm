from qlbm.components.spacetime.collision.d2q4_old import SpaceTimeD2Q4CollisionOperator
from qlbm.lattice import SpaceTimeLattice

# Build an example lattice
lattice = SpaceTimeLattice(
    num_timesteps=1,
    lattice_data={
        "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "D2Q4"},
        "geometry": [],
    },
)

# Draw the collision operator for 1 time step
SpaceTimeD2Q4CollisionOperator(lattice=lattice, timestep=1).draw("mpl")