from qlbm.components.spacetime import SpaceTimeGridVelocityMeasurement
from qlbm.lattice import SpaceTimeLattice

# Build an example lattice
lattice = SpaceTimeLattice(
    num_timesteps=1,
    lattice_data={
        "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "D2Q4"},
        "geometry": [],
    },
)

# Draw the measurement circuit
SpaceTimeGridVelocityMeasurement(lattice=lattice).draw("mpl")