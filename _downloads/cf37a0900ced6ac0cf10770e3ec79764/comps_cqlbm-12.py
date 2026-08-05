from qlbm.components.ms import ControlledIncrementer
from qlbm.lattice import MSLattice

# Build an example lattice
lattice = MSLattice(
    {
        "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
        "geometry": [],
    }
)

# Streaming velocities indexed 2 in the y (1) dimension
ControlledIncrementer(lattice=lattice).draw("mpl")