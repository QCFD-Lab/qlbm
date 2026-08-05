from qlbm.components.ms import MSStreamingOperator
from qlbm.lattice import MSLattice

# Build an example lattice
lattice = MSLattice(
    {
        "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
        "geometry": [],
    }
)

# Streaming the velocity with index 2
MSStreamingOperator(lattice=lattice, velocities=[2]).draw("mpl")