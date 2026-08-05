from qlbm.components.ms import StreamingAncillaPreparation
from qlbm.lattice import MSLattice

# Build an example lattice
lattice = MSLattice(
    {
        "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
        "geometry": [],
    }
)

# Streaming velocities indexed 2 in the y (1) dimension
StreamingAncillaPreparation(lattice=lattice, velocities=[2], dim=1).draw("mpl")