from qlbm.components.ab import ABGridMeasurement
from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {"dim": {"x": 32, "y": 8}, "velocities": "d2q9"},
        "geometry": [],
    }
)

ABGridMeasurement(lattice).draw("mpl")