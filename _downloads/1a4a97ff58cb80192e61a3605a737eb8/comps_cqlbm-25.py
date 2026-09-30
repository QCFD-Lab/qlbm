from qlbm.components.ab import ABGridMeasurement
from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {"dim": {"x": 16, "y": 8}, "velocities": "d2q9"},
        "geometry": [],
    }
)

ABGridMeasurement(lattice).draw("mpl")

# Include the four binary velocity-index qubits as well.
ABGridMeasurement(lattice, measure_velocity_qubits=True).draw("mpl")