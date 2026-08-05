from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {
            "dim": {"x": 16, "y": 16},
            "velocities": "D2Q9",
        },
    },
)

lattice.circuit.draw("mpl")