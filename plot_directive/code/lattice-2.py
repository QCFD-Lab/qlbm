from qlbm.lattice import ABLattice

ABLattice(
    {
        "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "D2Q9"},
        "geometry": [],
    }
).circuit.draw("mpl")