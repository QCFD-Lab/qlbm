from qlbm.lattice import CollisionlessLattice

CollisionlessLattice(
    {
        "lattice": {"dim": {"x": 8, "y": 8}, "velocities": {"x": 4, "y": 4}},
        "geometry": [{"x": [5, 6], "y": [1, 2], "boundary": "bounceback"}],
    }
).circuit.draw("mpl")