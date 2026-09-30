from qlbm.lattice import ABLattice

lattice = ABLattice(
    {
        "lattice": {
            "dim": {"x": 16, "y": 16},
            "velocities": "D2Q9",
        },
    },
)

lattice.set_geometries(
    [
        [
            {
                "shape": "cuboid",
                "x": [4, 6],
                "y": [4, 6],
                "boundary": "bounceback",
            }
        ],
        [
            {
                "shape": "cuboid",
                "x": [9, 11],
                "y": [9, 11],
                "boundary": "bounceback",
            }
        ],
    ]
)

lattice.circuit.draw("mpl")