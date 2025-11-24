from qlbm.lattice import LQLGALattice

lattice = LQLGALattice(
    {
        "lattice": {
            "dim": {"x": 5},
            "velocities": "D1Q2",
        },
    },
)

lattice.set_geometries(
    [
        [{"shape": "cuboid", "x": [3, 4], "boundary": "bounceback"}],
        [{"shape": "cuboid", "x": [1, 2], "boundary": "specular"}],
        [{"shape": "cuboid", "x": [1, 4], "boundary": "specular"}],
    ]
)

lattice.circuit.draw("mpl")