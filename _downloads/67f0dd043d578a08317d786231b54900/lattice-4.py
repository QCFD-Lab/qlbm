from qlbm.lattice import SpaceTimeLattice

SpaceTimeLattice(
    num_timesteps=1,
    lattice_data={
        "lattice": {"dim": {"x": 4, "y": 8}, "velocities": "D2Q4"},
        "geometry": [],
    }
).circuit.draw("mpl")