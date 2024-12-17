from qlbm.lattice import SpaceTimeLattice

SpaceTimeLattice(
    num_timesteps=1,
    lattice_data={
        "lattice": {"dim": {"x": 4, "y": 8}, "velocities": {"x": 2, "y": 2}},
        "geometry": [],
    }
).circuit.draw("mpl")