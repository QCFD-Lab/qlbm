import numpy as np
import pytest

from qlbm.lattice.bgk import D2Q9AngleEncoding
from qlbm.lattice.lattices.ab_bgk_lattice import ABBGKLattice


@pytest.fixture
def encoding() -> D2Q9AngleEncoding:
    return D2Q9AngleEncoding()


@pytest.fixture
def ab_bgk_lattice_4x4() -> ABBGKLattice:
    return ABBGKLattice(
        {
            "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
            "geometry": [],
        }
    )


@pytest.fixture
def ab_bgk_lattice_4x4_obstacle() -> ABBGKLattice:
    return ABBGKLattice(
        {
            "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [1, 2],
                    "y": [1, 2],
                    "boundary": "bounceback",
                }
            ],
        }
    )


@pytest.fixture
def velocity_samples() -> list[tuple[float, float]]:
    return [
        (0.0, 0.0),
        (0.05, -0.02),
        (-0.1, 0.1),
        (0.09, 0.03),
        (-0.07, -0.08),
    ]


@pytest.fixture
def taylor_green_field_4x4() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(4, dtype=float)[:, None]
    y = np.arange(4, dtype=float)[None, :]
    amplitude = 0.05

    return (
        np.ones((4, 4)),
        amplitude * np.sin(np.pi * x / 2) * np.cos(np.pi * y / 2),
        -amplitude * np.cos(np.pi * x / 2) * np.sin(np.pi * y / 2),
    )
