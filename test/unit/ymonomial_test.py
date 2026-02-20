from qlbm.lattice.geometry.shapes.ymonomial import YMonomial
from qlbm.tools.utils import ComparatorMode


def test_ymonomial_stl_mesh_stays_within_lattice_bounds():
    ymonomial = YMonomial([2, 4], "bounceback", 2, ComparatorMode.LE)

    ymonomial_mesh = ymonomial.stl_mesh()
    vertices = ymonomial_mesh.vectors.reshape(-1, 3)
    max_coords = vertices.max(axis=0)
    min_coords = vertices.min(axis=0)

    assert min_coords[0] >= 0
    assert min_coords[1] >= 0
    assert max_coords[0] <= 2**ymonomial.num_grid_qubits[0] - 1
    assert max_coords[1] <= 2**ymonomial.num_grid_qubits[1] - 1
    assert max_coords[0] == 2**ymonomial.num_grid_qubits[0] - 1
    assert max_coords[1] > 9
