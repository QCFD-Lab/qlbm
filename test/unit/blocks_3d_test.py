from typing import List

import pytest

from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.geometry.encodings.collisionless import ReflectionWall


@pytest.fixture
def simple_3d_block():
    return Block([(5, 6), (2, 10), (3, 8)], [4, 4, 4], "specular")


@pytest.fixture
def simple_symmetric_3d_block():
    return Block([(2, 5), (2, 5), (2, 5)], [3, 3, 3], "bounceback")


def test_3d_mesh(simple_3d_block):
    mesh = simple_3d_block.stl_mesh()

    vertex_lll = [5, 2, 3]
    vertex_llu = [5, 2, 8]
    # vertex_lul = [5, 10, 3]
    # vertex_luu = [5, 10, 8]
    vertex_ull = [6, 2, 3]
    vertex_ulu = [6, 2, 8]
    # vertex_uul = [6, 10, 3]
    # vertex_uuu = [6, 10, 8]

    # print(mesh.vectors)
    # Face 1, +Y
    assert all(a == b for a, b in zip(vertex_lll, mesh.vectors[0][0]))
    assert all(a == b for a, b in zip(vertex_llu, mesh.vectors[0][1]))
    assert all(a == b for a, b in zip(vertex_ull, mesh.vectors[0][2]))

    assert all(a == b for a, b in zip(vertex_llu, mesh.vectors[1][0]))
    assert all(a == b for a, b in zip(vertex_ull, mesh.vectors[1][1]))
    assert all(a == b for a, b in zip(vertex_ulu, mesh.vectors[1][2]))

    # TODO: complete the remaining 5 faces


def test_3d_inner_x_wall_points(simple_3d_block):
    assert len(simple_3d_block.walls_inside) == 3

    x_walls: List[ReflectionWall] = simple_3d_block.walls_inside[0]
    assert len(x_walls) == 2

    # The correct dimension velocity is reflected
    assert all(wall.dim == 0 for wall in x_walls)

    # The alignment dimensions are correct
    assert all(wall.alignment_dims == [1, 2] for wall in x_walls)

    # The bounds of the alignment dimensions are correct
    assert all(wall.lower_bounds == [2, 3] for wall in x_walls)

    # The correct grid qubits are inverted for the lower bound wall
    assert x_walls[0].data.qubits_to_invert == [1, 3]
    assert x_walls[1].data.qubits_to_invert == [0, 3]
    assert not x_walls[0].data.invert_velocity
    assert x_walls[1].data.invert_velocity


def test_3d_outer_x_wall_points(simple_3d_block):
    assert len(simple_3d_block.walls_inside) == 3

    x_walls: List[ReflectionWall] = simple_3d_block.walls_outside[0]
    assert len(x_walls) == 2

    # The correct dimension velocity is reflected
    assert all(wall.dim == 0 for wall in x_walls)

    # The alignment dimensions are correct
    assert all(wall.alignment_dims == [1, 2] for wall in x_walls)

    # The bounds of the alignment dimensions are correct
    assert all(wall.lower_bounds == [2, 3] for wall in x_walls)

    # The correct grid qubits are inverted for the lower bound wall
    assert x_walls[0].data.qubits_to_invert == [0, 1, 3]
    assert x_walls[1].data.qubits_to_invert == [3]
    assert x_walls[0].data.invert_velocity
    assert not x_walls[1].data.invert_velocity


def test_3d_inner_y_wall_points(simple_3d_block):
    assert len(simple_3d_block.walls_inside) == 3

    y_walls: List[ReflectionWall] = simple_3d_block.walls_inside[1]
    assert len(y_walls) == 2

    # The correct dimension velocity is reflected
    assert all(wall.dim == 1 for wall in y_walls)

    # The alignment dimensions are correct
    assert all(wall.alignment_dims == [0, 2] for wall in y_walls)

    # The bounds of the alignment dimensions are correct
    assert all(wall.lower_bounds == [5, 3] for wall in y_walls)

    # The correct grid qubits are inverted for the lower bound wall
    assert y_walls[0].data.qubits_to_invert == [4, 6, 7]
    assert y_walls[1].data.qubits_to_invert == [4, 6]
    assert not y_walls[0].data.invert_velocity
    assert y_walls[1].data.invert_velocity


def test_3d_outer_y_wall_points(simple_3d_block):
    assert len(simple_3d_block.walls_inside) == 3

    y_walls: List[ReflectionWall] = simple_3d_block.walls_outside[1]
    assert len(y_walls) == 2

    # The correct dimension velocity is reflected
    assert all(wall.dim == 1 for wall in y_walls)

    # The alignment dimensions are correct
    assert all(wall.alignment_dims == [0, 2] for wall in y_walls)

    # The bounds of the alignment dimensions are correct
    assert all(wall.lower_bounds == [5, 3] for wall in y_walls)

    # The correct grid qubits are inverted for the lower bound wall
    assert y_walls[0].data.qubits_to_invert == [5, 6, 7]
    assert y_walls[1].data.qubits_to_invert == [6]
    assert y_walls[0].data.invert_velocity
    assert not y_walls[1].data.invert_velocity


def test_3d_inner_z_wall_points(simple_3d_block):
    assert len(simple_3d_block.walls_inside) == 3

    z_walls: List[ReflectionWall] = simple_3d_block.walls_inside[2]
    assert len(z_walls) == 2

    # The correct dimension velocity is reflected
    assert all(wall.dim == 2 for wall in z_walls)

    # The alignment dimensions are correct
    assert all(wall.alignment_dims == [0, 1] for wall in z_walls)

    # The bounds of the alignment dimensions are correct
    assert all(wall.lower_bounds == [5, 2] for wall in z_walls)

    # The correct grid qubits are inverted for the lower bound wall
    assert z_walls[0].data.qubits_to_invert == [10, 11]
    assert z_walls[1].data.qubits_to_invert == [8, 9, 10]
    assert not z_walls[0].data.invert_velocity
    assert z_walls[1].data.invert_velocity


def test_3d_outer_z_wall_points(simple_3d_block):
    assert len(simple_3d_block.walls_inside) == 3

    z_walls: List[ReflectionWall] = simple_3d_block.walls_outside[2]
    assert len(z_walls) == 2

    # The correct dimension velocity is reflected
    assert all(wall.dim == 2 for wall in z_walls)

    # The alignment dimensions are correct
    assert all(wall.alignment_dims == [0, 1] for wall in z_walls)

    # The bounds of the alignment dimensions are correct
    assert all(wall.lower_bounds == [5, 2] for wall in z_walls)

    # The correct grid qubits are inverted for the lower bound wall
    assert z_walls[0].data.qubits_to_invert == [8, 10, 11]
    assert z_walls[1].data.qubits_to_invert == [9, 10]
    assert z_walls[0].data.invert_velocity
    assert not z_walls[1].data.invert_velocity


def test_3d_corners_inside_qubits(simple_3d_block):
    assert len(simple_3d_block.corners_inside) == 8
    assert all(point.num_dims == 3 for point in simple_3d_block.corners_inside)
    assert all(
        point.dims_inside == [0, 1, 2] for point in simple_3d_block.corners_inside
    )
    assert all(point.dims_outside == [] for point in simple_3d_block.corners_inside)
    assert all(len(point.data) == 3 for point in simple_3d_block.corners_inside)
    assert all(
        not point.is_near_corner_point for point in simple_3d_block.corners_inside
    )

    # LLL
    assert simple_3d_block.corners_inside[0].data[0].qubits_to_invert == [1, 3]
    assert simple_3d_block.corners_inside[0].data[1].qubits_to_invert == [4, 6, 7]
    assert simple_3d_block.corners_inside[0].data[2].qubits_to_invert == [10, 11]

    # LLU
    assert simple_3d_block.corners_inside[1].data[0].qubits_to_invert == [1, 3]
    assert simple_3d_block.corners_inside[1].data[1].qubits_to_invert == [4, 6, 7]
    assert simple_3d_block.corners_inside[1].data[2].qubits_to_invert == [8, 9, 10]

    # LUL
    assert simple_3d_block.corners_inside[2].data[0].qubits_to_invert == [1, 3]
    assert simple_3d_block.corners_inside[2].data[1].qubits_to_invert == [4, 6]
    assert simple_3d_block.corners_inside[2].data[2].qubits_to_invert == [10, 11]

    # LUU
    assert simple_3d_block.corners_inside[3].data[0].qubits_to_invert == [1, 3]
    assert simple_3d_block.corners_inside[3].data[1].qubits_to_invert == [4, 6]
    assert simple_3d_block.corners_inside[3].data[2].qubits_to_invert == [8, 9, 10]

    # ULL
    assert simple_3d_block.corners_inside[4].data[0].qubits_to_invert == [0, 3]
    assert simple_3d_block.corners_inside[4].data[1].qubits_to_invert == [4, 6, 7]
    assert simple_3d_block.corners_inside[4].data[2].qubits_to_invert == [10, 11]

    # ULU
    assert simple_3d_block.corners_inside[5].data[0].qubits_to_invert == [0, 3]
    assert simple_3d_block.corners_inside[5].data[1].qubits_to_invert == [4, 6, 7]
    assert simple_3d_block.corners_inside[5].data[2].qubits_to_invert == [8, 9, 10]

    # UUL
    assert simple_3d_block.corners_inside[6].data[0].qubits_to_invert == [0, 3]
    assert simple_3d_block.corners_inside[6].data[1].qubits_to_invert == [4, 6]
    assert simple_3d_block.corners_inside[6].data[2].qubits_to_invert == [10, 11]

    # UUU
    assert simple_3d_block.corners_inside[7].data[0].qubits_to_invert == [0, 3]
    assert simple_3d_block.corners_inside[7].data[1].qubits_to_invert == [4, 6]
    assert simple_3d_block.corners_inside[7].data[2].qubits_to_invert == [8, 9, 10]


def test_3d_corners_outside_qubits(simple_3d_block):
    assert len(simple_3d_block.corners_outside) == 8
    assert all(point.num_dims == 3 for point in simple_3d_block.corners_outside)
    assert all(point.dims_inside == [] for point in simple_3d_block.corners_outside)
    assert all(
        point.dims_outside == [0, 1, 2] for point in simple_3d_block.corners_outside
    )
    assert all(len(point.data) == 3 for point in simple_3d_block.corners_outside)
    assert all(
        not point.is_near_corner_point for point in simple_3d_block.corners_outside
    )

    # LLL
    assert simple_3d_block.corners_outside[0].data[0].qubits_to_invert == [0, 1, 3]
    assert simple_3d_block.corners_outside[0].data[1].qubits_to_invert == [5, 6, 7]
    assert simple_3d_block.corners_outside[0].data[2].qubits_to_invert == [8, 10, 11]

    # LLU
    assert simple_3d_block.corners_outside[1].data[0].qubits_to_invert == [0, 1, 3]
    assert simple_3d_block.corners_outside[1].data[1].qubits_to_invert == [5, 6, 7]
    assert simple_3d_block.corners_outside[1].data[2].qubits_to_invert == [9, 10]

    # LUL
    assert simple_3d_block.corners_outside[2].data[0].qubits_to_invert == [0, 1, 3]
    assert simple_3d_block.corners_outside[2].data[1].qubits_to_invert == [6]
    assert simple_3d_block.corners_outside[2].data[2].qubits_to_invert == [8, 10, 11]

    # LUU
    assert simple_3d_block.corners_outside[3].data[0].qubits_to_invert == [0, 1, 3]
    assert simple_3d_block.corners_outside[3].data[1].qubits_to_invert == [6]
    assert simple_3d_block.corners_outside[3].data[2].qubits_to_invert == [9, 10]

    # ULL
    assert simple_3d_block.corners_outside[4].data[0].qubits_to_invert == [3]
    assert simple_3d_block.corners_outside[4].data[1].qubits_to_invert == [5, 6, 7]
    assert simple_3d_block.corners_outside[4].data[2].qubits_to_invert == [8, 10, 11]

    # ULU
    assert simple_3d_block.corners_outside[5].data[0].qubits_to_invert == [3]
    assert simple_3d_block.corners_outside[5].data[1].qubits_to_invert == [5, 6, 7]
    assert simple_3d_block.corners_outside[5].data[2].qubits_to_invert == [9, 10]

    # UUL
    assert simple_3d_block.corners_outside[6].data[0].qubits_to_invert == [3]
    assert simple_3d_block.corners_outside[6].data[1].qubits_to_invert == [6]
    assert simple_3d_block.corners_outside[6].data[2].qubits_to_invert == [8, 10, 11]

    # UUU
    assert simple_3d_block.corners_outside[7].data[0].qubits_to_invert == [3]
    assert simple_3d_block.corners_outside[7].data[1].qubits_to_invert == [6]
    assert simple_3d_block.corners_outside[7].data[2].qubits_to_invert == [9, 10]


def test_3d_corners_outside_inversions(simple_3d_block):
    assert len(simple_3d_block.corners_outside) == 8

    inversion_dict = {
        0: [True, True, True],  # LLL
        1: [True, True, False],  # LLU
        2: [True, False, True],  # LUL
        3: [True, False, False],  # LUU
        4: [False, True, True],  # ULL
        5: [False, True, False],  # ULU
        6: [False, False, True],  # UUL
        7: [False, False, False],  # UUU
    }

    for outside_corner_index in range(8):
        assert (
            simple_3d_block.corners_outside[
                outside_corner_index
            ].invert_velocity_in_dimension
            == inversion_dict[outside_corner_index]
        )


def test_3d_corner_edges_outside_qubits_xy(simple_3d_block):
    assert len(simple_3d_block.corner_edges_3d) == 12

    xy_corner_edges = simple_3d_block.corner_edges_3d[:4]

    # Check that the correct dimensions are joined
    assert all(
        a == b
        for a, b in zip(
            [edge.dims_of_edge for edge in xy_corner_edges], [(0, 1) for _ in range(4)]
        )
    )

    # Check that there is no orthogonal dimension of the
    # Ones that are joined
    assert all(
        a == b
        for a, b in zip(
            [edge.dimension_outside for edge in xy_corner_edges],
            [None for _ in range(4)],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [edge.bounds_disconnected_dim for edge in xy_corner_edges],
            [simple_3d_block.bounds[2] for _ in range(4)],
        )
    )

    # LxLy
    assert xy_corner_edges[0].walls_joining == [
        simple_3d_block.outside_points_data[0][0],
        simple_3d_block.outside_points_data[1][0],
    ]

    # LxUy
    assert xy_corner_edges[1].walls_joining == [
        simple_3d_block.outside_points_data[0][0],
        simple_3d_block.outside_points_data[1][1],
    ]

    # UxLy
    assert xy_corner_edges[2].walls_joining == [
        simple_3d_block.outside_points_data[0][1],
        simple_3d_block.outside_points_data[1][0],
    ]

    # UxUy
    assert xy_corner_edges[3].walls_joining == [
        simple_3d_block.outside_points_data[0][1],
        simple_3d_block.outside_points_data[1][1],
    ]


def test_3d_corner_edges_outside_inversions_xy(simple_3d_block):
    xy_corner_edges = simple_3d_block.corner_edges_3d[:4]
    inversion_dict = {
        0: (True, True),  # LxLy
        1: (True, False),  # LxUy
        2: (False, True),  # UxLy
        3: (False, False),  # UxUy
    }

    for xy_corner_edge_index in range(4):
        edge = xy_corner_edges[xy_corner_edge_index]
        assert edge.invert_velocity_in_dimension == inversion_dict[xy_corner_edge_index]

        assert len(edge.reflected_velocities) == 2

        assert edge.reflected_velocities == list(edge.dims_of_edge)


def test_3d_corner_edges_outside_qubits_xz(simple_3d_block):
    assert len(simple_3d_block.corner_edges_3d) == 12

    xz_corner_edges = simple_3d_block.corner_edges_3d[4:8]

    # Check that the correct dimensions are joined
    assert all(
        a == b
        for a, b in zip(
            [edge.dims_of_edge for edge in xz_corner_edges], [(0, 2) for _ in range(4)]
        )
    )

    # Check that there is no orthogonal dimension of the
    # Ones that are joined
    assert all(
        a == b
        for a, b in zip(
            [edge.dimension_outside for edge in xz_corner_edges],
            [None for _ in range(4)],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [edge.bounds_disconnected_dim for edge in xz_corner_edges],
            [simple_3d_block.bounds[1] for _ in range(4)],
        )
    )

    # LxLz
    assert xz_corner_edges[0].walls_joining == [
        simple_3d_block.outside_points_data[0][0],
        simple_3d_block.outside_points_data[2][0],
    ]

    # LxUz
    assert xz_corner_edges[1].walls_joining == [
        simple_3d_block.outside_points_data[0][0],
        simple_3d_block.outside_points_data[2][1],
    ]

    # UxLz
    assert xz_corner_edges[2].walls_joining == [
        simple_3d_block.outside_points_data[0][1],
        simple_3d_block.outside_points_data[2][0],
    ]

    # UxUz
    assert xz_corner_edges[3].walls_joining == [
        simple_3d_block.outside_points_data[0][1],
        simple_3d_block.outside_points_data[2][1],
    ]


def test_3d_corner_edges_outside_inversions_xz(simple_3d_block):
    xz_corner_edges = simple_3d_block.corner_edges_3d[4:8]
    inversion_dict = {
        0: (True, True),  # LxLz
        1: (True, False),  # LxUz
        2: (False, True),  # UxLz
        3: (False, False),  # UxUz
    }

    for xz_corner_edges_index in range(4):
        edge = xz_corner_edges[xz_corner_edges_index]
        assert (
            edge.invert_velocity_in_dimension == inversion_dict[xz_corner_edges_index]
        )

        assert len(edge.reflected_velocities) == 2

        assert edge.reflected_velocities == list(edge.dims_of_edge)


def test_3d_corner_edges_outside_qubits_yz(simple_3d_block):
    assert len(simple_3d_block.corner_edges_3d) == 12

    yz_corner_edges = simple_3d_block.corner_edges_3d[8:]

    # Check that the correct dimensions are joined
    assert all(
        a == b
        for a, b in zip(
            [edge.dims_of_edge for edge in yz_corner_edges], [(1, 2) for _ in range(4)]
        )
    )

    # Check that there is no orthogonal dimension of the
    # Ones that are joined
    assert all(
        a == b
        for a, b in zip(
            [edge.dimension_outside for edge in yz_corner_edges],
            [None for _ in range(4)],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [edge.bounds_disconnected_dim for edge in yz_corner_edges],
            [simple_3d_block.bounds[0] for _ in range(4)],
        )
    )

    # LyLz
    assert yz_corner_edges[0].walls_joining == [
        simple_3d_block.outside_points_data[1][0],
        simple_3d_block.outside_points_data[2][0],
    ]

    # LyUz
    assert yz_corner_edges[1].walls_joining == [
        simple_3d_block.outside_points_data[1][0],
        simple_3d_block.outside_points_data[2][1],
    ]

    # UyLz
    assert yz_corner_edges[2].walls_joining == [
        simple_3d_block.outside_points_data[1][1],
        simple_3d_block.outside_points_data[2][0],
    ]

    # UyUz
    assert yz_corner_edges[3].walls_joining == [
        simple_3d_block.outside_points_data[1][1],
        simple_3d_block.outside_points_data[2][1],
    ]


def test_3d_corner_edges_outside_inversions_yz(simple_3d_block):
    yz_corner_edges = simple_3d_block.corner_edges_3d[8:]
    inversion_dict = {
        0: (True, True),  # LxLz
        1: (True, False),  # LxUz
        2: (False, True),  # UxLz
        3: (False, False),  # UxUz
    }

    for yz_corner_edges_index in range(4):
        edge = yz_corner_edges[yz_corner_edges_index]
        assert (
            edge.invert_velocity_in_dimension == inversion_dict[yz_corner_edges_index]
        )

        assert len(edge.reflected_velocities) == 2

        assert edge.reflected_velocities == list(edge.dims_of_edge)


def test_3d_near_edge_points_qubits_xy(simple_3d_block):
    assert len(simple_3d_block.near_corner_edges_3d) == 24

    # xy between 5, 6 and 2, 10
    xy_wall_edges = simple_3d_block.near_corner_edges_3d[:8]
    assert all(
        a == b
        for a, b in zip(
            [edge.dims_of_edge for edge in xy_wall_edges], [(0, 1) for _ in range(8)]
        )
    )

    assert all(
        a is b
        for a, b in zip(
            [edge.dimension_outside for edge in xy_wall_edges],
            [
                i
                % 2  # The pattern of which dimension is outside the object bounds is alternating
                for i in range(8)
            ],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [edge.bounds_disconnected_dim for edge in xy_wall_edges],
            [simple_3d_block.bounds[2] for _ in range(8)],
        )
    )

    # lx, ly, perpendicular to lx
    assert xy_wall_edges[0].walls_joining == [
        simple_3d_block.outside_points_data[0][0],
        simple_3d_block.inside_points_data[1][0],
    ]

    # lx, ly, perpendicular to ly
    assert xy_wall_edges[1].walls_joining == [
        simple_3d_block.inside_points_data[0][0],
        simple_3d_block.outside_points_data[1][0],
    ]

    # lx, uy, perpendicular to lx
    assert xy_wall_edges[2].walls_joining == [
        simple_3d_block.outside_points_data[0][0],
        simple_3d_block.inside_points_data[1][1],
    ]

    # lx, uy, perpendicular to ly
    assert xy_wall_edges[3].walls_joining == [
        simple_3d_block.inside_points_data[0][0],
        simple_3d_block.outside_points_data[1][1],
    ]

    # lx, ly, perpendicular to lx
    assert xy_wall_edges[4].walls_joining == [
        simple_3d_block.outside_points_data[0][1],
        simple_3d_block.inside_points_data[1][0],
    ]

    # lx, ly, perpendicular to ly
    assert xy_wall_edges[5].walls_joining == [
        simple_3d_block.inside_points_data[0][1],
        simple_3d_block.outside_points_data[1][0],
    ]

    # lx, uy, perpendicular to lx
    assert xy_wall_edges[6].walls_joining == [
        simple_3d_block.outside_points_data[0][1],
        simple_3d_block.inside_points_data[1][1],
    ]

    # lx, uy, perpendicular to ly
    assert xy_wall_edges[7].walls_joining == [
        simple_3d_block.inside_points_data[0][1],
        simple_3d_block.outside_points_data[1][1],
    ]


def test_3d_near_edge_points_inversions_xy(simple_3d_block):
    xy_wall_edges = simple_3d_block.near_corner_edges_3d[:8]

    inversion_dict = {
        0: (True, False),  # LxLyPx
        1: (False, True),  # LxLyPy
        2: (True, True),  # LxUyPx
        3: (False, False),  # LxUyPy
        4: (False, False),  # UxLyPx
        5: (True, True),  # UxLyPy
        6: (False, True),  # UxUyPx
        7: (True, False),  # UxUyPy
    }

    for xy_edge_index in range(8):
        edge = xy_wall_edges[xy_edge_index]
        assert edge.invert_velocity_in_dimension == inversion_dict[xy_edge_index]

        assert len(edge.reflected_velocities) == 1

        assert edge.reflected_velocities[0] in edge.dims_of_edge

        assert edge.reflected_velocities[0] == edge.dims_of_edge[edge.dimension_outside]


def test_3d_near_edge_points_qubits_xz(simple_3d_block):
    assert len(simple_3d_block.near_corner_edges_3d) == 24

    # xz between 5, 6 and 3, 8
    xz_wall_edges = simple_3d_block.near_corner_edges_3d[8:16]
    assert all(
        a == b
        for a, b in zip(
            [edge.dims_of_edge for edge in xz_wall_edges], [(0, 2) for _ in range(8)]
        )
    )

    assert all(
        a is b
        for a, b in zip(
            [edge.dimension_outside for edge in xz_wall_edges],
            [
                i
                % 2  # The pattern of which dimension is outside the object bounds is alternating
                for i in range(8)
            ],
        )
    )

    # assert all(
    #     a == b
    #     for a, b in zip(
    #         [edge.walls_joining[0] for edge in xz_wall_edges],
    #         [
    #             simple_3d_block.outside_points_data[edge.dims_of_edge[0]][(c % 4) > 1]
    #             if edge.dimension_outside == 0
    #             else simple_3d_block.inside_points_data[edge.dims_of_edge[0]][
    #                 (c % 4) > 1
    #             ]
    #             for c, edge in enumerate(xz_wall_edges)
    #         ],
    #     )
    # )

    assert all(
        a == b
        for a, b in zip(
            [edge.bounds_disconnected_dim for edge in xz_wall_edges],
            [simple_3d_block.bounds[1] for _ in range(8)],
        )
    )

    # LxLzPx
    assert xz_wall_edges[0].walls_joining == [
        simple_3d_block.outside_points_data[0][0],
        simple_3d_block.inside_points_data[2][0],
    ]

    # LxLzPz
    assert xz_wall_edges[1].walls_joining == [
        simple_3d_block.inside_points_data[0][0],
        simple_3d_block.outside_points_data[2][0],
    ]

    # LxUzPx
    assert xz_wall_edges[2].walls_joining == [
        simple_3d_block.outside_points_data[0][0],
        simple_3d_block.inside_points_data[2][1],
    ]

    # LxUzPz
    assert xz_wall_edges[3].walls_joining == [
        simple_3d_block.inside_points_data[0][0],
        simple_3d_block.outside_points_data[2][1],
    ]

    # UxLzPx
    assert xz_wall_edges[4].walls_joining == [
        simple_3d_block.outside_points_data[0][1],
        simple_3d_block.inside_points_data[2][0],
    ]

    # UxLzPz
    assert xz_wall_edges[5].walls_joining == [
        simple_3d_block.inside_points_data[0][1],
        simple_3d_block.outside_points_data[2][0],
    ]

    # UxUzPx
    assert xz_wall_edges[6].walls_joining == [
        simple_3d_block.outside_points_data[0][1],
        simple_3d_block.inside_points_data[2][1],
    ]

    # UxUzPz
    assert xz_wall_edges[7].walls_joining == [
        simple_3d_block.inside_points_data[0][1],
        simple_3d_block.outside_points_data[2][1],
    ]


def test_3d_near_edge_points_inversions_xz(simple_3d_block):
    xz_wall_edges = simple_3d_block.near_corner_edges_3d[8:16]

    inversion_dict = {
        0: (True, False),  # LxLzPx
        1: (False, True),  # LxLzPz
        2: (True, True),  # LxUzPx
        3: (False, False),  # LxUzPz
        4: (False, False),  # UxLzPx
        5: (True, True),  # UxLzPz
        6: (False, True),  # UxUzPx
        7: (True, False),  # UxUzPz
    }

    for xz_edge_index in range(8):
        edge = xz_wall_edges[xz_edge_index]
        assert edge.invert_velocity_in_dimension == inversion_dict[xz_edge_index]

        assert len(edge.reflected_velocities) == 1

        assert edge.reflected_velocities[0] in edge.dims_of_edge

        assert edge.reflected_velocities[0] == edge.dims_of_edge[edge.dimension_outside]


def test_3d_near_edge_points_qubits_yz(simple_3d_block):
    assert len(simple_3d_block.near_corner_edges_3d) == 24

    # xz between 5, 6 and 3, 8
    yz_wall_edges = simple_3d_block.near_corner_edges_3d[16:]
    assert all(
        a == b
        for a, b in zip(
            [edge.dims_of_edge for edge in yz_wall_edges], [(1, 2) for _ in range(8)]
        )
    )

    assert all(
        a is b
        for a, b in zip(
            [edge.dimension_outside for edge in yz_wall_edges],
            [
                i
                % 2  # The pattern of which dimension is outside the object bounds is alternating
                for i in range(8)
            ],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [edge.bounds_disconnected_dim for edge in yz_wall_edges],
            [simple_3d_block.bounds[0] for _ in range(8)],
        )
    )

    # LyLzPy
    assert yz_wall_edges[0].walls_joining == [
        simple_3d_block.outside_points_data[1][0],
        simple_3d_block.inside_points_data[2][0],
    ]

    # LyLzPz
    assert yz_wall_edges[1].walls_joining == [
        simple_3d_block.inside_points_data[1][0],
        simple_3d_block.outside_points_data[2][0],
    ]

    # LyUzPy
    assert yz_wall_edges[2].walls_joining == [
        simple_3d_block.outside_points_data[1][0],
        simple_3d_block.inside_points_data[2][1],
    ]

    # LyUzPz
    assert yz_wall_edges[3].walls_joining == [
        simple_3d_block.inside_points_data[1][0],
        simple_3d_block.outside_points_data[2][1],
    ]

    # UyLzPy
    assert yz_wall_edges[4].walls_joining == [
        simple_3d_block.outside_points_data[1][1],
        simple_3d_block.inside_points_data[2][0],
    ]

    # UyLzPz
    assert yz_wall_edges[5].walls_joining == [
        simple_3d_block.inside_points_data[1][1],
        simple_3d_block.outside_points_data[2][0],
    ]

    # UyUzPy
    assert yz_wall_edges[6].walls_joining == [
        simple_3d_block.outside_points_data[1][1],
        simple_3d_block.inside_points_data[2][1],
    ]

    # UyUzPz
    assert yz_wall_edges[7].walls_joining == [
        simple_3d_block.inside_points_data[1][1],
        simple_3d_block.outside_points_data[2][1],
    ]


def test_3d_near_edge_points_inversions_yz(simple_3d_block):
    yz_wall_edges = simple_3d_block.near_corner_edges_3d[16:]

    inversion_dict = {
        0: (True, False),  # LyLzPy
        1: (False, True),  # LyLzPz
        2: (True, True),  # LyUzPy
        3: (False, False),  # LyUzPz
        4: (False, False),  # UyLzPy
        5: (True, True),  # UyLzPz
        6: (False, True),  # UyUzPy
        7: (True, False),  # UyUzPz
    }

    for yz_edge_index in range(8):
        edge = yz_wall_edges[yz_edge_index]
        assert edge.invert_velocity_in_dimension == inversion_dict[yz_edge_index]

        assert len(edge.reflected_velocities) == 1

        assert edge.reflected_velocities[0] in edge.dims_of_edge

        assert edge.reflected_velocities[0] == edge.dims_of_edge[edge.dimension_outside]


def test_overlapping_near_corner_edge_points_x_wall(simple_3d_block):
    assert len(simple_3d_block.overlapping_near_corner_edge_points_3d) == 24

    x_wall_points = simple_3d_block.overlapping_near_corner_edge_points_3d[:8]
    negative_x_wall_points = x_wall_points[:4]
    positive_x_wall_points = x_wall_points[4:]

    expected_inversions = {
        0: [True, False, False],
        1: [True, False, True],
        2: [True, True, False],
        3: [True, True, True],
        4: [False, False, False],
        5: [False, False, True],
        6: [False, True, False],
        7: [False, True, True],
    }

    assert all(
        a == b
        for a, b in zip(
            [p.dims_inside for p in x_wall_points], [[1, 2] for _ in range(8)]
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [p.dims_outside for p in x_wall_points], [[0] for _ in range(8)]
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [expected_inversions[i] for i in range(4)],
            [p.invert_velocity_in_dimension for p in negative_x_wall_points],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [expected_inversions[i] for i in range(4, 8)],
            [p.invert_velocity_in_dimension for p in positive_x_wall_points],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [p.data[0] for p in negative_x_wall_points],
            [simple_3d_block.outside_points_data[0][0]],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [p.data[0] for p in positive_x_wall_points],
            [simple_3d_block.outside_points_data[0][1]],
        )
    )

    for near_edge_overlapping_point_type in (
        negative_x_wall_points,
        positive_x_wall_points,
    ):
        assert near_edge_overlapping_point_type[0].data[1:] == [
            simple_3d_block.inside_points_data[1][0],
            simple_3d_block.inside_points_data[2][0],
        ]

        assert near_edge_overlapping_point_type[1].data[1:] == [
            simple_3d_block.inside_points_data[1][0],
            simple_3d_block.inside_points_data[2][1],
        ]

        assert near_edge_overlapping_point_type[2].data[1:] == [
            simple_3d_block.inside_points_data[1][1],
            simple_3d_block.inside_points_data[2][0],
        ]

        assert near_edge_overlapping_point_type[3].data[1:] == [
            simple_3d_block.inside_points_data[1][1],
            simple_3d_block.inside_points_data[2][1],
        ]


def test_overlapping_near_corner_edge_points_y_wall(simple_3d_block):
    assert len(simple_3d_block.overlapping_near_corner_edge_points_3d) == 24

    y_wall_points = simple_3d_block.overlapping_near_corner_edge_points_3d[8:16]
    negative_y_wall_points = y_wall_points[:4]
    positive_y_wall_points = y_wall_points[4:]

    expected_inversions = {
        0: [False, True, False],
        1: [False, True, True],
        2: [True, True, False],
        3: [True, True, True],
        4: [False, False, False],
        5: [False, False, True],
        6: [True, False, False],
        7: [True, False, True],
    }

    assert all(
        a == b
        for a, b in zip(
            [p.dims_inside for p in y_wall_points], [[0, 2] for _ in range(8)]
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [p.dims_outside for p in y_wall_points], [[1] for _ in range(8)]
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [expected_inversions[i] for i in range(4)],
            [p.invert_velocity_in_dimension for p in negative_y_wall_points],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [expected_inversions[i] for i in range(4, 8)],
            [p.invert_velocity_in_dimension for p in positive_y_wall_points],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [p.data[1] for p in negative_y_wall_points],
            [simple_3d_block.outside_points_data[1][0]],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [p.data[1] for p in positive_y_wall_points],
            [simple_3d_block.outside_points_data[1][1]],
        )
    )

    for near_edge_overlapping_point_type in (
        negative_y_wall_points,
        positive_y_wall_points,
    ):
        assert near_edge_overlapping_point_type[0].data[::2] == [
            simple_3d_block.inside_points_data[0][0],
            simple_3d_block.inside_points_data[2][0],
        ]

        assert near_edge_overlapping_point_type[1].data[::2] == [
            simple_3d_block.inside_points_data[0][0],
            simple_3d_block.inside_points_data[2][1],
        ]

        assert near_edge_overlapping_point_type[2].data[::2] == [
            simple_3d_block.inside_points_data[0][1],
            simple_3d_block.inside_points_data[2][0],
        ]

        assert near_edge_overlapping_point_type[3].data[::2] == [
            simple_3d_block.inside_points_data[0][1],
            simple_3d_block.inside_points_data[2][1],
        ]


def test_overlapping_near_corner_edge_points_z_wall(simple_3d_block):
    assert len(simple_3d_block.overlapping_near_corner_edge_points_3d) == 24

    z_wall_points = simple_3d_block.overlapping_near_corner_edge_points_3d[16:]
    negative_z_wall_points = z_wall_points[:4]
    positive_z_wall_points = z_wall_points[4:]

    expected_inversions = {
        0: [False, False, True],
        1: [False, True, True],
        2: [True, False, True],
        3: [True, True, True],
        4: [False, False, False],
        5: [False, True, False],
        6: [True, False, False],
        7: [True, True, False],
    }

    assert all(
        a == b
        for a, b in zip(
            [p.dims_inside for p in z_wall_points], [[0, 1] for _ in range(8)]
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [p.dims_outside for p in z_wall_points], [[2] for _ in range(8)]
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [expected_inversions[i] for i in range(4)],
            [p.invert_velocity_in_dimension for p in negative_z_wall_points],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [expected_inversions[i] for i in range(4, 8)],
            [p.invert_velocity_in_dimension for p in positive_z_wall_points],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [p.data[2] for p in negative_z_wall_points],
            [simple_3d_block.outside_points_data[2][0]],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [p.data[2] for p in positive_z_wall_points],
            [simple_3d_block.outside_points_data[2][1]],
        )
    )

    for near_edge_overlapping_point_type in (
        negative_z_wall_points,
        positive_z_wall_points,
    ):
        assert near_edge_overlapping_point_type[0].data[:2] == [
            simple_3d_block.inside_points_data[0][0],
            simple_3d_block.inside_points_data[1][0],
        ]

        assert near_edge_overlapping_point_type[1].data[:2] == [
            simple_3d_block.inside_points_data[0][0],
            simple_3d_block.inside_points_data[1][1],
        ]

        assert near_edge_overlapping_point_type[2].data[:2] == [
            simple_3d_block.inside_points_data[0][1],
            simple_3d_block.inside_points_data[1][0],
        ]

        assert near_edge_overlapping_point_type[3].data[:2] == [
            simple_3d_block.inside_points_data[0][1],
            simple_3d_block.inside_points_data[1][1],
        ]
