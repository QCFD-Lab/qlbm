from typing import List

import pytest

from qlbm.lattice.geometry.encodings.collisionless import ReflectionPoint
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.tools.utils import flatten
from test.regression.blocks import Block2D as RegressionBlock2D


@pytest.fixture
def simple_2d_block():
    return Block([(5, 6), (2, 10)], [4, 4], "specular")


@pytest.fixture
def simple_3d_block():
    return Block([(5, 6), (2, 10), (3, 8)], [4, 4, 4], "specular")


@pytest.fixture
def simple_regression_2d_block():
    return RegressionBlock2D(5, 6, 2, 10, 4, 4)


def test_2d_mesh(simple_2d_block):
    mesh = simple_2d_block.stl_mesh()

    vertex_ll = [5, 2, 1]
    vertex_lu = [5, 10, 1]
    vertex_ul = [6, 2, 1]
    vertex_uu = [6, 10, 1]

    assert all(a == b for a, b in zip(vertex_ll, mesh.vectors[0][0]))
    assert all(a == b for a, b in zip(vertex_lu, mesh.vectors[0][1]))
    assert all(a == b for a, b in zip(vertex_ul, mesh.vectors[0][2]))

    assert all(a == b for a, b in zip(vertex_lu, mesh.vectors[1][0]))
    assert all(a == b for a, b in zip(vertex_ul, mesh.vectors[1][1]))
    assert all(a == b for a, b in zip(vertex_uu, mesh.vectors[1][2]))


def test_regression_2d_inner_qubits(simple_2d_block, simple_regression_2d_block):
    assert len(simple_2d_block.inside_points_data) == 2

    flattened_inner_qubit_list = flatten(simple_2d_block.inside_points_data)

    assert len(flattened_inner_qubit_list) == 4

    expected_name_list = ["swap_lx_in", "swap_ux_in", "swap_ly_in", "swap_uy_in"]
    assert all(
        a == b
        for a, b in zip(
            list(
                map(
                    lambda reflection_data: reflection_data.name,
                    flattened_inner_qubit_list,
                )
            ),
            expected_name_list,
        )
    )

    assert all(
        a == b
        for a, b in zip(
            simple_regression_2d_block.to_swap_lx,
            flattened_inner_qubit_list[0].qubits_to_invert,
        )
    )

    assert all(
        a == b
        for a, b in zip(
            simple_regression_2d_block.to_swap_ux,
            flattened_inner_qubit_list[1].qubits_to_invert,
        )
    )

    assert all(
        a == b
        for a, b in zip(
            simple_regression_2d_block.to_swap_ly,
            flattened_inner_qubit_list[2].qubits_to_invert,
        )
    )

    assert all(
        a == b
        for a, b in zip(
            simple_regression_2d_block.to_swap_uy,
            flattened_inner_qubit_list[3].qubits_to_invert,
        )
    )


def test_regression_2d_outer_qubits(simple_2d_block, simple_regression_2d_block):
    assert len(simple_2d_block.outside_points_data) == 2

    flattened_outer_qubit_list = flatten(simple_2d_block.outside_points_data)

    assert len(flattened_outer_qubit_list) == 4

    expected_name_list = ["swap_lx_out", "swap_ux_out", "swap_ly_out", "swap_uy_out"]

    assert all(
        a == b
        for a, b in zip(
            list(
                map(
                    lambda reflection_data: reflection_data.name,
                    flattened_outer_qubit_list,
                )
            ),
            expected_name_list,
        )
    )

    assert all(
        a == b
        for a, b in zip(
            simple_regression_2d_block.to_swap_corner_lx,
            flattened_outer_qubit_list[0].qubits_to_invert,
        )
    )

    assert all(
        a == b
        for a, b in zip(
            simple_regression_2d_block.to_swap_corner_ux,
            flattened_outer_qubit_list[1].qubits_to_invert,
        )
    )

    assert all(
        a == b
        for a, b in zip(
            simple_regression_2d_block.to_swap_corner_ly,
            flattened_outer_qubit_list[2].qubits_to_invert,
        )
    )

    assert all(
        a == b
        for a, b in zip(
            simple_regression_2d_block.to_swap_corner_uy,
            flattened_outer_qubit_list[3].qubits_to_invert,
        )
    )


def test_regression_2d_wall_inside_qubits(simple_2d_block, simple_regression_2d_block):
    assert len(simple_2d_block.walls_inside) == 2

    assert len(flatten(simple_2d_block.walls_inside)) == 4

    # Check whether the correct qubits are encoded
    assert all(
        a == b
        for a, b in zip(
            flatten(
                [
                    simple_regression_2d_block.x_walls[wall].invert
                    for wall in range(len(simple_regression_2d_block.x_walls))
                ]
            ),
            flatten(
                [
                    simple_2d_block.walls_inside[0][wall].data.qubits_to_invert
                    for wall in range(len(simple_regression_2d_block.x_walls))
                ]
            ),
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [
                simple_2d_block.walls_inside[0][wall].dim
                for wall in range(len(simple_regression_2d_block.x_walls))
            ],
            [0 for _ in range(len(simple_regression_2d_block.x_walls))],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            flatten(
                [
                    simple_regression_2d_block.y_walls[wall].invert
                    for wall in range(len(simple_regression_2d_block.y_walls))
                ]
            ),
            flatten(
                [
                    simple_2d_block.walls_inside[1][wall].data.qubits_to_invert
                    for wall in range(len(simple_regression_2d_block.y_walls))
                ]
            ),
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [
                simple_2d_block.walls_inside[1][wall].dim
                for wall in range(len(simple_regression_2d_block.y_walls))
            ],
            [1 for _ in range(len(simple_regression_2d_block.y_walls))],
        )
    )

    # Check whether the correct velocity flips are encoded
    # In earlier versions, we encoded the inverse of the velocity flip,
    # Hence the `not` in the lines below
    assert all(
        simple_regression_2d_block.x_walls[wall].direction
        == (not simple_2d_block.walls_inside[0][wall].data.invert_velocity)
        for wall in range(len(simple_regression_2d_block.x_walls))
    )

    assert all(
        simple_regression_2d_block.y_walls[wall].direction
        == (not simple_2d_block.walls_inside[1][wall].data.invert_velocity)
        for wall in range(len(simple_regression_2d_block.x_walls))
    )


def test_regression_2d_wall_outside_qubits(simple_2d_block, simple_regression_2d_block):
    assert len(simple_2d_block.walls_outside) == 2

    assert len(flatten(simple_2d_block.walls_outside)) == 4

    # Check whether the correct qubits are encoded
    assert all(
        a == b
        for a, b in zip(
            flatten(
                [
                    simple_regression_2d_block.outside_x[wall].invert
                    for wall in range(len(simple_regression_2d_block.outside_x))
                ]
            ),
            flatten(
                [
                    simple_2d_block.walls_outside[0][wall].data.qubits_to_invert
                    for wall in range(len(simple_regression_2d_block.outside_x))
                ]
            ),
        )
    )

    # Check whether the correct qubits are encoded
    assert all(
        a == b
        for a, b in zip(
            flatten(
                [
                    simple_regression_2d_block.outside_y[wall].invert
                    for wall in range(len(simple_regression_2d_block.outside_y))
                ]
            ),
            flatten(
                [
                    simple_2d_block.walls_outside[1][wall].data.qubits_to_invert
                    for wall in range(len(simple_regression_2d_block.outside_y))
                ]
            ),
        )
    )

    # Check whether the correct velocity flips are encoded
    # In earlier versions, we encoded the inverse of the velocity flip,
    # Hence the `not` in the lines below
    assert all(
        simple_regression_2d_block.outside_x[wall].direction
        == (not simple_2d_block.walls_outside[0][wall].data.invert_velocity)
        for wall in range(len(simple_regression_2d_block.outside_x))
    )

    assert all(
        simple_regression_2d_block.outside_y[wall].direction
        == (not simple_2d_block.walls_outside[1][wall].data.invert_velocity)
        for wall in range(len(simple_regression_2d_block.outside_y))
    )


def test_regression_2d_corners_inside(simple_2d_block, simple_regression_2d_block):
    assert len(simple_2d_block.corners_inside) == 4

    # Reversing the order as to iterate by y first
    reordered_regression_points = list(
        reversed(simple_regression_2d_block.corner_points)
    )

    # Check whether the correct qubits are encoded
    assert all(
        a == b
        for a, b in zip(
            flatten(
                [
                    reordered_regression_points[corner].point_list
                    for corner in range(len(reordered_regression_points))
                ]
            ),
            flatten(
                [
                    simple_2d_block.corners_inside[corner_index].qubits_to_invert
                    for corner_index in range(len(reordered_regression_points))
                ]
            ),
        )
    )


def test_regression_2d_corners_outside(simple_2d_block, simple_regression_2d_block):
    assert len(simple_2d_block.corners_outside) == 4

    # Check whether the correct qubits are encoded
    assert all(
        a == b
        for a, b in zip(
            flatten(
                [
                    simple_regression_2d_block.outside_corner_points[corner].point_list
                    for corner in range(
                        len(simple_regression_2d_block.outside_corner_points)
                    )
                ]
            ),
            flatten(
                [
                    simple_2d_block.corners_outside[corner_index].qubits_to_invert
                    for corner_index in range(
                        len(simple_regression_2d_block.outside_corner_points)
                    )
                ]
            ),
        )
    )


def test_near_2d_corner_flag(simple_2d_block):
    assert all(
        [
            (not inside_corner_data.is_near_corner_point)
            for inside_corner_data in simple_2d_block.corners_inside
        ]
    )

    assert all(
        [
            (not outside_corner_data.is_near_corner_point)
            for outside_corner_data in simple_2d_block.corners_outside
        ]
    )

    assert all(
        [
            (near_corner_point.is_near_corner_point)
            for near_corner_point in simple_2d_block.near_corner_points_2d
        ]
    )


def test_2d_corners_inside_reflection_dimensions(simple_2d_block):
    # Inside corner are inside object bounds in all dimensions
    assert all(
        a == b
        for a, b in zip(
            [
                inside_corner_data.dims_inside
                for inside_corner_data in simple_2d_block.corners_inside
            ],
            [[0, 1] for _ in range(len(simple_2d_block.corners_inside))],
        )
    )

    assert all(
        a == b
        for a, b in zip(  # type: ignore
            [
                inside_corner_data.dims_outside
                for inside_corner_data in simple_2d_block.corners_inside
            ],
            [[] for _ in range(len(simple_2d_block.corners_inside))],
        )
    )


def test_2d_corners_outside_reflection_dimensions(simple_2d_block):
    # Outside corner are outside object bounds in all dimensions
    assert all(
        a == b
        for a, b in zip(  # type: ignore
            [
                inside_corner_data.dims_inside
                for inside_corner_data in simple_2d_block.corners_outside
            ],
            [[] for _ in range(len(simple_2d_block.corners_outside))],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [
                inside_corner_data.dims_outside
                for inside_corner_data in simple_2d_block.corners_outside
            ],
            [[0, 1] for _ in range(len(simple_2d_block.corners_outside))],
        )
    )


def test_2d_near_corner_points_x_wall_reflection_dimensions(simple_2d_block):
    assert len(simple_2d_block.near_corner_points_2d) == 8
    x_wall_points: List[ReflectionPoint] = simple_2d_block.near_corner_points_2d[:4]

    # X-wall points are outside objects bounds in the x-dimension
    # And inside y-dimension bounds
    assert all(
        a == b
        for a, b in zip(
            [x_wall_point.dims_inside for x_wall_point in x_wall_points],
            [[1] for _ in range(len(x_wall_points))],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [x_wall_point.dims_outside for x_wall_point in x_wall_points],
            [[0] for _ in range(len(x_wall_points))],
        )
    )


def test_2d_near_corner_points_y_wall_reflection_dimensions(simple_2d_block):
    assert len(simple_2d_block.near_corner_points_2d) == 8
    y_wall_points: List[ReflectionPoint] = simple_2d_block.near_corner_points_2d[4:]
    # Y-wall points are outside objects bounds in the y-dimension
    # And inside x-dimension bounds
    assert all(
        a == b
        for a, b in zip(
            [y_wall_point.dims_inside for y_wall_point in y_wall_points],
            [[0] for _ in range(len(y_wall_points))],
        )
    )

    assert all(
        a == b
        for a, b in zip(
            [y_wall_point.dims_outside for y_wall_point in y_wall_points],
            [[1] for _ in range(len(y_wall_points))],
        )
    )


def test_2d_near_corner_points_x_wall_velocities(simple_2d_block):
    assert len(simple_2d_block.near_corner_points_2d) == 8
    x_wall_points: List[ReflectionPoint] = simple_2d_block.near_corner_points_2d[:4]

    # Resetting the object ancilla for points that are not reflected
    # This is the red arrows in figure 9 of the paper
    assert x_wall_points[0].invert_velocity_in_dimension == [True, False]  # lb, lb
    assert x_wall_points[1].invert_velocity_in_dimension == [True, True]  # lb, ub
    assert x_wall_points[2].invert_velocity_in_dimension == [False, False]  # ub, lb
    assert x_wall_points[3].invert_velocity_in_dimension == [False, True]  # ub, ub


def test_2d_near_corner_points_y_wall_velocities(simple_2d_block):
    assert len(simple_2d_block.near_corner_points_2d) == 8
    y_wall_points: List[ReflectionPoint] = simple_2d_block.near_corner_points_2d[4:]

    # Resetting the object ancilla for points that are not reflected
    # This is the red arrows in figure 9 of the paper
    assert y_wall_points[0].invert_velocity_in_dimension == [False, True]  # lb, lb
    assert y_wall_points[1].invert_velocity_in_dimension == [False, False]  # lb, ub
    assert y_wall_points[2].invert_velocity_in_dimension == [True, True]  # ub, lb
    assert y_wall_points[3].invert_velocity_in_dimension == [True, False]  # ub, ub


def test_2d_near_corner_x_walls_bounds(simple_2d_block):
    assert len(simple_2d_block.near_corner_points_2d) == 8
    x_wall_points: List[ReflectionPoint] = simple_2d_block.near_corner_points_2d[:4]

    # The first 4 points belong to x-walls, which means
    # They exceed object bounds *only* in the x axis
    assert all(
        point_data.data[0].is_outside_obstacle_bounds for point_data in x_wall_points
    )
    assert all(
        not point_data.data[1].is_outside_obstacle_bounds
        for point_data in x_wall_points
    )


def test_2d_near_corner_y_walls_bounds(simple_2d_block):
    assert len(simple_2d_block.near_corner_points_2d) == 8
    y_wall_points: List[ReflectionPoint] = simple_2d_block.near_corner_points_2d[4:]

    # The first 4 points belong to y-walls, which means
    # They exceed object bounds *only* in the y axis
    assert all(
        not point_data.data[0].is_outside_obstacle_bounds
        for point_data in y_wall_points
    )
    assert all(
        point_data.data[1].is_outside_obstacle_bounds for point_data in y_wall_points
    )
