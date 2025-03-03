from itertools import product
from typing import List, Set, Tuple

import pytest

from qlbm.lattice.geometry.shapes.circle import Circle
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice
from qlbm.tools.utils import flatten, get_qubits_to_invert


def non_intersecting_lists(lists: List[List[Tuple[int, int]]]) -> bool:
    seen: Set[Tuple[int, int]] = set()

    for sublist in lists:
        sublist_set = set(sublist)
        if seen & sublist_set:
            return False
        seen.update(sublist_set)

    return True


def is_appropriate_velocity_profile(circle, segment, pw_reflection_data):
    dimensional_bounds = [segment[0][dim] > circle.center[dim] for dim in [0, 1]]

    appropriate_reflection_directions = [
        [[0, 2], [2, 0]],
        [[1, 3], [3, 1]],
    ]

    # appropriate_reflection_directions = [
    #     list(reversed(dir))
    #     if pw_reflection_data.distance_from_boundary_point[dim] < 0
    #     else dir
    #     for dim, dir in enumerate(appropriate_reflection_directions)
    # ]

    nvp = pw_reflection_data.neighbor_velocity_pairs[0]
    nvp_velocities = [nvp[0][1], nvp[1][1]]
    dim_of_reflection = pw_reflection_data.velocity_indices_to_reflect[0] % 2
    return (
        nvp_velocities in appropriate_reflection_directions[dim_of_reflection]
        and dim_of_reflection in nvp_velocities
    )


def test_circle_contains_gridpoint_points_inside(small_circle):
    points_inside = []

    for y in range(6, 11):
        points_inside.append((12, y))
        points_inside.append((4, y))

    for y in range(5, 12):
        points_inside.append((11, y))
        points_inside.append((5, y))

    for x in range(6, 11):
        for y in range(4, 13):
            points_inside.append((x, y))

    assert all(small_circle.contains_gridpoint(p) for p in points_inside)

    all_points = list(product(list(range(16)), list(range(16))))
    points_outside = [p for p in all_points if (p not in points_inside)]

    assert all(not small_circle.contains_gridpoint(p) for p in points_outside)


def test_circle_perimeter(small_circle):
    perimeter_points = []
    for i in range(6, 11):
        perimeter_points.append((12, i))
        perimeter_points.append((4, i))
        perimeter_points.append((i, 4))
        perimeter_points.append((i, 12))

    for x in [5, 11]:
        for y in [5, 11]:
            perimeter_points.append((x, y))

    assert len(small_circle.perimeter_points) == 24
    assert all(p in small_circle.perimeter_points for p in perimeter_points)


@pytest.mark.parametrize(
    "circle_name",
    [
        "small_circle",
        "circle_2",
        "circle_3",
        "circle_4",
        "circle_5",
        "circle_6",
        "circle_7",
    ],
)
def test_circle_overlapping_segments(circle_name, request):
    circle = request.getfixturevalue(circle_name)
    axis_segments, diag_segments, individual_points = circle.split_perimeter_points(
        circle.perimeter_points
    )

    expanded_axis_segments = Circle.expand_axis_segments(axis_segments)
    expanded_diag_segments = Circle.expand_diagonal_segments(diag_segments)

    # All perimeter points are included in the segment encoding
    assert (
        p in expanded_axis_segments + expanded_diag_segments + individual_points
        for p in circle.perimeter_points
    )

    # There are no repeating points in any segment
    assert sorted(expanded_axis_segments) == sorted(list(set(expanded_axis_segments)))
    assert sorted(expanded_diag_segments) == sorted(list(set(expanded_diag_segments)))
    assert sorted(individual_points) == sorted(list(set(individual_points)))

    # There are no points included twice in any segment
    assert len(
        set(expanded_axis_segments + expanded_diag_segments + individual_points)
    ) == len(expanded_axis_segments) + len(expanded_diag_segments) + len(
        individual_points
    )


@pytest.mark.parametrize(
    "circle_name,num_steps",
    list(
        product(
            [
                "small_circle",
                "circle_2",
                "circle_3",
                "circle_4",
                "circle_6",
            ],
            list(range(1, 11)),
        )
    ),
)
def test_circle_number_of_reflection_points(circle_name, num_steps, request):
    circle = request.getfixturevalue(circle_name)
    dummy_lattice = request.getfixturevalue("dummy_lattice")

    axis_segments, diag_segments, individual_points = circle.split_perimeter_points(
        circle.perimeter_points
    )
    reflection_data = circle.get_spacetime_reflection_data_d2q4(
        dummy_lattice.properties, num_steps
    )

    assert len(reflection_data) == 2 * num_steps * num_steps * (
        len(Circle.expand_axis_segments(axis_segments))
        + 2 * len(Circle.expand_diagonal_segments(diag_segments))
        + 2 * len(individual_points)
    )


@pytest.mark.parametrize(
    "circle_name,num_steps",
    list(
        product(
            [
                "small_circle",
                "circle_2",
                "circle_3",
                "circle_4",
                "circle_6",
            ],
            list(range(1, 5)),
        )
    ),
)
def test_circle_number_of_diagonal_reflection_points(
    circle_name,
    num_steps,
    request,
):
    circle = request.getfixturevalue(circle_name)
    dummy_lattice = request.getfixturevalue("dummy_lattice")

    axis_segments, diag_segments, _ = circle.split_perimeter_points(
        circle.perimeter_points
    )
    reflection_data = circle.get_spacetime_reflection_data_d2q4(
        dummy_lattice.properties, num_steps
    )

    start = 2 * num_steps * num_steps * len(Circle.expand_axis_segments(axis_segments))

    for c, diag_segment in enumerate(diag_segments):
        increment = (
            2
            * num_steps
            * num_steps
            * len(Circle.expand_diagonal_segments(diag_segments))
        )

        points_of_segment = reflection_data[
            start + c * increment : start + (c + 1) * increment
        ]

        assert all(
            is_appropriate_velocity_profile(circle, diag_segment, p)
            for p in points_of_segment
        )
