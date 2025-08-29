from typing import List, Tuple

import pytest
from qlbm.components.lqlga.streaming import LQLGAStreamingOperator


def verify_streaming_line_correctness(
    size: int, swaps: List[List[Tuple[int, int]]]
) -> bool:
    configuration = list(range(size))
    for layer in swaps:
        for i, j in layer:
            configuration[i], configuration[j] = (
                configuration[j],
                configuration[i],
            )

    return configuration == [size - 1] + list(range(size - 1))


@pytest.mark.parametrize("size", list(map(lambda x: 2**x, list(range(10)))))
def test_streaming_line_creation_power_of_2(lattice_d1q2_8, size):
    operator = LQLGAStreamingOperator(lattice_d1q2_8)
    swaps = operator.logarithmic_depth_streaming_line_swaps(size, False)

    assert verify_streaming_line_correctness(size, swaps)


@pytest.mark.parametrize("size", list(map(lambda x: 2**x + 1, list(range(1, 10)))))
def test_streaming_line_creation_off_power_of_2_positive(lattice_d1q2_8, size):
    operator = LQLGAStreamingOperator(lattice_d1q2_8)
    swaps = operator.logarithmic_depth_streaming_line_swaps(size, False)

    assert verify_streaming_line_correctness(size, swaps)


@pytest.mark.parametrize("size", list(map(lambda x: 2**x - 1, list(range(3, 10)))))
def test_streaming_line_creation__off_power_2_negative(lattice_d1q2_8, size):
    operator = LQLGAStreamingOperator(lattice_d1q2_8)
    swaps = operator.logarithmic_depth_streaming_line_swaps(size, False)

    assert verify_streaming_line_correctness(size, swaps)


@pytest.mark.parametrize("size", list(range(7, 16)))
def test_streaming_line_creation_intermediate(lattice_d1q2_8, size):
    operator = LQLGAStreamingOperator(lattice_d1q2_8)
    swaps = operator.logarithmic_depth_streaming_line_swaps(size, False)

    assert verify_streaming_line_correctness(size, swaps)
