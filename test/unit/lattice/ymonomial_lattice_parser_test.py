"""Unit tests for ymonomial geometry parsing in lattice base parser."""

import json

import pytest

from qlbm.lattice import MSLattice
from qlbm.lattice.geometry.shapes.block import Block
from qlbm.lattice.geometry.shapes.ymonomial import YMonomial
from qlbm.tools.exceptions import LatticeException
from qlbm.tools.utils import ComparatorMode


def _lattice_with_geometry(geometry):
    """Build a minimal valid 2D MS lattice with a configurable geometry list."""
    return MSLattice(
        {
            "lattice": {
                "dim": {"x": 16, "y": 16},
                "velocities": {"x": 4, "y": 4},
            },
            "geometry": geometry,
        }
    )


def test_parse_ymonomial_specular_geometry():
    """Parses a specular ymonomial obstacle and stores the expected mode."""
    lattice = _lattice_with_geometry(
        [
            {
                "shape": "ymonomial",
                "exponent": 2,
                "comparator": "<=",
                "boundary": "specular",
            }
        ]
    )

    assert len(lattice.shapes["specular"]) == 1
    shape = lattice.shapes["specular"][0]
    assert isinstance(shape, YMonomial)
    assert shape.exponent == 2
    assert shape.comparator_mode == ComparatorMode.LE


def test_parse_ymonomial_bounceback_geometry():
    """Parses a bounceback ymonomial obstacle and stores the expected mode."""
    lattice = _lattice_with_geometry(
        [
            {
                "shape": "ymonomial",
                "exponent": 3,
                "comparator": ">",
                "boundary": "bounceback",
            }
        ]
    )

    assert len(lattice.shapes["bounceback"]) == 1
    shape = lattice.shapes["bounceback"][0]
    assert isinstance(shape, YMonomial)
    assert shape.comparator_mode == ComparatorMode.GT


@pytest.mark.parametrize(
    "comparator_symbol, expected_mode",
    [
        ("<", ComparatorMode.LT),
        ("<=", ComparatorMode.LE),
        (">", ComparatorMode.GT),
        (">=", ComparatorMode.GE),
    ],
)
def test_parse_ymonomial_comparator_modes(comparator_symbol, expected_mode):
    """Parses each supported comparator symbol into the expected mode."""
    lattice = _lattice_with_geometry(
        [
            {
                "shape": "ymonomial",
                "exponent": 1,
                "comparator": comparator_symbol,
                "boundary": "specular",
            }
        ]
    )

    ymonomial_shape = lattice.shapes["specular"][0]
    assert isinstance(ymonomial_shape, YMonomial)
    assert ymonomial_shape.comparator_mode == expected_mode


def test_parse_ymonomial_and_cuboid_geometry_together():
    """Keeps cuboid parsing intact while adding ymonomial parsing."""
    lattice = _lattice_with_geometry(
        [
            {"shape": "cuboid", "x": [4, 6], "y": [3, 5], "boundary": "specular"},
            {
                "shape": "ymonomial",
                "exponent": 2,
                "comparator": "<=",
                "boundary": "bounceback",
            },
        ]
    )

    assert len(lattice.shapes["specular"]) == 1
    assert len(lattice.shapes["bounceback"]) == 1
    assert isinstance(lattice.shapes["specular"][0], Block)
    assert isinstance(lattice.shapes["bounceback"][0], YMonomial)


def test_ymonomial_to_json_roundtrip_shape_and_parameters():
    """Serializes parsed ymonomial geometry back with expected parameters."""
    lattice = _lattice_with_geometry(
        [
            {
                "shape": "ymonomial",
                "exponent": 4,
                "comparator": ">=",
                "boundary": "specular",
            }
        ]
    )

    geometry = json.loads(lattice.to_json())["geometry"]

    assert len(geometry) == 1
    assert geometry[0]["shape"] == "ymonomial"
    assert geometry[0]["exponent"] == 4
    assert geometry[0]["comparator"] == ">="
    assert geometry[0]["boundary"] == "specular"


def test_lattice_exception_ymonomial_missing_exponent():
    """Raises an informative exception when ymonomial exponent is missing."""
    with pytest.raises(LatticeException) as excinfo:
        _lattice_with_geometry(
            [
                {
                    "shape": "ymonomial",
                    "comparator": "<=",
                    "boundary": "specular",
                }
            ]
        )

    assert (
        "Obstacle 1: ymonomial obstacle does not specify an exponent."
        == str(excinfo.value)
    )


def test_lattice_exception_ymonomial_non_integer_exponent():
    """Raises an informative exception when ymonomial exponent is non-integer."""
    with pytest.raises(LatticeException) as excinfo:
        _lattice_with_geometry(
            [
                {
                    "shape": "ymonomial",
                    "exponent": "abc",
                    "comparator": "<=",
                    "boundary": "specular",
                }
            ]
        )

    assert (
        "Obstacle 1: ymonomial exponent abc is not an integer."
        == str(excinfo.value)
    )


def test_lattice_exception_ymonomial_negative_exponent():
    """Raises an informative exception when ymonomial exponent is negative."""
    with pytest.raises(LatticeException) as excinfo:
        _lattice_with_geometry(
            [
                {
                    "shape": "ymonomial",
                    "exponent": -1,
                    "comparator": "<=",
                    "boundary": "specular",
                }
            ]
        )

    assert (
        "Obstacle 1: ymonomial exponent -1 must be non-negative."
        == str(excinfo.value)
    )


def test_lattice_exception_ymonomial_missing_comparator():
    """Raises an informative exception when ymonomial comparator is missing."""
    with pytest.raises(LatticeException) as excinfo:
        _lattice_with_geometry(
            [
                {
                    "shape": "ymonomial",
                    "exponent": 2,
                    "boundary": "specular",
                }
            ]
        )

    assert (
        "Obstacle 1: ymonomial obstacle does not specify a comparator."
        == str(excinfo.value)
    )


def test_lattice_exception_ymonomial_non_string_comparator():
    """Raises an informative exception when ymonomial comparator is not a string."""
    with pytest.raises(LatticeException) as excinfo:
        _lattice_with_geometry(
            [
                {
                    "shape": "ymonomial",
                    "exponent": 2,
                    "comparator": 123,
                    "boundary": "specular",
                }
            ]
        )

    assert "Obstacle 1: ymonomial comparator must be a string." == str(excinfo.value)


def test_lattice_exception_ymonomial_invalid_comparator_symbol():
    """Propagates comparator symbol validation for unsupported operators."""
    with pytest.raises(LatticeException) as excinfo:
        _lattice_with_geometry(
            [
                {
                    "shape": "ymonomial",
                    "exponent": 2,
                    "comparator": "!=",
                    "boundary": "specular",
                }
            ]
        )

    assert (
        "Unsupported comparator mode '!='. Expected one of: <, <=, >, >=."
        == str(excinfo.value)
    )


def test_lattice_exception_ymonomial_in_1d_lattice():
    """Raises an informative exception when ymonomial is used outside 2D."""
    with pytest.raises(LatticeException) as excinfo:
        MSLattice(
            {
                "lattice": {
                    "dim": {"x": 16},
                    "velocities": {"x": 4},
                },
                "geometry": [
                    {
                        "shape": "ymonomial",
                        "exponent": 2,
                        "comparator": "<=",
                        "boundary": "specular",
                    }
                ],
            }
        )

    assert (
        "Obstacle 1: ymonomial is only supported for 2-dimensional lattices."
        == str(excinfo.value)
    )