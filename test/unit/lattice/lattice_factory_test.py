"""Tests for the Lattice factory methods create_result and create_reinitializer."""

import os
import shutil
import tempfile

import pytest

from qlbm.infra.compiler import CircuitCompiler
from qlbm.infra.reinitialize.base import Reinitializer
from qlbm.infra.reinitialize.identity_reinitializer import IdentityReinitializer
from qlbm.infra.reinitialize.spacetime_reinitializer import SpaceTimeReinitializer
from qlbm.infra.result.amplitude_result import AmplitudeResult
from qlbm.infra.result.base import QBMResult
from qlbm.infra.result.lqlga_result import LQLGAResult
from qlbm.infra.result.spacetime_result import SpaceTimeResult
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice
from qlbm.lattice.lattices.ms_lattice import MSLattice
from qlbm.lattice.lattices.oh_lattice import OHLattice
from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice


@pytest.fixture
def ms_lattice() -> MSLattice:
    return MSLattice(
        {
            "lattice": {"dim": {"x": 16, "y": 16}, "velocities": {"x": 4, "y": 4}},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [2, 6],
                    "y": [5, 10],
                    "boundary": "bounceback",
                },
            ],
        },
    )


@pytest.fixture
def ab_lattice() -> ABLattice:
    return ABLattice(
        {
            "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
            "geometry": [],
        },
    )


@pytest.fixture
def oh_lattice() -> OHLattice:
    return OHLattice(
        {
            "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "D2Q9"},
            "geometry": [],
        },
    )


@pytest.fixture
def spacetime_lattice() -> SpaceTimeLattice:
    return SpaceTimeLattice(
        1,
        {
            "lattice": {"dim": {"x": 16, "y": 16}, "velocities": "D2Q4"},
            "geometry": [
                {
                    "shape": "cuboid",
                    "x": [2, 6],
                    "y": [5, 10],
                    "boundary": "bounceback",
                },
            ],
        },
    )


@pytest.fixture
def lqlga_lattice() -> LQLGALattice:
    return LQLGALattice(
        {
            "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "D2Q4"},
        },
    )


@pytest.fixture
def compiler() -> CircuitCompiler:
    return CircuitCompiler("QISKIT", "QISKIT")


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ========================
# create_result tests
# ========================


class TestCreateResult:
    """Tests for the create_result factory method across all lattice types."""

    def test_ms_lattice_creates_amplitude_result(self, ms_lattice, temp_dir):
        result = ms_lattice.create_result(temp_dir, "step")
        assert isinstance(result, AmplitudeResult)
        assert isinstance(result, QBMResult)

    def test_ab_lattice_creates_amplitude_result(self, ab_lattice, temp_dir):
        result = ab_lattice.create_result(temp_dir, "step")
        assert isinstance(result, AmplitudeResult)
        assert isinstance(result, QBMResult)

    def test_oh_lattice_creates_amplitude_result(self, oh_lattice, temp_dir):
        result = oh_lattice.create_result(temp_dir, "step")
        assert isinstance(result, AmplitudeResult)
        assert isinstance(result, QBMResult)

    def test_spacetime_lattice_creates_spacetime_result(
        self, spacetime_lattice, temp_dir
    ):
        result = spacetime_lattice.create_result(temp_dir, "step")
        assert isinstance(result, SpaceTimeResult)
        assert isinstance(result, QBMResult)

    def test_lqlga_lattice_creates_lqlga_result(self, lqlga_lattice, temp_dir):
        result = lqlga_lattice.create_result(temp_dir, "step")
        assert isinstance(result, LQLGAResult)
        assert isinstance(result, QBMResult)

    def test_result_has_correct_directory(self, ms_lattice, temp_dir):
        result = ms_lattice.create_result(temp_dir, "output")
        assert result.directory == temp_dir

    def test_result_has_correct_file_name(self, ms_lattice, temp_dir):
        result = ms_lattice.create_result(temp_dir, "my_output")
        assert result.output_file_name == "my_output"

    def test_result_stores_lattice(self, ab_lattice, temp_dir):
        result = ab_lattice.create_result(temp_dir, "step")
        assert result.lattice is ab_lattice

    def test_result_creates_output_directory(self, ms_lattice, temp_dir):
        output_dir = os.path.join(temp_dir, "nested", "output")
        result = ms_lattice.create_result(output_dir, "step")
        assert os.path.isdir(output_dir)

    def test_result_writes_lattice_json(self, spacetime_lattice, temp_dir):
        spacetime_lattice.create_result(temp_dir, "step")
        lattice_json_path = os.path.join(temp_dir, "lattice.json")
        assert os.path.isfile(lattice_json_path)


class TestCreateReinitializer:
    """Tests for the create_reinitializer factory method across all lattice types."""

    def test_ms_lattice_creates_identity_reinitializer(self, ms_lattice, compiler):
        reinit = ms_lattice.create_reinitializer(compiler)
        assert isinstance(reinit, IdentityReinitializer)
        assert isinstance(reinit, Reinitializer)

    def test_ab_lattice_creates_identity_reinitializer(self, ab_lattice, compiler):
        reinit = ab_lattice.create_reinitializer(compiler)
        assert isinstance(reinit, IdentityReinitializer)
        assert isinstance(reinit, Reinitializer)

    def test_oh_lattice_creates_identity_reinitializer(self, oh_lattice, compiler):
        reinit = oh_lattice.create_reinitializer(compiler)
        assert isinstance(reinit, IdentityReinitializer)
        assert isinstance(reinit, Reinitializer)

    def test_spacetime_lattice_creates_spacetime_reinitializer(
        self, spacetime_lattice, compiler
    ):
        reinit = spacetime_lattice.create_reinitializer(compiler)
        assert isinstance(reinit, SpaceTimeReinitializer)
        assert isinstance(reinit, Reinitializer)

    def test_lqlga_lattice_creates_identity_reinitializer(
        self, lqlga_lattice, compiler
    ):
        reinit = lqlga_lattice.create_reinitializer(compiler)
        assert isinstance(reinit, IdentityReinitializer)
        assert isinstance(reinit, Reinitializer)

    def test_reinitializer_stores_lattice(self, ms_lattice, compiler):
        reinit = ms_lattice.create_reinitializer(compiler)
        assert reinit.lattice is ms_lattice

    def test_reinitializer_stores_compiler(self, ab_lattice, compiler):
        reinit = ab_lattice.create_reinitializer(compiler)
        assert reinit.compiler is compiler

    def test_spacetime_reinitializer_stores_lattice(self, spacetime_lattice, compiler):
        reinit = spacetime_lattice.create_reinitializer(compiler)
        assert reinit.lattice is spacetime_lattice

    def test_identity_reinitializer_requires_statevector(self, ms_lattice, compiler):
        reinit = ms_lattice.create_reinitializer(compiler)
        assert reinit.requires_statevector() is True

    def test_spacetime_reinitializer_requires_statevector(
        self, spacetime_lattice, compiler
    ):
        reinit = spacetime_lattice.create_reinitializer(compiler)
        assert reinit.requires_statevector() is False


class TestFactoryConsistency:
    """Tests ensuring the factory methods produce the same types as the old isinstance dispatch."""

    @pytest.mark.parametrize(
        "lattice_fixture,expected_result_type",
        [
            ("ms_lattice", AmplitudeResult),
            ("ab_lattice", AmplitudeResult),
            ("oh_lattice", AmplitudeResult),
            ("spacetime_lattice", SpaceTimeResult),
            ("lqlga_lattice", LQLGAResult),
        ],
    )
    def test_result_type_matches_lattice(
        self, lattice_fixture, expected_result_type, temp_dir, request
    ):
        lattice = request.getfixturevalue(lattice_fixture)
        result = lattice.create_result(temp_dir, "step")
        assert type(result) is expected_result_type

    @pytest.mark.parametrize(
        "lattice_fixture,expected_reinit_type",
        [
            ("ms_lattice", IdentityReinitializer),
            ("ab_lattice", IdentityReinitializer),
            ("oh_lattice", IdentityReinitializer),
            ("spacetime_lattice", SpaceTimeReinitializer),
            ("lqlga_lattice", IdentityReinitializer),
        ],
    )
    def test_reinitializer_type_matches_lattice(
        self, lattice_fixture, expected_reinit_type, compiler, request
    ):
        lattice = request.getfixturevalue(lattice_fixture)
        reinit = lattice.create_reinitializer(compiler)
        assert type(reinit) is expected_reinit_type
