from itertools import product

import pytest
from qiskit import QuantumCircuit as QiskitQC
from qiskit_aer import AerSimulator

from qlbm.components.ms import (
    CQLBM,
    MSInitialConditions,
    GridMeasurement,
)
from qlbm.components.common import EmptyPrimitive
from qlbm.infra.runner.simulation_config import SimulationConfig
from qlbm.lattice import MSLattice
from qlbm.tools.exceptions import ExecutionException


@pytest.fixture
def symmetric_2d_no_osbtacle_circuits():
    lattice = MSLattice("test/resources/symmetric_2d_no_obstacles.json")

    return {
        "initial_conditions": MSInitialConditions(lattice),
        "algorithm": CQLBM(lattice),
        "postprocessing": EmptyPrimitive(lattice),
        "measurement": GridMeasurement(lattice),
    }


@pytest.fixture
def symmetric_2d_one_osbtacle_circuits():
    lattice = MSLattice("test/resources/symmetric_2d_1_obstacle.json")

    return {
        "initial_conditions": MSInitialConditions(lattice),
        "algorithm": CQLBM(lattice),
        "postprocessing": EmptyPrimitive(lattice),
        "measurement": GridMeasurement(lattice),
    }


@pytest.mark.parametrize(
    "circuits,execution_backend_method,sampling_backend_method,statevector_sampling",
    list(
        product(
            ["symmetric_2d_no_osbtacle_circuits", "symmetric_2d_one_osbtacle_circuits"],
            ["statevector", "density_matrix", "unitary"],
            ["density_matrix", "unitary"],
            [True, False],
        )
    ),
)
def test_qiskit_simulation_configuration_validation(
    circuits,
    execution_backend_method,
    sampling_backend_method,
    statevector_sampling,
    request,
):
    circuits_dict = request.getfixturevalue(circuits)
    execution_backend = AerSimulator(method=execution_backend_method)
    sampling_backend = AerSimulator(method=sampling_backend_method)

    cfg = SimulationConfig(
        initial_conditions=circuits_dict["initial_conditions"],
        algorithm=circuits_dict["algorithm"],
        postprocessing=circuits_dict["postprocessing"],
        measurement=circuits_dict["measurement"],
        target_platform="QISKIT",
        compiler_platform="TKET",
        optimization_level=0,
        statevector_sampling=statevector_sampling,
        execution_backend=execution_backend,
        sampling_backend=sampling_backend,
    )

    cfg.validate()
    cfg.target_platform = "QULACS"

    with pytest.raises(ExecutionException) as excinfo:
        cfg.validate()

    if statevector_sampling:
        assert (
            "Execution Backend object of type <class 'qiskit_aer.backends.aer_simulator.AerSimulator'> is not in supported types [<class 'NoneType'>] for platform QULACS"
            == str(excinfo.value)
        )


@pytest.mark.parametrize(
    "circuits,sampling_backend_method,statevector_sampling",
    list(
        product(
            ["symmetric_2d_no_osbtacle_circuits", "symmetric_2d_one_osbtacle_circuits"],
            ["density_matrix", "unitary"],
            [True, False],
        )
    ),
)
def test_qulacs_simulation_configuration_validation(
    circuits,
    sampling_backend_method,
    statevector_sampling,
    request,
):
    circuits_dict = request.getfixturevalue(circuits)
    sampling_backend = AerSimulator(method=sampling_backend_method)

    cfg = SimulationConfig(
        initial_conditions=circuits_dict["initial_conditions"],
        algorithm=circuits_dict["algorithm"],
        postprocessing=circuits_dict["postprocessing"],
        measurement=circuits_dict["measurement"],
        target_platform="QULACS",
        compiler_platform="TKET",
        optimization_level=0,
        statevector_sampling=statevector_sampling,
        execution_backend=None,
        sampling_backend=sampling_backend,
    )

    cfg.validate()

    cfg.target_platform = "QISKIT"

    with pytest.raises(ExecutionException) as excinfo:
        cfg.validate()

    assert (
        "Execution Backend object of type <class 'NoneType'> is not in supported types"
        in str(excinfo.value)
    )


@pytest.mark.parametrize(
    "circuits,execution_backend_method,sampling_backend_method,compiler_platform,statevector_sampling",
    list(
        product(
            ["symmetric_2d_no_osbtacle_circuits", "symmetric_2d_one_osbtacle_circuits"],
            ["statevector", "matrix_product_state"],
            ["statevector", "matrix_product_state"],
            ["QISKIT", "TKET"],
            [True, False],
        )
    ),
)
def test_qiskit_simulation_configuration_pereparation(
    circuits,
    execution_backend_method,
    sampling_backend_method,
    compiler_platform,
    statevector_sampling,
    request,
):
    circuits_dict = request.getfixturevalue(circuits)
    execution_backend = AerSimulator(method=execution_backend_method)
    sampling_backend = AerSimulator(method=sampling_backend_method)

    cfg = SimulationConfig(
        initial_conditions=circuits_dict["initial_conditions"],
        algorithm=circuits_dict["algorithm"],
        postprocessing=circuits_dict["postprocessing"],
        measurement=circuits_dict["measurement"],
        target_platform="QISKIT",
        compiler_platform=compiler_platform,
        optimization_level=0,
        statevector_sampling=statevector_sampling,
        execution_backend=execution_backend,
        sampling_backend=sampling_backend,
    )

    cfg.validate()
    cfg.prepare_for_simulation()

    assert isinstance(cfg.initial_conditions, QiskitQC)
    assert isinstance(cfg.algorithm, QiskitQC)
    assert isinstance(cfg.postprocessing, QiskitQC)
    assert isinstance(cfg.measurement, QiskitQC)
