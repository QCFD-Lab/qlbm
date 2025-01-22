"""A ``SimulationConfig`` ties together algorithmic quantum components, circuit compilers, runners, and performance optimizations."""

from logging import Logger, getLogger
from typing import Any, List

from qiskit import QuantumCircuit as QiskitQC
from qiskit.quantum_info import Statevector
from qiskit_aer.backends.aerbackend import AerBackend
from qulacs import QuantumCircuit as QulacsQC
from qulacs import QuantumState

from qlbm.components.base import QuantumComponent
from qlbm.infra.compiler import CircuitCompiler
from qlbm.tools.exceptions import ExecutionException


class SimulationConfig:
    """
    A ``SimulationConfig`` ties together algorithmic quantum components, circuit compilers, runners, and performance optimizations.

    This is the most convenient access point for performing simulations with ``qlbm``.
    In total, the config contains 11 relevant class attributes that together
    allow users to customize their simulations in a declarative manner.
    For convenience, we split these attributes by the purpose they serve
    for the simulation workflow.

    Algorithmic attributes specify the complete, end-to-end, QLBM algorithm.
    This includes initial conditions, the time step circuit,
    an optional postprocessing step, and a final measurement procedure.

    .. list-table:: Algorithmic attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`initial_conditions`
          - The initial conditions of the simulations. For example, :class:`.CollisionlessInitialConditions` or :class:`.PointWiseSpaceTimeInitialConditions`.
        * - :attr:`algorithm`
          - The algorithm that performs the QLBM time step computation. For example, :class:`.CQLBM` or :class:`.SpaceTimeQLBM`.
        * - :attr:`postprocessing`
          - The quantum component concataned to the ``algorithm``. Usually :class:`.EmptyPrimitive`.
        * - :attr:`measurement`
          - The circuit that samples the quantum state. For example, :class:`.GridMeasurement` or :class:`.SpaceTimeGridVelocityMeasurement`.

    Compiler-related attributes govern how compilers convert algorithmic attributes to the appropriate format.
    All quantum circuits will be compiled using the same settings.

    .. list-table:: Compiler-related attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`target_platform`
          - The platform that the simulation will be carried out on. Either ``"QISKIT"`` or ``"QULACS"``.
        * - :attr:`compiler_platform`
          - The platform of the compiler to use. Either ``"QISKIT"`` or ``"TKET"``.
        * - :attr:`optimization_level`
          - The compiler optimization level.

    Runner-related attributes prescribe how the simulation should proceede.
    This includes the specific simulators that will execute the circuits,
    and performance optimization settings.

    .. list-table:: Runner-related attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`execution_backend`
          - The specific ``AerSimulator`` use (if using Qiskit) or ``None`` if using Qulacs.
        * - :attr:`sampling_backend`
          - The specific ``AerSimulator`` to use if ``statevector_sampling`` is enabled.
        * - :attr:`statevector_sampling`
          - Whether statevector sampling should be utilized.

    .. note::
        Example configuration: simulating :class:`.SpaceTimeQLBM` with Qiskit.
        First, we set up the config with the circuits we want to simulate,
        and the infrastructure we want to use.

        .. code-block:: python

            SimulationConfig(
                initial_conditions=SpaceTimeInitialConditions(
                    lattice, grid_data=[((1, 5), (True, True, True, True))]
                ),
                algorithm=SpaceTimeQLBM(lattice),
                postprocessing=EmptyPrimitive(lattice),
                measurement=SpaceTimeGridVelocityMeasurement(lattice),
                target_platform="QISKIT",
                compiler_platform="QISKIT",
                optimization_level=0,
                statevector_sampling=True,
                execution_backend=AerSimulator(method="statevector"),
                sampling_backend=AerSimulator(method="statevector"),
            )

        Once constructed, the ``cfg`` will figure out the appropriate
        compiler calls to convert the circuit to the appropriate format.
        All the user needs to do is call the :meth:`prepare_for_simulation()` method:

        .. code-block:: python

            cfg.prepare_for_simulation()

        The circuits are compiled in-place, which makes it
        easy to plug in the ``cfg`` object into a :class:`.QiskitRunner`:

        .. code-block:: python

            # Create a runner object to simulate the circuit
            runner = QiskitRunner(
                cfg,
                lattice,
            )

            # Simulate the circuits
            runner.run(
                10
                2**12,
                "output_dir",
                False
            )


    .. note::
        Example configuration: simulating :class:`.CQLBM` with Qulacs and Tket.

        .. code-block:: python

            cfg = SimulationConfig(
                initial_conditions=CollisionlessInitialConditions(lattice, logger),
                algorithm=CQLBM(lattice, logger),
                postprocessing=EmptyPrimitive(lattice, logger),
                measurement=GridMeasurement(lattice, logger),
                target_platform="QULACS",
                compiler_platform="TKET",
                optimization_level=0,
                statevector_sampling=statevector_sampling,
                execution_backend=None,
                sampling_backend=AerSimulator(method="statevector"),
                logger=logger,
            )

        Once constructed, the ``cfg`` will figure out the appropriate
        compiler calls to convert the circuit to the appropriate format.
        All the user needs to do is call the :meth:`prepare_for_simulation()` method:

        .. code-block:: python

            cfg.prepare_for_simulation()

        The circuits are compiled in-place, which makes it
        easy to plug in the ``cfg`` object into a :class:`.QulacsRunner`:

        .. code-block:: python

            # Create a runner object to simulate the circuit
            runner = QulacsRunner(
                cfg,
                lattice,
            )

            # Simulate the circuits
            runner.run(
                10
                2**12,
                "output_dir",
                True
            )

    """

    initial_conditions: (
        Statevector | QiskitQC | QuantumState | QulacsQC | QuantumComponent
    )
    algorithm: QiskitQC | QulacsQC | QuantumComponent
    postprocessing: QiskitQC | QulacsQC | QuantumComponent
    measurement: QiskitQC | QuantumComponent
    execution_backend: AerBackend | None
    sampling_backend: AerBackend
    logger: Logger

    QISKIT = "QISKIT"
    QULACS = "QULACS"

    initial_conditions_types = {
        QISKIT: [Statevector, QiskitQC, QuantumComponent],
        QULACS: [QuantumState, QulacsQC, QuantumComponent],
    }

    algorithm_types = {
        QISKIT: [QiskitQC, QuantumComponent],
        QULACS: [QulacsQC, QuantumComponent],
    }

    measurement_types = {
        QISKIT: [QiskitQC, QuantumComponent],
        QULACS: [QiskitQC, QuantumComponent],
    }

    execution_backend_types = {QISKIT: [AerBackend], QULACS: [type(None)]}

    sampling_backend_types = {
        QISKIT: [AerBackend],
        QULACS: [AerBackend],
    }

    def __init__(
        self,
        initial_conditions: Statevector
        | QiskitQC
        | QuantumState
        | QulacsQC
        | QuantumComponent,
        algorithm: QiskitQC | QulacsQC | QuantumComponent,
        postprocessing: QiskitQC | QulacsQC | QuantumComponent,
        measurement: QiskitQC | QuantumComponent,
        target_platform: str,
        compiler_platform: str,
        optimization_level: int,
        statevector_sampling: bool,
        execution_backend: AerBackend | None,
        sampling_backend: AerBackend | None,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        # Circuits
        self.initial_conditions = initial_conditions
        self.algorithm = algorithm
        self.postprocessing = postprocessing
        self.measurement = measurement

        # Simulation details
        self.target_platform = target_platform
        self.compiler_platform = compiler_platform
        self.optimization_level = optimization_level
        self.statevector_sampling = statevector_sampling

        # Backends
        self.execution_backend = execution_backend
        self.sampling_backend = sampling_backend
        self.logger = logger

    def validate(
        self,
    ):
        """
        Validates the configuration.

        This includes the following checks:

        #. The algorithmic attributes are of compatible types.
        #. The target platform is available.
        #. The execution backend (if enabled) is compatible with the target platform.
        #. The sampling backend (if enabled) is compatible with the target platform.

        This function simply checks that the provided attributes are
        suitable - it does not perform any conversions.
        """
        self.__is_appropriate_target_platform(self.target_platform)
        self.__is_compatible_type(
            self.initial_conditions,
            self.initial_conditions_types[self.target_platform],
            "Initial conditions",
            self.target_platform,
        )

        self.__is_compatible_type(
            self.algorithm,
            self.algorithm_types[self.target_platform],
            "Algorithm",
            self.target_platform,
        )

        self.__is_compatible_type(
            self.postprocessing,
            self.algorithm_types[self.target_platform],
            "Postprocessing",
            self.target_platform,
        )

        self.__is_compatible_type(
            self.measurement,
            self.measurement_types[self.target_platform],
            "Measurement",
            self.target_platform,
        )

        if self.statevector_sampling:
            self.__is_compatible_type(
                self.sampling_backend,
                self.sampling_backend_types[self.target_platform],
                "Sampling Backend",
                self.target_platform,
            )

        self.__is_compatible_type(
            self.execution_backend,
            self.execution_backend_types[self.target_platform],
            "Execution Backend",
            self.target_platform,
        )

    def __is_compatible_type(
        self,
        object_to_validate: Any,
        accepted_types: List[Any],
        object_name: str,
        platform: str,
    ) -> None:
        if not any(isinstance(object_to_validate, t) for t in accepted_types):
            raise ExecutionException(
                f"{object_name} object of type {type(object_to_validate)} is not in supported types {accepted_types} for platform {platform}",
            )

    def __is_appropriate_target_platform(self, platform: str) -> None:
        if platform not in [self.QISKIT, self.QULACS]:
            raise ExecutionException(
                f"Platform {platform} unsupported. Supported platforms are: {[self.QISKIT, self.QULACS]}"
            )

    def prepare_for_simulation(
        self,
    ) -> None:
        """Converts all algorithmic components to the target platform according to the specification, in place."""
        self.__perpare_circuits(
            self.get_execution_compiler(),
            self.get_sampling_compiler(),
        )

    def get_execution_compiler(self) -> CircuitCompiler:
        """
        Get the :class:`CircuitCompiler` that can converts the algorithmic attributes to the target ``sampling_backend`` simulator.

        Returns
        -------
        CircuitCompiler
            A compatible circuit compiler.
        """
        self.__is_appropriate_target_platform(self.target_platform)

        return CircuitCompiler(
            self.compiler_platform, self.target_platform, self.logger
        )

    def get_sampling_compiler(self) -> CircuitCompiler:
        """
        Get the :class:`CircuitCompiler` that can converts the algorithmic attributes to the target ``execution_backend`` simulator.

        Returns
        -------
        CircuitCompiler
            A compatible circuit compiler.
        """
        return (
            CircuitCompiler(self.compiler_platform, "QISKIT", self.logger)
            if self.statevector_sampling
            else self.get_execution_compiler()
        )

    def __perpare_circuits(
        self, execution_compiler: CircuitCompiler, sampling_compiler: CircuitCompiler
    ) -> None:
        if not isinstance(self.initial_conditions, Statevector):
            self.initial_conditions = execution_compiler.compile(
                self.initial_conditions, self.execution_backend, self.optimization_level
            )

        self.algorithm = execution_compiler.compile(
            self.algorithm, self.execution_backend, self.optimization_level
        )
        self.postprocessing = sampling_compiler.compile(
            self.postprocessing,
            self.sampling_backend
            if self.statevector_sampling
            else self.execution_backend,
            self.optimization_level,
        )
        self.measurement = sampling_compiler.compile(
            self.measurement,
            self.sampling_backend
            if self.statevector_sampling
            else self.execution_backend,
            self.optimization_level,
        )
