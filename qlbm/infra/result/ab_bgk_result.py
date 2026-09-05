r""":class:`.ABBGKQLBM`-specific implementation of the :class:`.QBMResult`."""

import re
from os import listdir
from typing import Dict, Tuple

import numpy as np
import vtk
from qiskit.quantum_info import Statevector
from typing_extensions import override
from vtkmodules.util import numpy_support

from qlbm.lattice.lattices.ab_bgk_lattice import ABBGKLattice

from .base import QBMResult


class ABBGKResult(QBMResult):
    r"""
    :class:`.ABBGKQLBM`-specific implementation of the :class:`.QBMResult`.

    Processes counts sampled from :class:`.ABBGKMeasurement` primitives, which measure
    the grid, velocity, and marker registers jointly. Turning those counts into a flow
    field takes three steps:

    #. Discard the marker-zero samples, which belong to the auxiliary sector that makes
       the collision unitary and carry no physical meaning.
    #. Convert the remaining sample frequencies into populations through
       :math:`f_i = \alpha \lVert \rho \rVert_2 \sqrt{P_i}`.
    #. Take the moments of those populations to obtain :math:`\rho` and :math:`\mathbf{u}`.

    The visualized scalar is the velocity magnitude :math:`\lvert \mathbf{u} \rvert`,
    while the full fields of the most recent time step remain available in
    :attr:`density`, :attr:`velocity_x`, and :attr:`velocity_y`.

    .. note::
        Populations are recovered from a *square root* of a sampled frequency, so shot
        noise enters the flow field as :math:`\mathcal{O}(1 / \sqrt{f_i})` rather than
        linearly. Rare channels are therefore the noisiest, and small populations need
        disproportionately many shots. Statevector-based post-processing through
        :meth:`.ABBGKReinitializer.decode` avoids the issue entirely and is what the
        time-stepping loop itself uses.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`lattice`             The :class:`.ABBGKLattice` of the simulated system.
    :attr:`directory`           The directory to which the results outputs data to.
    :attr:`output_file_name`    The root name for files containing time step artifacts, by default "step".
    =========================== ======================================================================
    """

    num_steps: int
    """The time step to which this result corresponds."""

    directory: str
    """The output directory for the results."""

    output_file_name: str
    """The name of the file to output the artifacts to."""

    lattice: ABBGKLattice
    """The lattice the result corresponds to."""

    density: np.typing.NDArray[np.float64]
    r"""The density field :math:`\rho` of the most recently processed time step."""

    velocity_x: np.typing.NDArray[np.float64]
    r"""The :math:`x` velocity field of the most recently processed time step."""

    velocity_y: np.typing.NDArray[np.float64]
    r"""The :math:`y` velocity field of the most recently processed time step."""

    physical_sector_probability: float
    """The share of samples that landed in the physical sector of the most recent time step."""

    flow_fields: Dict[
        int,
        Tuple[
            np.typing.NDArray[np.float64],
            np.typing.NDArray[np.float64],
            np.typing.NDArray[np.float64],
        ],
    ]
    """The density and velocity fields of every processed time step, keyed by time step."""

    def __init__(
        self,
        lattice: ABBGKLattice,
        directory: str,
        output_file_name: str = "step",
    ) -> None:
        super().__init__(lattice, directory, output_file_name)
        self.lattice = lattice

        grid_shape = (
            lattice.num_gridpoints[0] + 1,
            lattice.num_gridpoints[1] + 1,
        )
        self.density = np.zeros(grid_shape)
        self.velocity_x = np.zeros(grid_shape)
        self.velocity_y = np.zeros(grid_shape)
        self.physical_sector_probability = 0.0
        self.flow_fields = {}

    @override
    def save_timestep_counts(
        self,
        counts: Dict[str, float],
        timestep: int,
        create_vis: bool = True,
        save_array: bool = False,
    ):
        self.__save_flow_field(
            self.counts_to_flow_field(counts, collided=timestep > 0),
            timestep,
            create_vis=create_vis,
            save_array=save_array,
        )

    @override
    def save_statevector(self, statevector: Statevector, step: int):
        r"""
        Saves the statevector and replaces the sampled flow field of the time step with its exact counterpart.

        Populations are recovered through a square root of a sampled frequency, which
        turns shot noise into an :math:`\mathcal{O}(1 / \sqrt{f_i})` error on the flow
        field. Whenever a runner is configured to keep statevector snapshots, the exact
        amplitudes are available and are used instead, so the artifacts of that time step
        describe the flow field the algorithm actually produced.

        Parameters
        ----------
        statevector : qiskit.quantum_info.Statevector
            The statevector at the end of the time step.
        step : int
            The time step the statevector corresponds to.
        """
        super().save_statevector(statevector, step)

        self.__save_flow_field(
            self.statevector_to_flow_field(statevector, collided=step > 0),
            step,
        )

    def statevector_to_flow_field(
        self,
        statevector: Statevector,
        collided: bool = True,
    ) -> Tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ]:
        """
        Converts exact amplitudes into the macroscopic flow field they encode.

        Parameters
        ----------
        statevector : qiskit.quantum_info.Statevector
            The statevector to decode.
        collided : bool, optional
            Whether the state has been through the collision, by default True. The
            un-evolved initial state has not, and is decoded through
            :meth:`.D2Q9AngleEncoding.flow_field_from_branch_amplitudes` instead.

        Returns
        -------
        Tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
            The density and the two velocity components, each of shape ``(nx, ny)``.
        """
        encoding = self.lattice.encoding
        amplitudes = np.asarray(statevector)

        if not collided:
            return encoding.flow_field_from_branch_amplitudes(
                amplitudes[self.lattice.collision_basis_indices()].real,
                self.lattice.density_norm,
            )

        probabilities = np.abs(amplitudes[self.lattice.physical_basis_indices()]) ** 2
        self.physical_sector_probability = float(np.sum(probabilities))

        populations = encoding.populations_from_probabilities(
            probabilities, self.lattice.density_norm
        )

        return encoding.macroscopic(populations, np.sum(populations, axis=-1) <= 0.0)

    def __save_flow_field(
        self,
        flow_field: Tuple[
            np.typing.NDArray[np.float64],
            np.typing.NDArray[np.float64],
            np.typing.NDArray[np.float64],
        ],
        timestep: int,
        create_vis: bool = True,
        save_array: bool = False,
    ) -> None:
        """
        Records a flow field and writes the artifacts of its time step.

        Parameters
        ----------
        flow_field : Tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
            The density and the two velocity components.
        timestep : int
            The time step the field corresponds to.
        create_vis : bool, optional
            Whether to create the visualization, by default True.
        save_array : bool, optional
            Whether to save the raw array to a CSV file, by default False.
        """
        self.density, self.velocity_x, self.velocity_y = flow_field
        self.flow_fields[timestep] = flow_field

        self.save_timestep_array(
            np.sqrt(self.velocity_x**2 + self.velocity_y**2),
            timestep,
            create_vis=create_vis,
            save_counts_array=save_array,
        )

    def counts_to_flow_field(
        self,
        counts: Dict[str, float],
        collided: bool = True,
    ) -> Tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
    ]:
        r"""
        Converts sampled counts into the macroscopic flow field they encode.

        Also updates :attr:`physical_sector_probability`, whose shortfall from one is the
        share of the sampling budget that the auxiliary sector absorbed.

        Parameters
        ----------
        counts : Dict[str, float]
            The counts sampled from an :class:`.ABBGKMeasurement`.
        collided : bool, optional
            Whether the sampled state has been through the collision, by default True.
            The un-evolved initial state has not, and holds branch--angle amplitudes
            instead of populations. Its velocity *signs* cannot be recovered from
            frequencies alone, though its speed can.

        Returns
        -------
        Tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]
            The density and the two velocity components, each of shape ``(nx, ny)``.
        """
        encoding = self.lattice.encoding
        num_x_bits = self.lattice.num_gridpoints[0].bit_length()
        num_y_bits = self.lattice.num_gridpoints[1].bit_length()
        num_grid_bits = num_x_bits + num_y_bits

        probabilities = np.zeros(
            (
                self.lattice.num_gridpoints[0] + 1,
                self.lattice.num_gridpoints[1] + 1,
                encoding.NUM_STATES,
            )
        )

        total_samples = sum(counts.values())

        for count, frequency in counts.items():
            # Qiskit prints the highest classical bit first, so reversing the string
            # recovers the order in which the qubits were measured.
            measured = count.replace(" ", "")[::-1]

            x = int(measured[:num_x_bits][::-1], 2)
            y = int(measured[num_x_bits:num_grid_bits][::-1], 2)
            collision_state = int(
                measured[num_grid_bits : num_grid_bits + encoding.NUM_COLLISION_QUBITS][
                    ::-1
                ],
                2,
            )

            probabilities[x, y, collision_state] += frequency

        if total_samples > 0:
            probabilities /= total_samples

        physical = probabilities[..., encoding.physical_state_indices()]
        self.physical_sector_probability = float(np.sum(physical))

        if not collided:
            return encoding.flow_field_from_branch_amplitudes(
                np.sqrt(probabilities), self.lattice.density_norm
            )

        populations = encoding.populations_from_probabilities(
            physical, self.lattice.density_norm
        )

        # Gridpoints that received no sample at all would otherwise divide by zero.
        return encoding.macroscopic(populations, np.sum(populations, axis=-1) <= 0.0)

    @override
    def visualize_all_numpy_data(self):
        # Filter the algorithm output files
        data_file_pattern = re.compile("[a-zA-Z0-9]+_[0-9]+.csv")

        for data_file_name in filter(data_file_pattern.match, listdir(self.directory)):
            data = np.genfromtxt(
                f"{self.directory}/{data_file_name}",
                dtype=None,
                delimiter=",",
                autostrip=True,
            )

            vtk_data = numpy_support.numpy_to_vtk(
                num_array=data, deep=False, array_type=vtk.VTK_FLOAT
            )
            image = vtk.vtkImageData()
            image.GetPointData().SetScalars(vtk_data)
            image.SetDimensions(
                self.lattice.num_gridpoints[0] + 1,
                self.lattice.num_gridpoints[1] + 1,
                1,
            )

            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(
                f"{self.paraview_dir}/{data_file_name.split('.')[0]}.vti"
            )
            writer.SetInputData(image)
            writer.Write()
