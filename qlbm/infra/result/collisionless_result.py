""":class:`.CQLBM`-specific implementation of the :class:`.QBMResult`."""

import re
from os import listdir
from typing import Dict

import numpy as np
import vtk
from typing_extensions import override
from vtkmodules.util import numpy_support

from qlbm.lattice import CollisionlessLattice
from qlbm.lattice.lattices.abe_lattice import ABLattice

from .base import QBMResult


class CollisionlessResult(QBMResult):
    """
    :class:`.CQLBM`-specific implementation of the :class:`.QBMResult`.

    Processes counts sampled from :class:`.GridMeasurement` primitives.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`lattice`             The :class:`.CollisionlessLattice` of the simulated system.
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

    lattice: CollisionlessLattice | ABLattice
    """The lattice the result corresponds to."""

    def __init__(
        self,
        lattice: CollisionlessLattice | ABLattice,
        directory: str,
        output_file_name: str = "step",
    ) -> None:
        super().__init__(lattice, directory, output_file_name)

    @override
    def save_timestep_counts(
        self,
        counts: Dict[str, float],
        timestep: int,
        create_vis: bool = True,
        save_array: bool = False,
    ):
        dimension_bit_counts = (
            self.lattice.num_gridpoints[0].bit_length(),
            self.lattice.num_gridpoints[0].bit_length()
            + self.lattice.num_gridpoints[1].bit_length()
            if self.lattice.num_dims > 1
            else 0,
            self.lattice.num_gridpoints[0].bit_length()
            + self.lattice.num_gridpoints[1].bit_length()
            + self.lattice.num_gridpoints[2].bit_length()
            if self.lattice.num_dims > 2
            else 0,
        )

        if self.lattice.num_dims == 1:
            # The second dimension is a dirty rendering trick for VTK and Paraview
            count_history = np.zeros((self.lattice.num_gridpoints[0] + 1, 2))
            for count in counts:
                x = int(
                    count[self.lattice.num_velocity_qubits :],
                    2,
                )
                # Another dirty rendering trick for VTK and Paraview
                count_history[x][0] += counts[count] * (
                    1 + (int(count[: self.lattice.num_velocity_qubits] == "00"))
                )
                count_history[x][1] += counts[count] * (
                    1 + (int(count[: self.lattice.num_velocity_qubits] == "00"))
                )

        elif self.lattice.num_dims == 2:
            count_history = np.zeros(
                (self.lattice.num_gridpoints[0] + 1, self.lattice.num_gridpoints[1] + 1)
            )

            for count in counts:
                x = int(count[: dimension_bit_counts[0]], 2)
                y = int(
                    count[dimension_bit_counts[0] : dimension_bit_counts[1]],
                    2,
                )
                count_history[x][y] = counts[count]

        elif self.lattice.num_dims == 3:
            count_history = np.zeros(  # type: ignore
                (
                    self.lattice.num_gridpoints[0] + 1,
                    self.lattice.num_gridpoints[1] + 1,
                    self.lattice.num_gridpoints[2] + 1,
                )
            )

            for count in counts:
                x = int(count[: dimension_bit_counts[0]], 2)
                y = int(
                    count[dimension_bit_counts[0] : dimension_bit_counts[1]],
                    2,
                )
                z = int(
                    count[dimension_bit_counts[1] : dimension_bit_counts[2]],
                    2,
                )
                count_history[x][y][z] = counts[count]

        self.save_timestep_array(
            count_history if self.lattice.num_dims > 1 else np.transpose(count_history),
            timestep,
            create_vis=create_vis,
            save_counts_array=save_array,
        )

    @override
    def visualize_all_numpy_data(self):
        # Filter the algorithm output files
        r = re.compile("[a-zA-Z0-9]+_[0-9]+.csv")
        for data_file_name in filter(r.match, listdir(self.directory)):
            data = np.genfromtxt(
                f"{self.directory}/{data_file_name}",
                dtype=None,
                delimiter=",",
                autostrip=True,
            )
            vtk_data = numpy_support.numpy_to_vtk(
                num_array=data, deep=True, array_type=vtk.VTK_FLOAT
            )
            img = vtk.vtkImageData()
            img.SetDimensions(
                self.lattice.num_gridpoints[0] + 1,
                self.lattice.num_gridpoints[1] + 1 if self.lattice.num_dims > 1 else 1,
                self.lattice.num_gridpoints[2] + 1 if self.lattice.num_dims > 2 else 1,
            )
            img.GetPointData().SetScalars(vtk_data)

            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(
                f"{self.paraview_dir}/{data_file_name.split('.')[0]}.vti"
            )
            writer.SetInputData(img)
            writer.Write()
