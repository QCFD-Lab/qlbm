""":class:`.LQLGA`-specific implementation of the :class:`.QBMResult`."""

import re
from os import listdir
from typing import Dict

import numpy as np
import vtk
from typing_extensions import override
from vtkmodules.util import numpy_support

from qlbm.lattice.lattices.lqlga_lattice import LQLGALattice

from .base import QBMResult


class LQLGAResult(QBMResult):
    """
    :class:`.LQLGA`-specific implementation of the :class:`.QBMResult`.

    Processes counts sampled from :class:`.LQLGAGridVelocityMeasurement` primitives.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`lattice`             The :class:`.LQLGA` of the simulated system.
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

    lattice: LQLGALattice
    """The lattice the result corresponds to."""

    def __init__(
        self,
        lattice: LQLGALattice,
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
        if self.lattice.num_dims == 1:
            # The second dimension is a dirty rendering trick for VTK and Paraview
            count_history = np.zeros((self.lattice.num_gridpoints[0] + 1, 2))
            for count in counts:
                count_inverse = count[::-1]
                num_vel = self.lattice.num_velocities_per_point
                for gp in range(
                    self.lattice.num_base_qubits
                    // self.lattice.num_velocities_per_point
                ):
                    num_populations = int(
                        count_inverse[gp * num_vel : (gp + 1) * num_vel][::-1].count(
                            "1"  # The number of 1s is the number of populations
                        )
                    )
                    # Another dirty rendering trick for VTK and Paraview
                    count_history[gp][0] = count_history[gp][1] = (
                        counts[count] * num_populations
                    )
        self.save_timestep_array(
            np.transpose(count_history),
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
                num_array=data.flatten, deep=True, array_type=vtk.VTK_FLOAT
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
