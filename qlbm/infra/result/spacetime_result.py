import re
from os import listdir
from typing import Dict

import numpy as np
import vtk
from vtkmodules.util import numpy_support

from qlbm.lattice import SpaceTimeLattice

from .base import QBMResult


class SpaceTimeResult(QBMResult):
    num_steps: int
    directory: str
    output_file_name: str
    lattice: SpaceTimeLattice

    def __init__(
        self,
        lattice: SpaceTimeLattice,
        directory: str,
        output_file_name: str = "step",
    ) -> None:
        super().__init__(lattice, directory, output_file_name)

    def save_timestep_counts(
        self,
        counts: Dict[str, float],
        timestep: int,
        create_vis: bool = True,
        save_array: bool = False,
    ):
        dimension_bit_counts = (
            self.lattice.num_gridpoints[0].bit_length(),
            self.lattice.num_gridpoints[1].bit_length()
            if self.lattice.num_dimensions > 1
            else 0,
            self.lattice.num_gridpoints[2].bit_length()
            if self.lattice.num_dimensions > 2
            else 0,
        )

        if self.lattice.num_dimensions == 2:
            count_history = np.zeros(
                (self.lattice.num_gridpoints[0] + 1, self.lattice.num_gridpoints[1] + 1)
            )
            for count in counts:
                count_inverse = count[::-1]
                x = int(
                    count_inverse[: dimension_bit_counts[0]][::-1],
                    2,
                )
                y = int(
                    count_inverse[
                        dimension_bit_counts[0] : dimension_bit_counts[0]
                        + dimension_bit_counts[1]
                    ][::-1],
                    2,
                )
                num_populations = int(
                    count_inverse[
                        dimension_bit_counts[0] + dimension_bit_counts[1] :
                    ].count("1")  # The number of 1s is the number of populations
                )
                count_history[x][y] = counts[count] * num_populations
        self.save_timestep_array(
            np.transpose(count_history),
            timestep,
            create_vis=create_vis,
            save_counts_array=save_array,
        )

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
                self.lattice.num_gridpoints[1] + 1
                if self.lattice.num_dimensions > 1
                else 1,
                self.lattice.num_gridpoints[2] + 1
                if self.lattice.num_dimensions > 2
                else 1,
            )
            img.GetPointData().SetScalars(vtk_data)

            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(
                f"{self.paraview_dir}/{data_file_name.split('.')[0]}.vti"
            )
            writer.SetInputData(img)
            writer.Write()
