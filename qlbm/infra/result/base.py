from abc import ABC, abstractmethod
from os.path import isdir
from typing import Dict

import numpy as np
import vtk
from vtkmodules.util import numpy_support

from qlbm.lattice import Lattice
from qlbm.tools.utils import create_directory_and_parents, flatten


class QBMResult(ABC):
    num_steps: int
    directory: str
    output_file_name: str

    def __init__(
        self,
        lattice: Lattice,
        directory: str,
        output_file_name: str = "step",
    ) -> None:
        self.directory = directory
        self.output_file_name = output_file_name
        self.paraview_dir = f"{self.directory}/paraview"

        if not isdir(directory):
            create_directory_and_parents(directory)

        if not isdir(self.paraview_dir):
            create_directory_and_parents(self.paraview_dir)

        with open(f"{directory}/lattice.json", "w+") as f:
            f.write(lattice.to_json())

        self.lattice = lattice

    def visualize_geometry(self):
        for c, block in enumerate(flatten(self.lattice.blocks.values())):
            block.stl_mesh().save(f"{self.paraview_dir}/cube_{c}.stl")

    def save_timestep_array(
        self,
        numpy_res: np.ndarray,
        timestep: int,
        create_vis: bool = True,
        save_counts_array: bool = False,
    ):
        if save_counts_array:
            numpy_res.tofile(
                f"{self.directory}/{self.output_file_name}_{timestep}.csv",
                sep=",",
            )

        if create_vis:
            self.create_visualization(data=numpy_res, timestep=timestep)

    def create_visualization(self, data: np.ndarray | None, timestep: int):
        if not isdir(self.paraview_dir):
            create_directory_and_parents(self.paraview_dir)

        if data is None:
            self.visualize_all_numpy_data()
            return

        vtk_data = numpy_support.numpy_to_vtk(
            num_array=data.flatten(), deep=False, array_type=vtk.VTK_FLOAT
        )
        img = vtk.vtkImageData()
        img.GetPointData().SetScalars(vtk_data)
        img.SetDimensions(
            self.lattice.num_gridpoints[0] + 1,
            self.lattice.num_gridpoints[1] + 1
            if self.lattice.num_dimensions > 1
            else 1,
            self.lattice.num_gridpoints[2] + 1
            if self.lattice.num_dimensions > 2
            else 1,
        )

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(f"{self.paraview_dir}/step_{timestep}.vti")
        writer.SetInputData(img)
        writer.Write()

    @abstractmethod
    def save_timestep_counts(
        self,
        counts: Dict[str, float],
        timestep: int,
        create_vis: bool = True,
        save_array: bool = False,
    ):
        pass

    @abstractmethod
    def visualize_all_numpy_data(self):
        pass
