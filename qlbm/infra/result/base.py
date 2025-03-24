"""Base class for all algorithm-specific results."""

from abc import ABC, abstractmethod
from os.path import isdir
from typing import Dict

import numpy as np
import vtk
from vtkmodules.util import numpy_support

from qlbm.lattice import Lattice
from qlbm.tools.utils import create_directory_and_parents, flatten


class QBMResult(ABC):
    """Base class for all algorithm-specific results.

    A ``Result`` object parses the counts extracted from the quantum state
    at the end of the simulation of some number of time steps.
    This information is then either translated into a visual representation
    of the encoded flow field or parsed into a format that is suitable for reinitialization.
    Results can additionally create visual representations of
    lattice geometry and save data to disk in compressed formats.

    =========================== ======================================================================
    Attribute                   Summary
    =========================== ======================================================================
    :attr:`lattice`             The :class:`.Lattice` of the simulated system.
    :attr:`directory`           The directory to which the results outputs data to.
    :attr:`paraview_dir`        The subdirectory under ``directory`` which stores the Paraview files.
    :attr:`output_file_name`    The root name for files containing time step artifacts, by default "step".
    =========================== ======================================================================
    """

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
        """
        Creates ``stl`` files for each block in the lattice.

        Output files are formatted as ``output_dir/paraview_dir/cube_<x>.stl``.
        The output is created through the :class:`.Block`'s :meth:`.Block.stl_mesh` method.
        """
        for c, shape in enumerate(flatten(self.lattice.blocks.values())):
            shape.stl_mesh().save(f"{self.paraview_dir}/{shape.name()}_{c}.stl")

    def save_timestep_array(
        self,
        numpy_res: np.ndarray,
        timestep: int,
        create_vis: bool = True,
        save_counts_array: bool = False,
    ):
        """
        Saves the time step array to a file.

        Parameters
        ----------
        numpy_res : np.ndarray
            The result in array format.
        timestep : int
            The time step to which the result corresponds.
        create_vis : bool, optional
            Whether to create the visualization, by default True.
        save_counts_array : bool, optional
            Whether to save the raw counts object to a CSV file, by default False.
        """
        if save_counts_array:
            numpy_res.tofile(
                f"{self.directory}/{self.output_file_name}_{timestep}.csv",
                sep=",",
            )

        if create_vis:
            self.create_visualization(data=numpy_res, timestep=timestep)

    def create_visualization(self, data: np.ndarray | None, timestep: int):
        """
        Creates a ``vtk`` visual representation of the data.

        Parameters
        ----------
        data : np.ndarray | None
            The ``np.ndarray`` representation of the population density at each grid location.
        timestep : int
            The time step to which the data corresponds.
        """
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
            self.lattice.num_gridpoints[1] + 1 if self.lattice.num_dims > 1 else 2,
            self.lattice.num_gridpoints[2] + 1 if self.lattice.num_dims > 2 else 1,
        )

        formatted_timestep = "{:03d}".format(timestep)
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(f"{self.paraview_dir}/step_{formatted_timestep}.vti")
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
        """
        Saves the time step counts to a file.

        Parameters
        ----------
        counts: Dict[str, float]
            The result in Qiskit ``Counts`` format.
        timestep : int
            The time step to which the result corresponds.k
        create_vis : bool, optional
            Whether to create the visualization, by default True.
        save_array : bool, optional
            Whether to save the raw counts object to a CSV file, by default False.
        """
        pass

    @abstractmethod
    def visualize_all_numpy_data(self):
        """Converts all numpy data saved to disk to ``vti`` files."""
        pass
