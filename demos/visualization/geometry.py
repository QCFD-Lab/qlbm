from qlbm.infra import CollisionlessResult
from qlbm.lattice import CollisionlessLattice
from qlbm.tools.utils import create_directory_and_parents

if __name__ == "__main__":
    lattice_2d = CollisionlessLattice("demos/lattices/2d_32x32_7_obstacle.json")
    lattice_3d = CollisionlessLattice("demos/lattices/3d_16x16x16_1_obstacle.json")

    root_directory_2d = "./qlbm-output/visualization-components-2d"
    root_directory_3d = "./qlbm-output/visualization-components-3d"
    create_directory_and_parents(root_directory_2d)
    create_directory_and_parents(root_directory_3d)

    # Will output seven 2D stl files under `qlbm-output/visualization-components-2d/paraview`
    CollisionlessResult(lattice_2d, root_directory_2d).visualize_geometry()

    # Will output one 3D stl files under `qlbm-output/visualization-components-3d/paraview`
    CollisionlessResult(lattice_3d, root_directory_3d).visualize_geometry()
