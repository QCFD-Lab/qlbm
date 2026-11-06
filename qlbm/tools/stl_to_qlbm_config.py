"""Create a 2D staircase approximation of an STL object for QLBM simulations."""

import warnings
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from trimesh.ray.ray_triangle import RayMeshIntersector

# Define tuple aliases to make the coordinate dimensions clear.
Vector2: TypeAlias = tuple[float, float]
Vector3: TypeAlias = tuple[float, float, float]
GridShape: TypeAlias = tuple[int, int]
FilePath: TypeAlias = str | PathLike[str]


def finite_vector(value, n, name):
    """Convert a vector to floats and reject invalid dimensions or values."""
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {n} finite numbers.")
    return a


def classify(mesh, points):
    """
    Determine which points are inside a closed mesh using ray intersections.

    An odd number of intersections indicates an inside point, while an even number
    indicates an outside point. Cast rays in three directions and use the
    majority vote. Points detected within 1e-8 of the surface count as solid.

    Args:
        mesh: Closed triangular mesh.
        points: NumPy array of shape (N, 3), containing point coordinates
            in the same coordinate system as the mesh.

    Returns
    -------
        A boolean array of length N, where True means solid, and the number
        of non-surface points for which the three ray tests disagreed.
    """
    # Prepare the mesh for ray-intersection calculations.
    ray = RayMeshIntersector(mesh)

    # Choose three tilted directions to reduce alignment with mesh edges and faces.
    directions = np.array(
        [[1.0, 0.173, 0.317], [0.271, 1.0, 0.193], [0.137, 0.293, 1.0]]
    )
    # Make each direction vector have length 1.
    directions /= np.linalg.norm(directions, axis=1)[:, None]

    # Track how many directions classify each point as inside.
    votes = np.zeros(len(points), dtype=np.uint8)

    # Track points lying directly on the surface.
    surface = np.zeros(len(points), dtype=bool)
    for direction in directions:
        # Cast one ray from every point
        locations, ids, _ = ray.intersects_location(
            points, np.broadcast_to(direction, points.shape), multiple_hits=True
        )

        # Count intersections per point and add one inside vote if the count is odd.
        votes += (np.bincount(ids, minlength=len(points)) % 2).astype(np.uint8)

        # If an intersection is extremely close to its starting point,
        # that point is treated as on the surface and marked solid.
        if len(ids):
            surface[ids[np.linalg.norm(locations - points[ids], axis=1) <= 1e-8]] = True

    # Count how many non-surface points had mixed votes
    disagreement = int(np.count_nonzero((votes > 0) & (votes < 3) & ~surface))

    # A point is solid if at least two rays vote inside, or it lies on the surface.
    return (votes >= 2) | surface, disagreement


class STLGeometry:
    """Generate a 2D staircase cross-section from a 3D STL."""

    def __init__(
        self,
        stl_file: FilePath,
        *,
        grid_shape: GridShape,
        domain_origin: Vector2 | None = None,
        object_center: Vector3 | None = None,
        object_size: float | None = None,
        scale: float = 1.0,
        rotation_deg: Vector3 = (0.0, 0.0, 0.0),
        plane: Literal["xy", "xz", "yz"] = "xy",
        slice_coordinate: float | None = None,
        chunk_size: int = 2000,
        max_points: int = 2000000,
    ) -> None:
        """Creates stair case approximations of curved geometries from STL files.

        Args:
            stl_file : FilePath
                 Path to the STL file to load.

            grid_shape : GridShape
                (Nu, Nv): number of cells along the two selected plane axes.
                Each grid dimension must be a power of two and contain at least
                two cells.

            domain_origin : Vector2 | None
                Corner where the grid begins. Uses coordinates within the selected plane.
                Defaults to (0, 0).

            object_center : Vector3 | None
                Position (x, y, z) where the object's centre should be placed.
                The centre is the midpoint of the original STL's bounding box.
                If None, places it at the grid centre on the slicing plane.

            object_size : float | None
                Target size of the object's longest dimension in lattice units.
                The other dimensions are scaled by the same factor to preserve the
                object's proportions. For example, object_size=20 resizes a
                100 x 50 x 25 object to 20 x 10 x 5 lattice units.
                If None, the object's size is determined by scale instead.
                When provided, scale must remain 1.0.

            scale : float
                Factor that multiplies all original STL dimensions when object_size
                is None. For example, 2.0 doubles the dimensions and 0.5 halves them.
                Must be positive. Defaults to 1.0, keeping the original dimensions.
                Must remain 1.0 when object_size is provided.

            rotation_deg : Vector3
                Rotation angles in degrees around the fixed x, y, and z axes,
                applied in that order. Rotates the object around its original
                bounding-box centre before placing it at object_center.
                Defaults to (0, 0, 0), keeping the STL's original orientation.

            plane : Literal["xy", "xz", "yz"]
                Coordinate plane for a 2D section.

            slice_coordinate : float | None
                Position of the 2D cutting plane: z for plane="xy", y for "xz",
                or x for "yz". For example, plane="xy" with slice_coordinate=10
                takes a cross-section at z=10. If None, uses the corresponding
                coordinate of object_center, or 0 if object_center is also None.

            chunk_size : int
                Number of grid nodes checked at a time for whether they are inside
                the object. Must be positive. Smaller chunk_size uses less temporary
                memory without changing the geometry. Defaults to 2000.

            max_points : int
                Safety limit on the total number of cells to classify. Increase it
                for larger grids if the additional processing time and memory use are
                acceptable. Unlike chunk_size, this limits the total work, not the
                number of cells processed at once. Defaults to 2,000,000.

        """
        dx = 1.0  # spacing in lattice units

        # Check path to STL file
        if not isinstance(stl_file, (str, PathLike)):
            raise TypeError("stl_file must be a path to an STL file.")

        # Checks fo the grid_shape argument
        raw = np.asarray(grid_shape)
        if (
            raw.ndim != 1  # nested arrays
            or len(raw) != 2  # the grid must have exactly two dimensions
            or not np.isfinite(raw).all()  # infinity or undefined
            or not np.equal(raw, np.floor(raw)).all()  # fractional grid points
            or (raw < 2).any()  # dimensions with fewer than 2 grid points
            or any(int(n) & (int(n) - 1) for n in raw)  # not a power of two
        ):
            raise ValueError(
                "Specify two grid dimensions, for example (64, 32). "
                "Each number must be a power of two: 2, 4, 8, 16, 32, etc. "
                f"Got: {grid_shape!r}."
            )

        shape = tuple(int(n) for n in raw)

        # Check the domain_origin argument
        origin = (
            np.zeros(2)
            if domain_origin is None
            else finite_vector(domain_origin, 2, "domain_origin")
        )

        # Check the object_center argument
        target = (
            None
            if object_center is None
            else finite_vector(object_center, 3, "object_center")
        )

        # Check the scale argument
        if scale <= 0:
            raise ValueError("scale must be positive.")

        # Check the rotation argument
        rotation = finite_vector(rotation_deg, 3, "rotation_deg")

        # Check the number of grid points.
        count = int(np.prod(shape, dtype=object))
        if count > max_points:
            raise ValueError(f"{count:,} grid points exceed max_points={max_points:,}.")

        # Check the chunk_size argument
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")

        # Check plane argument.
        if plane not in ("xy", "xz", "yz"):
            raise ValueError("plane must be 'xy', 'xz', or 'yz'.")
        # Find which axis varies and which stays fixed. For 2D grid keep
        # will be [0, 1] or [0, 2] or [1, 2] representing 'xy', 'xz', or 'yz'.
        keep = ["xyz".index(a) for a in plane]
        fixed = next(i for i in range(3) if i not in keep)

        # Chooses the slice position, if slice_coordinate is not given, it uses the
        # corresponding coordinate of object_center, or zero if neither was supplied.
        coord = (
            (target[fixed] if target is not None else 0.0)
            if slice_coordinate is None
            else float(slice_coordinate)
        )

        # Construct the plane's position and directions.
        po = np.zeros(3)
        po[fixed] = coord  # The plane's local origin in 3D

        # Select the 3D direction vectors for the two axes of the slicing plane.
        # For "xz", keep=[0, 2] selects [1, 0, 0] (x) and [0, 0, 1] (z).
        basis = np.eye(3)[keep]

        # Use the plane's axis names as labels, e.g. "xz" becomes ["x", "z"].
        labels = list(plane)
        if target is None:
            # No object_center supplied, place the object at the 2D grid centre.
            # Map that centre into 3D using the plane's directions and origin.
            target = po + (origin + np.array(shape) * dx / 2) @ basis

        # Load the STL with trimesh's basic mesh processing enabled.
        mesh = trimesh.load_mesh(stl_file, process=True)

        # Require a triangle mesh with at least one face and finite vertex coordinates.
        if (
            not isinstance(mesh, trimesh.Trimesh)
            or len(mesh.faces) == 0
            or not np.isfinite(mesh.vertices).all()
        ):
            raise ValueError("Expected a nonempty finite triangular mesh.")

        # Require a closed surface so we can classify points as inside or outside.
        if not mesh.is_watertight:
            raise ValueError("Mesh is not watertight! Repair or cap it.")

        # Find the longest dimension of the original object's bounding box.
        size = float(mesh.extents.max())
        if size <= 0:
            raise ValueError("Mesh has zero extent.")

        if object_size is not None:
            # Choose either a target size or a scaling factor, not both.
            if scale != 1.0:
                raise ValueError("Use object_size or scale, not both.")

            # The target size must be finite and greater than zero.
            if not np.isfinite(object_size) or object_size <= 0:
                raise ValueError("object_size must be positive.")

            # Calculate the factor needed to reach the requested size.
            # For example, resizing from 100 to 20 requires scale = 20 / 100 = 0.2.
            scale = float(object_size) / size

        # Find the centre of the original STL's bounding box.
        source_center = mesh.bounds.mean(axis=0)

        # Convert degrees to radians and build the rotation matrix.
        angles = np.deg2rad(rotation)
        rotation_matrix = trimesh.transformations.euler_matrix(
            angles[0], angles[1], angles[2], "sxyz"
        )[:3, :3]

        # Build a 4×4 matrix combining scaling, rotation, and translation.
        transform = np.eye(4)
        transform[:3, :3] = scale * rotation_matrix

        # Choose the translation so the original bounding-box centre
        # ends up at target after scaling and rotation.
        transform[:3, 3] = target - transform[:3, :3] @ source_center

        # Apply these changes to the mesh's vertices.
        mesh.apply_transform(transform)

        # For 2D, express the vertices along the slicing plane's two axes.
        local_vertices = (mesh.vertices - po) @ basis.T

        # Find the domain's upper corner, opposite origin.
        high = origin + np.array(shape) * dx

        # Allow a small floating-point rounding errors at the domain boundaries.
        tol = 1e-10 * max(float(np.max(high - origin)), 1.0)

        # Check whether any part of the object's bounds extends outside the grid.
        # For 2D, check the whole object's projection, not just the cross-section.
        clipped = bool(
            np.any(local_vertices.min(axis=0) < origin - tol)
            or np.any(local_vertices.max(axis=0) > high + tol)
        )
        if clipped:
            raise ValueError(
                "The object extends outside the grid. "
                "Resize or reposition it to fit inside. "
                "For 2D, the whole object's projection must fit."
            )

        # Calculate cell-centre coordinates along each grid axis.
        axes = tuple(origin[d] + (np.arange(n) + 0.5) * dx for d, n in enumerate(shape))

        # Make a temporary mesh centred near zero with longest dimension 1.
        center = mesh.bounds.mean(axis=0)
        normalization = float(mesh.extents.max())
        query = mesh.copy()
        query.vertices = (mesh.vertices - center) / normalization

        # Initially mark every grid point as fluid (False).
        solid = np.zeros(count, dtype=bool)

        # Count points where the different ray tests disagree about inside/outside.
        disagreements = 0

        # Process grid points in batches to limit temporary memory use.
        for start in range(0, count, chunk_size):
            # Get the flat indices of the points in this batch.
            ids = np.arange(start, min(start + chunk_size, count))

            # Convert flat indices into the two plane-local grid indices (iu, iv).
            ij = np.unravel_index(ids, shape)

            # Look up the cell-centre coordinates for those grid indices.
            local = np.column_stack([axes[d][ij[d]] for d in range(2)])

            # For 2D, map the points on the slicing plane into 3D world coordinates.
            world = po + local @ basis

            # Points outside the mesh's bounding box cannot be inside the object.
            valid = np.all(
                (world >= mesh.bounds[0]) & (world <= mesh.bounds[1]), axis=1
            )

            if valid.any():
                # Normalize the remaining points exactly like the temporary mesh,
                # then test whether each point is inside the object.
                occupied, n = classify(query, (world[valid] - center) / normalization)

                # Store the solid/fluid results at their original grid indices.
                solid[ids[valid]] = occupied
                disagreements += n

        # Warn if no grid cell centres were classified as inside the object.
        # The slice might miss the object, or the object might be too small/thin
        # for the grid to capture.
        if not solid.any():
            warnings.warn(
                "No solid nodes: plane may miss object, or resolution may be too coarse.",
            )

        # Warn if ray tests from different directions disagreed about whether
        # some points were inside or outside the object.
        if disagreements:
            warnings.warn(
                f"{disagreements} points had inconsistent ray votes, inspect geometry.",
            )
        # Keep only the data needed for config export and plotting.
        self.solid_mask = solid.reshape(shape)
        self.mesh = mesh
        self.origin = origin
        self.dx = dx
        self.plane_origin = po
        self.plane_basis = basis
        self.axis_labels = labels

    def to_qlbm_config(self) -> dict:
        """Convert the 2D staircase geometry into a QLBM configuration.

        The solid mask contains True for solid cells and False for fluid cells.
        To represent this mask using QLBM's supported geometry shapes, group
        consecutive solid cells in each row into horizontal stretches. Merge
        stretches with identical x bounds in consecutive rows into rectangles.

        The rectangles cover exactly the solid cells. They may touch, but they
        do not overlap.

        Use the configuration with ABQLBM(..., use_agnostic_bcs=True), which
        handles the touching rectangles as parts of the same solid region.
        The obstacle must leave at least one layer of fluid cells along every
        outer edge of the grid.

        NOTE: Only 2D D2Q9 configurations are supported because the current ABQLBM
        zone-agnostic reflection implementation does not support 3D.

        Returns
        -------
            dict: Grid dimensions and the D2Q9 velocity model under "lattice",
                and rectangles with bounce-back boundaries under "geometry".
                Pass this dictionary to ABLattice(config).
        """
        # Use a shorter name for the same mask.
        s = self.solid_mask

        # Reject solid cells on any of the four outer grid edges.
        if s[0].any() or s[-1].any() or s[:, 0].any() or s[:, -1].any():
            raise ValueError("Keep at least one fluid-node layer at every domain edge.")

        # blocks holds finished rectangles. active maps each (start_x, end_x)
        # stretch to the first y row where its current rectangle appeared.
        # For example: active = {(1, 3): 1} means x=1..3, starting at y=1.
        blocks: list[dict] = []
        active: dict[tuple[int, int], int] = {}

        # Scan rows along y. The mask is indexed [x, y], so s[:, iy] is a row.
        # The extra iteration has no solid stretches and closes any rectangles
        # still active after the last real row.
        for iy in range(s.shape[1] + 1):
            runs: set[tuple[int, int]] = set()
            if iy < s.shape[1]:
                # Convert False/True to 0/1, pad both ends with zeros, and find
                # differences between neighbors. +1 starts a solid stretch,
                # -1 marks the first position after it, so subtract 1 for its end.
                d = np.diff(np.pad(s[:, iy].astype(np.int8), (1, 1)))
                # Example: [0, 1, 1, 1, 0, 0] gives the inclusive run (1, 3).
                runs = {
                    (int(start), int(end))
                    for start, end in zip(
                        np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
                    )
                }

            # If a stretch disappears or changes width, finish its rectangle
            # at the previous row (iy - 1). list(active) lets us remove entries
            # from the dictionary while iterating over this snapshot of its keys.
            for run in list(active):
                if run not in runs:
                    blocks.append(
                        {
                            "shape": "cuboid",
                            "x": [int(run[0]), int(run[1])],
                            # pop retrieves the starting row and removes the run.
                            "y": [active.pop(run), iy - 1],
                            "boundary": "bounceback",
                        }
                    )

            # setdefault starts a new rectangle at iy only if the run is absent.
            # An identical stretch in the next row keeps its original start y.
            for run in sorted(runs):
                active.setdefault(run, iy)

        # Package the rectangles and lattice settings for ABLattice.
        # For example, a mask shape of (64, 32) gives {"x": 64, "y": 32}.
        return {
            "lattice": {
                "dim": dict(zip(("x", "y"), self.solid_mask.shape)),
                "velocities": "D2Q9",
            },
            "geometry": blocks,
        }

    def export_paraview_stl(self, filename: FilePath) -> None:
        """Save the complete staircase as one STL overlay for ParaView."""
        blocks = self.to_qlbm_config()["geometry"]
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        for block in blocks:
            # Inclusive index bounds become cell edges half a unit farther out.
            # This gives single-row rectangles area and joins adjacent pieces.
            x0, x1 = block["x"][0] - 0.5, block["x"][1] + 0.5
            y0, y1 = block["y"][0] - 0.5, block["y"][1] + 0.5
            offset = len(vertices)
            vertices.extend([(x0, y0, 1), (x1, y0, 1), (x1, y1, 1), (x0, y1, 1)])
            faces.extend(
                [(offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)]
            )

        overlay = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=float).reshape(-1, 3),
            faces=np.asarray(faces, dtype=int).reshape(-1, 3),
            process=False,
        )
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(trimesh.exchange.stl.export_stl(overlay))

    def plot(self) -> Figure:
        """Build a preview of the grid and obstacle."""
        solid_mask = self.solid_mask
        shape = solid_mask.shape
        origin, dx = self.origin, self.dx
        mesh = self.mesh
        po, basis = self.plane_origin, self.plane_basis
        labels = self.axis_labels

        edges = [origin[d] + np.arange(n + 1) * dx for d, n in enumerate(shape)]
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.pcolormesh(
            *edges,
            solid_mask.T,
            shading="flat",
            edgecolors="#b9c7d0",
            linewidth=0.25,
            vmin=0,
            vmax=1,
            cmap=ListedColormap(["#e3eff7", "#e78a39"]),
        )
        # Exact triangle/plane section for comparison with the staircase.
        section = trimesh.intersections.mesh_plane(
            mesh,
            plane_normal=np.cross(*basis),
            plane_origin=po,
        )

        legend_handles: list[Artist] = [
            Patch(facecolor="#e3eff7", edgecolor="#b9c7d0", label="Fluid cells"),
            Patch(facecolor="#e78a39", edgecolor="#b9c7d0", label="Solid cells"),
        ]
        if len(section):
            uv = (section - po) @ basis.T
            outline = LineCollection(
                uv, colors="#20374b", linewidths=0.9, label="STL outline"
            )
            ax.add_collection(outline)
            legend_handles.append(outline)
        ax.legend(handles=legend_handles, loc="upper right")
        ax.set(
            xlim=(edges[0][0], edges[0][-1]),
            ylim=(edges[1][0], edges[1][-1]),
            xlabel=labels[0],
            ylabel=labels[1],
            aspect="equal",
        )
        return fig
