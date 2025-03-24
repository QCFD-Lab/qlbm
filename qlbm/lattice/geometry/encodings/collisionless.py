"""Geometrical data encodings specific to the :class:`CQLBM` algorithm."""

from typing import Callable, List, Tuple

from qlbm.tools.utils import flatten


class DimensionalReflectionData:
    r"""
    Contains one-dimensional information about the position of a grid point relevant to the obstacle.

    Used for edge cases relating to either inside or outside corner points.

    .. list-table:: Class attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`qubits_to_invert`
          - The ``List[int]`` of qubit indices that should be inverted in order to convert the state of grid qubits encoding this dimension to :math:`\ket{1}^{\otimes n_{g_d}}`. See the example below.
        * - :attr:`bound_type`
          - The ``bool`` indicating the type of bound this point belongs to. ``False`` indicates a lower bound, and ``True`` indicates an upper bound.
        * - :attr:`is_outside_obstacle_bounds`
          - The ``bool`` indicating the whether the point belongs to the solid domain. ``False`` that the point is inside the solid domain, and ``True`` indicates the outside.
        * - :attr:`dim`
          - The ``int`` indicating which dimension this object refers to.
        * - :attr:`gridpoint_encoded`
          - The ``int`` indicating which grid point this object encodes. Used for debugging purposes.
        * - :attr:`name`
          - A string assigned to each dimensional data object in the :class:`Block` constructor. Used for debugging purposes.

    .. note::
       Consider for example encoding the grid point at location 2
       (encoded as :math:`\ket{010}`) on the :math:`x`-axis on an :math:`8\\times 8` 2D grid.

       The ``DimensionalReflectionData`` object encoding this information
       would have a ``qubits_to_invert`` value of ``[0, 2]``.
       This means that the :math:`0^{th}` and :math:`2^{nd}` qubits
       would have to be inverted to produce the :math:`\ket{111}` state.
       This information is passed on to the reflection operators,
       which place the :math:`X` gates at the appropriate positions in the register,
       and can then use the :math:`g_x` register to control reflection.

       If we wanted to encode point :math:`3` (:math:`\ket{011}`) on :math:`y`-axis on the same grid,
       this would result in ``qubits_to_invert = [5]``, since the most significant qubit (index 2 of the :math:`y`-axis)
       is encoded last in the register, and there are 3 qubits encoding the :math:`x`-axis "in front" of it.
    """

    def __init__(
        self,
        qubits_to_invert: List[int],
        bound_type: bool,
        is_outside_obstacle_bounds: bool,
        gridpoint_encoded: int,
        dim: int,
        name: str,
    ) -> None:
        self.qubits_to_invert = qubits_to_invert
        self.bound_type = bound_type
        self.is_outside_obstacle_bounds = is_outside_obstacle_bounds
        self.gridpoint_encoded = gridpoint_encoded
        self.dim = dim
        self.name = name
        # XOR between lower and outside gives the same truth table that decides whether to invert the velocities
        self.invert_velocity = self.bound_type ^ self.is_outside_obstacle_bounds


class ReflectionResetEdge:
    r"""
    Encodes the information required to perform reflection on an edge in 3D.

    An edge is encoded as 2 fixed points as :class:`DimensionalReflectionData` objects, and one spanning dimension.
    This classes processes the information encoded in the reflection data
    object into boolean valued attributes that determine
    whether the directional velocity qubits should be inverted to perform reflection.

    .. list-table:: Class attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`walls_joining`
          - The ``List[DimensionalReflectionData]`` containing the point data for the two fixed dimensions, stored in ascending order (:math:`x < y < z`).
        * - :attr:`dims_of_edge`
          - The ``Tuple[int, int]`` indicating the fixed dimensions.
        * - :attr:`dim_disconnected`
          - The ``int`` indicating the dimension the edge spans.
        * - :attr:`bounds_disconnected_dim`
          - The ``Tuple[int, int]`` that specifies the bounds of the edge in the dimension that it spans.
        * - :attr:`reflected_velocities`
          - The ``List[int]`` that indicates to which dimensions the velocities that this edge affects belong to.
        * - :attr:`invert_velocity_in_dimension`
          - The ``List[bool]`` indicating whether the directional velocity qubits should be inverted, per dimensions.
        * - :attr:`is_corner_edge`
          - The ``bool`` indicating whether the edge is directly at the corner of the object.
    """

    def __init__(
        self,
        walls_joining: List[DimensionalReflectionData],
        dims_of_edge: Tuple[int, int],
        bounds_disconnected_dim: Tuple[int, int],
        dimension_outside: bool | None,
    ):
        self.walls_joining = walls_joining
        self.dims_of_edge = dims_of_edge
        self.dim_disconnected = [d for d in range(3) if d not in dims_of_edge][0]
        self.bounds_disconnected_dim = bounds_disconnected_dim
        self.dimension_outside = dimension_outside
        self.is_corner_edge = dimension_outside is None
        self.reflected_velocities: List[int] = (
            list(dims_of_edge)
            if self.is_corner_edge
            else [
                dims_of_edge[bool(dimension_outside)]  # type : ignore
            ]  # The velocity that the wall reflects is the one of the inside dimension
        )
        self.invert_velocity_in_dimension = (
            self.__get_corner_inversions(
                (walls_joining[0].bound_type, walls_joining[1].bound_type)
            )
            if self.is_corner_edge
            else self.__get_near_corner_inversions(
                (walls_joining[0].bound_type, walls_joining[1].bound_type),
                dimension_outside,  # type: ignore
            )
        )

    def __get_corner_inversions(
        self, dim_bounds: Tuple[bool, bool]
    ) -> Tuple[bool, bool]:
        return (not dim_bounds[0], not dim_bounds[1])

    def __get_near_corner_inversions(
        self, dim_bounds: Tuple[bool, bool], orthogonal: bool
    ) -> Tuple[bool, bool]:
        predicate: bool = dim_bounds[0] ^ dim_bounds[1]
        xor: bool = dim_bounds[0] ^ orthogonal

        if predicate:
            return (not xor, not xor)

        return (not xor, xor)


class ReflectionPoint:
    r"""
    Encodes the information required to perform reflection on a single point.

    A point is encoded as 2 or 3 fixed :class:`DimensionalReflectionData` objects, one per dimension.
    This classes processes the information encoded in the reflection data
    objects into boolean valued attributes that determine
    whether the directional velocity qubits should be inverted to perform reflection.

    .. list-table:: Class attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`data`
          - The ``List[DimensionalReflectionData]`` containing the point data for each dimension.
        * - :attr:`num_dims`
          - The ``int`` number of dimensions of this point (also of the corresponding lattice).
        * - :attr:`dims_inside`
          - The ``List[int]`` that specifies which of the ``data`` entries are `inside` obstacle bounds in their respective dimension.
        * - :attr:`dims_outside`
          - The ``List[int]`` that specifies which of the ``data`` entries are `outside` obstacle bounds in their respective dimension.
        * - :attr:`qubits_to_invert`
          -  The ``List[int]`` of qubit indices that should be inverted in order to convert the state of grid qubits to :math:`\ket{1}^{\otimes n_{g_d}}`.
        * - :attr:`inversion_function`
          - The ``Callable[[List[DimensionalReflectionData]], List[bool]]`` function that converts the input data into a list of booleans that determine whether the directional velocity qubits should be inverted, per dimensions.
        * - :attr:`invert_velocity_in_dimension`
          - The ``List[bool]`` obtained by calling the ``inversion_function`` on the ``data``, indicating whether the directional velocity qubits should be inverted, per dimensions.
        * - :attr:`is_near_corner_point`
          - The ``bool`` indicating whether the point is a near-corner point (used in 2D reflection).
    """

    def __init__(
        self,
        data: List[DimensionalReflectionData],
        dims_inside: List[int],
        dims_outside: List[int],
        inversion_function: Callable[[List[DimensionalReflectionData]], List[bool]],
    ):
        self.num_dims: int = len(data)
        self.qubits_to_invert: List[int] = flatten(
            [case.qubits_to_invert for case in data]
        )
        self.dims_inside = dims_inside
        self.dims_outside = dims_outside
        self.data = data
        # A point is near a corner if it is outside the obstacle in at least one bound, but not all
        self.is_near_corner_point = (len(dims_inside) != 0) and (len(dims_outside) != 0)
        self.invert_velocity_in_dimension: List[bool] = inversion_function(data)


class ReflectionWall:
    r"""
    Encodes the information required to perform reflection on a wall.

    Each wall is encoded as fixed over one dimensions and spanning one or two `alignment` dimensions.
    This in turn models which qubits are used for the comparator operations of the reflection operators.
    The information required for the alignment dimensions only consists of bounds,
    while the fixed dimension uses its :class:`DimensionalReflectionData` representation.

    .. list-table:: Class attributes
        :widths: 25 50
        :header-rows: 1

        * - Attribute
          - Description
        * - :attr:`lower_bounds`
          - The ``List[int]`` of lower bounds for each dimension.
        * - :attr:`upper_bounds`
          - The ``List[int]`` of upper bounds for each dimension.
        * - :attr:`data`
          - The ``DimensionalReflectionData`` of the fixed dimension.
        * - :attr:`dim`
          - The ``int`` indicating the fixed dimension.
        * - :attr:`alignment_dims`
          - The ``List[int]`` indicating the one or two alignment dimensions.
        * - :attr:`bounceback_loose_bounds`
          - The ``List[List[bool]]`` indicating whether the comparators should span the dimensions using tight (i.e., :math:`<`) or loose (i.e., :math:`\leq`) bounds.
    """

    def __init__(
        self,
        dim: int,
        lower_bounds: List[int],
        upper_bounds: List[int],
        reflection_data: DimensionalReflectionData,
    ):
        self.num_dims = len(lower_bounds) + 1
        self.dim = dim
        self.alignment_dims: List[int] = [
            dimension for dimension in range(len(lower_bounds) + 1) if dimension != dim
        ]
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.data = reflection_data
        self.bounceback_loose_bounds: List[List[bool]] = (
            self.__get_bounceback_wall_loose_bounds()
        )

    def __get_bounceback_wall_loose_bounds(self):
        # Whether to use comparator bounds (LE, GE - True) or (LT, GT - False)
        # When performing BB reflection on the inside walls.
        # Organized as outer list -> dimension, inner list -> alignment dimensions
        # Many combinations are possible here
        if self.num_dims == 2:
            return [[True], [False]]
        else:
            return [[True, True], [False, False], [False, True]]