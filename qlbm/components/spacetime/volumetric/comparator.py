# from logging import Logger, getLogger
# from time import perf_counter_ns
# from typing import List, Tuple

# from qlbm.components.base import LBMPrimitive
# from qlbm.components.collisionless.primitives import Comparator, ComparatorMode
# from qlbm.lattice.lattices.spacetime_lattice import SpaceTimeLattice


# class PeriodicVolumeComparator(LBMPrimitive):
#     def __init__(
#         self,
#         lattice: SpaceTimeLattice,
#         cuboid_bounds: List[Tuple[int, int]],
#         logger: Logger = getLogger("qlbm"),
#     ):
#         super().__init__(logger)

#         self.lattice = lattice
#         self.cuboid_bounds = cuboid_bounds

#         self.logger.info(f"Creating circuit {str(self)}...")
#         circuit_creation_start_time = perf_counter_ns()
#         self.circuit = self.create_circuit()
#         self.logger.info(
#             f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
#         )

#     def create_circuit(self):
#         circuit = self.lattice.circuit.copy()
#         circuit.h(self.lattice.grid_index())

#         lb_comparator_circuits = [
#             Comparator(
#                 self.lattice.num_gridpoints[dim].bit_length() + 1,
#                 self.cuboid_bounds[dim][0],
#                 ComparatorMode.GE,
#                 logger=self.logger,
#             ).circuit
#             for dim in range(self.lattice.num_dims)
#         ]

#         ub_comparator_circuits = [
#             Comparator(
#                 self.lattice.num_gridpoints[dim].bit_length() + 1,
#                 self.cuboid_bounds[dim][1],
#                 ComparatorMode.LE,
#                 logger=self.logger,
#             ).circuit
#             for dim in range(self.lattice.num_dims)
#         ]

        

#         circuit.compose(self.set_grid_value(grid_point_data[0]), inplace=True)
#         circuit.compose(
#             MCMT(
#                 XGate(),
#                 num_ctrl_qubits=self.lattice.properties.get_num_grid_qubits(),
#                 num_target_qubits=sum(
#                     grid_point_data[1]
#                 ),  # The sum is equal to the number of velocities set to true
#             ),
#             qubits=list(
#                 self.lattice.grid_index()
#                 + flatten(
#                     [
#                         self.lattice.velocity_index(0, c)
#                         for c, is_velocity_enabled in enumerate(grid_point_data[1])
#                         if is_velocity_enabled
#                     ]
#                 )
#             ),
#             inplace=True,
#         )
#         circuit.compose(self.set_grid_value(grid_point_data[0]), inplace=True)

#         return circuit