# """Tests for the ABZoneAgnosticSRCheck primitive.

# Verifies that after the main oracle sets a_o=1, the SR check correctly
# identifies which dimension(s) caused a diagonal particle to enter the obstacle.
# """

# import pytest
# from qiskit import QuantumCircuit, transpile
# from qiskit.quantum_info import Statevector
# from qiskit_aer import AerSimulator

# from qlbm.components.ab.reflection.agnosotic_reflection import (
#     ABZoneAgnosticReflectionOracle,
#     ABZoneAgnosticSRCheck,
# )
# from qlbm.lattice import ABLattice

# _SIMULATOR = AerSimulator(method="statevector")


# def _simulate_statevector(circuit: QuantumCircuit) -> Statevector:
#     """Run a circuit on AerSimulator and return the final statevector."""
#     qc = circuit.copy()
#     qc.save_statevector()
#     tqc = transpile(qc, _SIMULATOR, optimization_level=0)
#     result = _SIMULATOR.run(tqc).result()
#     return result.data(0)["statevector"]


# def _make_lattice() -> ABLattice:
#     """8x8 lattice with a 2x2 obstacle at [3,4] x [3,4]."""
#     return ABLattice(
#         {
#             "lattice": {"dim": {"x": 8, "y": 8}, "velocities": "d2q9"},
#             "geometry": [
#                 {
#                     "shape": "cuboid",
#                     "x": [3, 6],
#                     "y": [3, 6],
#                     "boundary": "specular",
#                 }
#             ],
#         }
#     )


# def _encode_and_stream(lattice: ABLattice, x: int, y: int, v: int) -> QuantumCircuit:
#     """Encode a basis state |x, y, v> — the state AFTER streaming.

#     This simulates a particle that has already been streamed into position (x, y)
#     with velocity v. We set a_o=1 manually to represent that the main oracle
#     has already detected the particle is inside the obstacle.
#     """
#     circuit = lattice.circuit.copy()

#     for i, q in enumerate(lattice.grid_index(0)):
#         if (x >> i) & 1:
#             print(f"X{q}=gx[{i}]")
#             circuit.x(q)

#     for i, q in enumerate(lattice.grid_index(1)):
#         if (y >> i) & 1:
#             print(f"X{q}=gy[{i}]")
#             circuit.x(q)

#     for i, q in enumerate(lattice.velocity_index()):
#         if (v >> i) & 1:
#             print(f"X{q}=v[{i}]")
#             circuit.x(q)

#     # Set a_o = 1 (simulating that the main oracle has fired)
#     circuit.x(lattice.ancillae_obstacle_index(0))

#     return circuit


# def _get_ancilla_values(lattice: ABLattice, sv: Statevector) -> dict:
#     """Extract the obstacle ancilla qubit values from a statevector.

#     Returns dict with keys 'a_o', 'a_x', 'a_y' mapping to 0 or 1.
#     """
#     obstacle_indices = lattice.ancillae_obstacle_index()
#     # Find the single nonzero amplitude
#     for i, amp in enumerate(sv.data):
#         if abs(amp) ** 2 > 0.5:
#             a_o = (i >> obstacle_indices[0]) & 1
#             a_x = (i >> obstacle_indices[1]) & 1
#             a_y = (i >> obstacle_indices[2]) & 1
#             a_xy = (i >> obstacle_indices[2]) & 1
#             return {"a_o": a_o, "a_x": a_x, "a_y": a_y, "a_xy": a_xy}
#     raise ValueError("No dominant amplitude found in statevector")


# class TestSRCheckDimensionIdentification:
#     """Tests that ABZoneAgnosticSRCheck correctly identifies which dimension caused entry."""

#     def test_v5_hits_left_wall_ax1_ay0(self):
#         """Velocity 5 [+1,+1] hitting a left wall: x caused entry, not y."""
#         lattice = _make_lattice()
#         shapes = lattice.shapes["specular"]

#         prep = _encode_and_stream(lattice, x=4, y=4, v=5)

#         sr_check = ABZoneAgnosticSRCheck(
#             lattice, lattice.discretization, shapes, check_negative_direction=True
#         )
#         prep.compose(sr_check.circuit, inplace=True)

#         sv = _simulate_statevector(prep)
#         vals = _get_ancilla_values(lattice, sv)

#         print(vals)

#         assert vals["a_o"] == 1, "a_o should remain 1"
#         assert vals["a_x"] == 1, "a_x should be 0"
#         assert vals["a_y"] == 0, "a_y should be 0"
#         assert vals["a_xy"] == 0, "a_xy should be 1"

#     def test_v5_hits_bottom_wall_ax0_ay1(self):
#         """Velocity 5 [+1,+1] hitting a bottom wall: y caused entry, not x.

#         Obstacle at [3,4] x [3,4]. Particle at (4, 3) with v=5.
#         Pre-stream position was (3, 2). Unstream x -> (3, 3): inside -> a_x=0.
#         Unstream y -> (4, 2): outside -> a_y=1.
#         """
#         lattice = _make_lattice()
#         shapes = lattice.shapes["specular"]

#         prep = _encode_and_stream(lattice, x=4, y=3, v=5)

#         sr_check = ABZoneAgnosticSRCheck(lattice, lattice.discretization, shapes)
#         prep.compose(sr_check.circuit, inplace=True)

#         sv = _simulate_statevector(prep)
#         vals = _get_ancilla_values(lattice, sv)

#         assert vals["a_o"] == 1, "a_o should remain 1"
#         assert vals["a_x"] == 0, "a_x should be 0 (x did not cause entry)"
#         assert vals["a_y"] == 1, "a_y should be 1 (y caused entry)"

#     def test_v5_hits_corner_ax1_ay1(self):
#         """Velocity 5 [+1,+1] hitting a corner: both dims caused entry.

#         Obstacle at [3,4] x [3,4]. Particle at (3, 3) with v=5.
#         Pre-stream position was (2, 2). Unstream x -> (2, 3): outside -> a_x=1.
#         Unstream y -> (3, 2): outside -> a_y=1.
#         """
#         lattice = _make_lattice()
#         shapes = lattice.shapes["specular"]

#         prep = _encode_and_stream(lattice, x=3, y=3, v=5)

#         sr_check = ABZoneAgnosticSRCheck(lattice, lattice.discretization, shapes)
#         prep.compose(sr_check.circuit, inplace=True)

#         sv = _simulate_statevector(prep)
#         vals = _get_ancilla_values(lattice, sv)

#         assert vals["a_o"] == 1, "a_o should remain 1"
#         assert vals["a_x"] == 1, "a_x should be 1 (x caused entry)"
#         assert vals["a_y"] == 1, "a_y should be 1 (y caused entry)"
#         assert vals["a_xy"] == 0, "a_y should be 1 (y caused entry)"

#     def test_v6_hits_right_wall_ax1_ay0(self):
#         """Velocity 6 [-1,+1] hitting a right wall: x caused entry, not y.

#         Obstacle at [3,4] x [3,4]. Particle at (4, 4) with v=6.
#         Pre-stream position was (5, 3). Unstream x -> (5, 4): outside -> a_x=1.
#         Unstream y -> (4, 3): inside -> a_y=0.
#         """
#         lattice = _make_lattice()
#         shapes = lattice.shapes["specular"]

#         prep = _encode_and_stream(lattice, x=4, y=4, v=6)

#         sr_check = ABZoneAgnosticSRCheck(lattice, lattice.discretization, shapes)
#         prep.compose(sr_check.circuit, inplace=True)

#         sv = _simulate_statevector(prep)
#         vals = _get_ancilla_values(lattice, sv)

#         assert vals["a_o"] == 1, "a_o should remain 1"
#         assert vals["a_x"] == 1, "a_x should be 1 (x caused entry)"
#         assert vals["a_y"] == 0, "a_y should be 0 (y did not cause entry)"

#     def test_v7_hits_top_wall_ax0_ay1(self):
#         """Velocity 7 [-1,-1] hitting a top wall: y caused entry, not x.

#         Obstacle at [3,4] x [3,4]. Particle at (3, 4) with v=7.
#         Pre-stream position was (4, 5). Unstream x -> (4, 4): inside -> a_x=0.
#         Unstream y -> (3, 5): outside -> a_y=1.
#         """
#         lattice = _make_lattice()
#         shapes = lattice.shapes["specular"]

#         prep = _encode_and_stream(lattice, x=3, y=4, v=7)

#         sr_check = ABZoneAgnosticSRCheck(lattice, lattice.discretization, shapes)
#         prep.compose(sr_check.circuit, inplace=True)

#         sv = _simulate_statevector(prep)
#         vals = _get_ancilla_values(lattice, sv)

#         assert vals["a_o"] == 1, "a_o should remain 1"
#         assert vals["a_x"] == 0, "a_x should be 0 (x did not cause entry)"
#         assert vals["a_y"] == 1, "a_y should be 1 (y caused entry)"

#     def test_v8_hits_left_wall_ax1_ay0(self):
#         """Velocity 8 [+1,-1] hitting a left wall: x caused entry, not y.

#         Obstacle at [3,4] x [3,4]. Particle at (3, 3) with v=8.
#         Pre-stream position was (2, 4). Unstream x -> (2, 3): outside -> a_x=1.
#         Unstream y -> (3, 4): inside -> a_y=0.
#         """
#         lattice = _make_lattice()
#         shapes = lattice.shapes["specular"]

#         prep = _encode_and_stream(lattice, x=3, y=3, v=8)

#         sr_check = ABZoneAgnosticSRCheck(lattice, lattice.discretization, shapes)
#         prep.compose(sr_check.circuit, inplace=True)

#         sv = _simulate_statevector(prep)
#         vals = _get_ancilla_values(lattice, sv)

#         assert vals["a_o"] == 1, "a_o should remain 1"
#         assert vals["a_x"] == 1, "a_x should be 1 (x caused entry)"
#         assert vals["a_y"] == 0, "a_y should be 0 (y did not cause entry)"


#     def test_sr_check_is_self_inverse(self):
#         """Applying the SR check twice should return to the original state."""
#         lattice = _make_lattice()
#         shapes = lattice.shapes["specular"]

#         prep = _encode_and_stream(lattice, x=3, y=4, v=5)

#         sr_check = ABZoneAgnosticSRCheck(lattice, lattice.discretization, shapes)

#         prep.compose(sr_check.circuit, inplace=True)
#         prep.compose(sr_check.circuit.inverse(), inplace=True)

#         sv = _simulate_statevector(prep)
#         vals = _get_ancilla_values(lattice, sv)

#         assert vals["a_o"] == 1, "a_o should be 1 (back to initial)"
#         assert vals["a_x"] == 0, "a_x should be 0 (uncomputed)"
#         assert vals["a_y"] == 0, "a_y should be 0 (uncomputed)"
