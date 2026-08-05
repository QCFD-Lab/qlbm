from qlbm.components.common import EQCRedistribution
from qlbm.lattice import LatticeDiscretization
from qlbm.lattice.eqc import EquivalenceClassGenerator

# Generate some equivalence classes
eqcs = EquivalenceClassGenerator(
    LatticeDiscretization.D3Q6
).generate_equivalence_classes()

# Select one at random and draw its decomposed circuit
EQCRedistribution(eqcs.pop(), decompose_block=True).circuit.draw("mpl")