from qlbm.components.common import EQCPermutation
from qlbm.lattice import LatticeDiscretization
from qlbm.lattice.eqc import EquivalenceClassGenerator

# Generate some equivalence classes
eqcs = EquivalenceClassGenerator(
    LatticeDiscretization.D3Q6
).generate_equivalence_classes()

# Select one at random and draw its circuit
EQCPermutation(eqcs.pop(), inverse=False).circuit.draw("mpl")