from qlbm.components.common import EQCCollisionOperator
from qlbm.lattice import LatticeDiscretization

# Select a discretization and draw its circuit
EQCCollisionOperator(
    LatticeDiscretization.D3Q6
).draw("mpl")