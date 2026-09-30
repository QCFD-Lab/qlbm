from qlbm.components.common import TruncatedQFT

TruncatedQFT(4, 5).circuit.decompose(reps=2).draw("mpl")