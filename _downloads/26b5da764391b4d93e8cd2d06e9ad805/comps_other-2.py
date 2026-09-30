from qlbm.components.common.comparators import TwoRegisterComparator
from qlbm.tools import ComparatorMode

# Compare two registers of size 4
TwoRegisterComparator(num_qubits=4, mode=ComparatorMode.LT).draw("mpl")