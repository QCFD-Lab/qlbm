from qlbm.components.common.comparators import SingleRegisterComparator
from qlbm.tools.utils import ComparatorMode

# On a 5 qubit register, compare the number 3
SingleRegisterComparator(num_qubits=5,
                        num_to_compare=3,
                        mode=ComparatorMode.LT).draw("mpl")