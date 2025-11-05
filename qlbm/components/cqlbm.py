"""The end-to-end algorithm of the Collisionless Quantum Lattice Boltzmann Algorithm, or Quantum Transport Method first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`.

This is a common entrypoint that supports implementations based on the :class:`.MSLattice` and :class:`.ABLattice`.

Implementations can be found in the :class:`MSQLBM` and :class:`.ABQLBM`, respectively.
"""

from logging import Logger, getLogger
from time import perf_counter_ns
from typing import cast

from typing_extensions import override

from qlbm.components.ab.ab import ABQLBM
from qlbm.components.base import LBMAlgorithm
from qlbm.components.ms.msqlbm import MSQLBM
from qlbm.lattice import MSLattice
from qlbm.lattice.lattices.ab_lattice import ABLattice
from qlbm.lattice.lattices.base import AmplitudeLattice
from qlbm.tools.exceptions import LatticeException


class CQLBM(LBMAlgorithm):
    """The end-to-end algorithm of the Collisionless Quantum Lattice Boltzmann Algorithm first introduced in :cite:t:`collisionless` and later extended in :cite:t:`qmem`.

    Implementations based on lattices with the DdQq discretization use the :class:`.ABQLBM`:

    .. plot::
        :include-source:

        from qlbm.components import CQLBM
        from qlbm.lattice import ABLattice

        lattice = ABLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": "d2q9"},
                "geometry": [],
            }
        )

        CQLBM(lattice).draw("mpl")

    Implementations where the number of velocities is defined per dimension delegate to the :class:`.MSQLBM`.

    .. plot::
        :include-source:

        from qlbm.components import CQLBM
        from qlbm.lattice import MSLattice

        lattice = MSLattice(
            {
                "lattice": {"dim": {"x": 4, "y": 4}, "velocities": {"x": 4, "y": 4}},
                "geometry": [],
            }
        )

        CQLBM(lattice).draw("mpl")

    """

    def __init__(
        self,
        lattice: AmplitudeLattice,
        logger: Logger = getLogger("qlbm"),
    ) -> None:
        super().__init__(lattice, logger)
        self.lattice: AmplitudeLattice = lattice

        self.logger.info(f"Creating circuit {str(self)}...")
        circuit_creation_start_time = perf_counter_ns()
        self.circuit = self.create_circuit()
        self.logger.info(
            f"Creating circuit {str(self)} took {perf_counter_ns() - circuit_creation_start_time} (ns)"
        )

    @override
    def create_circuit(self):
        if isinstance(self.lattice, MSLattice):
            return MSQLBM(cast(MSLattice, self.lattice), self.logger)
        elif isinstance(self.lattice, ABLattice):
            return ABQLBM(cast(ABLattice, self.lattice), self.logger)
        else:
            raise LatticeException(
                f"CQLBM does not support lattices of type {type(self.lattice)}"
            )

    @override
    def __str__(self) -> str:
        return f"[Algorithm CQLBM with lattice {self.lattice}]"
