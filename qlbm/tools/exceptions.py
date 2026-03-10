"""Contains custom exceptions for the QLBM package."""


class LatticeException(Exception):
    """Exception raised when encountering invalid or misaligned lattice properties."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ResultsException(Exception):
    """Exception raised during the processing of :class:`QBMResults` objects."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class CompilerException(Exception):
    """Exception raised when encountering a circuit compilation exception."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class CircuitException(Exception):
    """Exception raised when attempting to compile to an unsupported target."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ExecutionException(Exception):
    """Exception raised when attempting to execute circuits with mismatched properties."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
