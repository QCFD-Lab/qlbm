"""Encoding utilities for amplitude-based lattices and components."""

from matplotlib.pylab import Enum


class ABEncodingType(Enum):
    r"""Enumerator for the kinds of encodings under the Amplitude-Based encoding umbrella.

    The modes are as follows:

    * (1, ``ABEncodingType.AB``, The regular AB encoding.);
    * (2, ``ABEncodingType.OH``, The one-hot encoding.);
    * (3, ``ABEncodingType.MS``, The multi-speed encoding.).
    """

    AB = (1, )
    OH = (2, )
    MS = (3, )
